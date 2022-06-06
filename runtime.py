from __future__ import print_function

import argparse
import datetime
import os
import pprint
import random
import sys
from itertools import zip_longest

import dateutil.tz
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from torch.backends import cudnn

from miscs.config import cfg, readConfigFile
from miscs.dataprocessing import TextProcessing
from miscs.dataprocessing import preprocessData
from miscs.utils import mkdirWithPath
from models.DAMSM import TextEncoderRNN, CustomLSTM
from models.model import NetG, NetD

# imort pickle to load the captions pkl file
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

savePath = (os.path.abspath(os.path.join(os.path.realpath(__file__), 'code')))
sys.path.append(savePath)

# To enable multiprocessing
import multiprocessing

multiprocessing.set_start_method('spawn', True)


def parseCommandLineArgumnets():
    """
    Function to parse X arguments
    """

    argumnetParser = argparse.ArgumentParser(description='Parser to parse X arguments to the script')
    argumnetParser.add_argument('--manualSeed', type=int, help='seed for the random number generator', default=0)
    argumnetParser.add_argument('--dataDirectory', type=str, help='path to the data directory', default='data')
    argumnetParser.add_argument('--cfg', dest='cfg_file',
                                help='optional config file',
                                default='config_test.yml', type=str)
    argumnetParser.add_argument('--pathOfDataDir', dest='pathOfDataDir', type=str, default='')
    argumnetParser.add_argument('--gpu', dest='gpu_id', type=int, default=0, help='GPU device id to use')
    args = argumnetParser.parse_args()
    return args


def test(txtEncoder, netGen, dataLoader, device):
    """
    Function to get the samples from the generator for all test data
    :param txtEncoder: Text Encoder
    :param netGen: Generator
    :param dataLoader: Data Loader
    :param device: Device to use
    :return: generated fake images
    """

    modelPath = cfg.TRAIN.NET_G
    splitDirectory = 'valid'
    # Set the model to eval mode
    netGen.eval()

    bathcSize = cfg.TRAIN.BATCH_SIZE
    tempPath = modelPath
    saveDirectory = '%s/%s' % (tempPath, splitDirectory)
    mkdirWithPath(saveDirectory)
    count = 0
    counter = 0
    for i in range(10):
        for step, data in enumerate(dataLoader, 0):
            images, captions, capLengths, keys = preprocessData(data)
            count += bathcSize

            if count > 5*bathcSize:
                break

            hidden = txtEncoder.initializeHidden(bathcSize)

            wordEmbeddings, sentEmbeddings = txtEncoder(captions, capLengths, hidden)
            wordEmbeddings, sentEmbeddings = wordEmbeddings.detach(), sentEmbeddings.detach()

            # Generate fake images
            with torch.no_grad():
                noise = torch.randn(bathcSize, 100)
                noise = noise.to(device)
                netGen.lstm.initializeHidden(noise)

                fakeImages = netGen(noise, sentEmbeddings)
            for j in range(bathcSize):
                tempPath = '%s/single/%s' % (saveDirectory, keys[j])
                folder = tempPath[:tempPath.rfind('/')]
                if not os.path.isdir(folder):
                    mkdirWithPath(folder)  # create folder
                image = fakeImages[j].data.cpu().numpy()
                # map images to [0,255]
                image = (image + 1.0) * 127.5
                image = image.astype(np.uint8)
                image = np.transpose(image, (1, 2, 0))
                image = Image.fromarray(image)
                counter+=1
                fullpath = ('./imgs/image_%d.png' % counter)
                image.save(fullpath)


def train(dataloader, netGen, netDisc, txtEncoder, optimizerGen, optimizerDisc, currentEpoch, batchSize, device):
    """
    Function to Train the model
    :param dataloader: DataLoader
    :param netGen: Generator
    :param netDisc: Discriminator
    :param txtEncoder: TextEncoder
    :param optimizerGen: Optimizer for Generator
    :param optimizerDisc: Optimizer for Discriminator
    :param currentEpoch: Current epoch number to resume training
    :param batchSize: Batch size
    :param device: Device to use
    """

    mkdirWithPath('../models/%s' % cfg.CONFIG_NAME)

    for epoch in range(currentEpoch + 1, cfg.TRAIN.MAX_EPOCH + 1):
        torch.cuda.empty_cache()

        for step, data in enumerate(dataloader, 0):
            realImages, captions, capLengths, keys = preprocessData(data)
            hidden = txtEncoder.initializeHidden(batchSize)

            wordEmbeddings, sentEmbeddings = txtEncoder(captions, capLengths, hidden)
            wordEmbeddings, sentEmbeddings = wordEmbeddings.detach(), sentEmbeddings.detach()

            realImages = realImages[0].to(device)
            realFeatures = netDisc(realImages)
            output = netDisc.COND_DNET(realFeatures, sentEmbeddings)
            errorDiscReal = torch.nn.ReLU()(1.0 - output).mean()

            output = netDisc.COND_DNET(realFeatures[:(batchSize - 1)], sentEmbeddings[1:batchSize])
            errorDiscMismatch = torch.nn.ReLU()(1.0 + output).mean()

            # Generate fake realImages
            noise = torch.randn(batchSize, 100)  # Noise X
            noise = noise.to(device)  # Send to GPU
            netGen.lstm.initializeHidden(noise)  # Initialize hidden state of LSTM

            # Pass through generator
            fakeImages = netGen(noise, sentEmbeddings)

            # Discriminator sees generated fake realImages
            fakeFeatures = netDisc(fakeImages.detach())

            # Discriminator determines validity of fake realImages
            errorDiscFake = netDisc.COND_DNET(fakeFeatures, sentEmbeddings)
            errorDiscFake = torch.nn.ReLU()(1.0 + errorDiscFake).mean()

            # Gradient penalty
            errorDisc = errorDiscReal + (errorDiscFake + errorDiscMismatch) / 2.0  # Total discriminator loss
            optimizerDisc.zero_grad()
            optimizerGen.zero_grad()
            errorDisc.backward()
            optimizerDisc.step()

            # Generator tries to fool the discriminator
            interpolated = (realImages.data).requires_grad_()
            sentInter = (sentEmbeddings.data).requires_grad_()
            features = netDisc(interpolated)
            out = netDisc.COND_DNET(features, sentInter)
            grads = torch.autograd.grad(outputs=out,
                                        inputs=(interpolated, sentInter),
                                        grad_outputs=torch.ones(out.size()).to(device),
                                        retain_graph=True,
                                        create_graph=True,
                                        only_inputs=True)
            grad0 = grads[0].view(grads[0].size(0), -1)
            grad1 = grads[1].view(grads[1].size(0), -1)
            grad = torch.cat((grad0, grad1), dim=1)
            gradL2Norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            discLoss_gp = torch.mean((gradL2Norm) ** 6)
            disc_loss = 2.0 * discLoss_gp
            optimizerDisc.zero_grad()
            optimizerGen.zero_grad()
            disc_loss.backward()
            optimizerDisc.step()

            # Generator update
            features = netDisc(fakeImages)
            output = netDisc.COND_DNET(features, sentEmbeddings)
            errorGen = - output.mean()
            optimizerGen.zero_grad()
            optimizerDisc.zero_grad()
            errorGen.backward()
            optimizerGen.step()

            # Logging training progress to console
            print('[%d/%d][%d/%d] Loss_Disc: %.4f Loss_Gen %.4f'
                  % (epoch, cfg.TRAIN.MAX_EPOCH, step, len(dataloader), errorDisc.item(), errorGen.item()))

        # Save losses to log file for every epoch
        if epoch % 1 == 0:
            logsPath = '../logs'  # Path to save logs
            if not os.path.exists(logsPath):
                os.makedirs(logsPath)  # Create folder if not exist
            with open(logsPath + '/hist_D.txt', 'ab') as f:
                np.savetxt(f, np.array(errorDisc.item()).mean().reshape(1, 1))
            with open(logsPath + '/hist_G.txt', 'ab') as f:
                np.savetxt(f, np.array(errorGen.item()).mean().reshape(1, 1))

        vutils.save_image(fakeImages.data, '%s/fake_samples_epoch_%03d.png' % ('../imgs', epoch),
                          normalize=True)

        # Save models checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(netGen.state_dict(), os.path.join(cfg.MODELS_DIR, 'netG_%03d.pth' % epoch))
            torch.save(netDisc.state_dict(), os.path.join(cfg.MODELS_DIR, 'netD_%03d.pth' % epoch))

    return count


def text2Images(captionList, numOfSamples, netGen, textEncoder):
    """
    Generate fake images from captions
    :param captionList: List of captions to generate images from
    :param numOfSamples: number of images to generate for each caption in the list
    :param netGen: Generator network to use
    :param textEncoder: Text encoder network to use
    :return: list of generated images for each caption
    """
    saveDirectory = './imgs'  # Path to save images
    if not os.path.isdir(saveDirectory):
        mkdirWithPath(saveDirectory)  # Create folder if not exist

    # Load word to index dictionary from file
    with open('./data/captions.pickle', 'rb') as f:
        data = pickle.load(f)
        wordToIdx = data[3]
        del data

    # Make fileNames for generated images
    fileNames = []
    for i in range(len(captionList)):
        fileName = captionList[i].split()
        fileName = '_'.join(fileName)
        fileNames.append(fileName)

    # Set model to eval mode
    netGen.eval()

    batchSize = torch.tensor(len(captionList), dtype=torch.int64).reshape(-1)
    hidden = textEncoder.initializeHidden(batchSize)

    # Convert words in the captions to indices
    for i in range(len(captionList)):
        dummyCaption = captionList[i].split(' ')
        captionList[i] = [wordToIdx[word] for word in dummyCaption if word in wordToIdx]

    captions = torch.tensor(list(zip_longest(*captionList, fillvalue=0)), dtype=torch.int64).T
    captionLengths = torch.tensor([len(caption) for caption in captionList], dtype=torch.int64)

    wordEmbeddings, sentEmbeddings = textEncoder(captions, captionLengths, hidden)
    wordEmbeddings, sentEmbeddings = wordEmbeddings.detach(), sentEmbeddings.detach()

    # Generate numOfSamples fake images for each caption
    for i in range(numOfSamples):
        with torch.no_grad():
            noise = torch.randn(batchSize, 100)  # 100-dim noise
            noise = noise.to(device)  # Send to GPU
            netGen.lstm.initializeHidden(noise)  # Initialize hidden state
            genImages = netGen(noise, sentEmbeddings)

        for j in range(batchSize):
            print(f'Generating image {i+1} for caption {j+1}')
            generatedImage = genImages[j].data.cpu().numpy()
            # Map image from [-1, 1] to [0. 255]
            generatedImage = (generatedImage + 1.0) * 127.5
            # Cast image from double to uint8
            generatedImage = generatedImage.astype(np.uint8)
            generatedImage = np.transpose(generatedImage, (1, 2, 0))
            generatedImage = Image.fromarray(generatedImage)

            # Save image
            generatedImage.save('%s/%s_%d.png' % (saveDirectory, fileNames[j], i + 1))


def getModelStats(netGen, netDisc, txtEncoder, dataloader):
    """
    Prints output model information, including number of parameters, and number of trainable parameters
    also, produces a block diagram of the model architecture
    :param netGen: Generator network
    :param netDisc: Discriminator network
    :param txtEncoder: Text encoder network
    :param dataloader: DataLoader object
    """
    # Import required libraries
    from torchviz import make_dot
    from torchinfo import summary

    for step, data in enumerate(dataloader, 0):

        if step > 1:
            break   # Only run once

        realImages, captions, capLengths, keys = preprocessData(data)
        hidden = txtEncoder.initializeHidden(batchSize)

        wordEmbeddings, sentEmbeddings = txtEncoder(captions, capLengths, hidden)
        wordEmbeddings, sentEmbeddings = wordEmbeddings.detach(), sentEmbeddings.detach()

        realImages = realImages[0].to(device)
        realFeatures = netDisc(realImages)
        output = netDisc.COND_DNET(realFeatures, sentEmbeddings)
        errorDiscReal = torch.nn.ReLU()(1.0 - output).mean()

        output = netDisc.COND_DNET(realFeatures[:(batchSize - 1)], sentEmbeddings[1:batchSize])
        errorDiscMismatch = torch.nn.ReLU()(1.0 + output).mean()

        # Generate fake realImages
        noise = torch.randn(batchSize, 100)  # Noise input to generator
        noise = noise.to(device)             # Send to GPU
        netGen.lstm.initializeHidden(noise)  # Initialize hidden state of LSTM

        # Pass through generator
        fakeImages = netGen(noise, sentEmbeddings)

    # make_dot((wordEmbeddings, sentEmbeddings), params=dict(txtEncoder.named_parameters())).render('text_embedding', format='png')
    make_dot(fakeImages, params=dict(netGen.named_parameters())).render('netGen', format='png')
    make_dot(realFeatures, params=dict(netDisc.named_parameters())).render('netDisc', format='png')

    summary(netGen, [noise.shape, sentEmbeddings.shape])
    summary(netDisc, realImages.shape)
    # summary(txtEncoder, [captions.shape, capLengths.shape, hidden.shape])


if __name__ == "__main__":
    args = parseCommandLineArgumnets()
    if args.cfg_file is not None:
        readConfigFile(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        torch.cuda.set_device(cfg.GPU_ID)
        cudnn.benchmark = True
        cfg.GPU_ID = args.gpu_id

    if args.dataDirectory != '':
        cfg.DATA_DIR = args.dataDirectory
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = 100

    # Set random seed
    print("Random seed is set to: ", args.manualSeed)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################
    ############################## DATASET ####################################

    # Get data loader
    imageSize = cfg.TREE.BASE_SIZE
    batchSize = cfg.TRAIN.BATCH_SIZE
    imageTransform = transforms.Compose([
        transforms.Resize(int(imageSize * 76 / 64)),
        transforms.RandomCrop(imageSize),
        transforms.RandomHorizontalFlip()])

    if cfg.TEST >= 1:
        # Testing
        dataset = TextProcessing(cfg.DATA_DIR, 'test',
                                 baseSize=cfg.TREE.BASE_SIZE,
                                 transform=imageTransform)
        print(dataset.numOfWords, dataset.numOfEmbeddings)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batchSize, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))
    else:
        # Training when cfg.TEST == 0
        dataset = TextProcessing(cfg.DATA_DIR, 'train',
                                 baseSize=cfg.TREE.BASE_SIZE,
                                 transform=imageTransform)
        print(dataset.numOfWords, dataset.numOfEmbeddings)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batchSize, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))

    # Check if cuda is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the custom lstm block
    lstm = CustomLSTM(256, 256)

    # Initialize generator and discriminator
    netGen = NetG(cfg.TRAIN.NF, 100, lstm).to(device)
    netDisc = NetD(cfg.TRAIN.NF).to(device)

    # Initialize the text encoder
    textEncoder = TextEncoderRNN(dataset.numOfWords, nhidden=cfg.TEXT.EMBEDDING_DIM)
    # Load the model
    encoderPath = os.path.join(cfg.MODELS_DIR, cfg.TEXT.DAMSM_NAME)
    state_dict = torch.load(encoderPath, map_location=lambda storage, loc: storage)
    textEncoder.load_state_dict(state_dict)
    textEncoder.to(device)

    # Set the text encoder to eval mode and non-trainable as it is already trained
    for p in textEncoder.parameters():
        p.requires_grad = False
    textEncoder.eval()

    # Set current epoch from checkpoint
    # Check the highest epoch from the checkpoints and load the corresponding model
    if os.path.isdir(cfg.MODELS_DIR):
        print(os.listdir(cfg.MODELS_DIR))
        saved_epoch = max([int(f.split('_')[-1].split('.')[0]) for f in os.listdir(cfg.MODELS_DIR) if
                           os.path.isfile(os.path.join(cfg.MODELS_DIR, f)) and 'netG' in f])
        try:
            netGen.load_state_dict(torch.load(os.path.join(cfg.MODELS_DIR, 'netG_%03d.pth' % saved_epoch), map_location=torch.device(device)))
            netDisc.load_state_dict(torch.load(os.path.join(cfg.MODELS_DIR, 'netD_%03d.pth' % saved_epoch), map_location=torch.device(device)))
            print('Loaded the model from epoch %d' % saved_epoch)
        except:
            print(f'No checkpoint found in {cfg.MODELS_DIR}')
            saved_epoch = 0
            print('Initializing the model from scratch')
    else:
        saved_epoch = 0

    # Initialize the optimizers
    optimizerG = torch.optim.Adam(netGen.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optimizerD = torch.optim.Adam(netDisc.parameters(), lr=0.0004, betas=(0.0, 0.9))

    if cfg.TEST == 1:
        count = test(textEncoder, netGen, dataloader, device)  # generate images for the whole valid dataset
        print('Using the model from epoch %d' % saved_epoch)

    elif cfg.TEST == 2:
        print('Using the model from epoch %d' % saved_epoch)
        textList = ['the flower has petals that are lavender with purple filaments and large center with white stigma',
                    'a flower with stringy looking purple and yellow petals and a green and yellow center',
                    'a flower with many white petals and long purple and white stamen at it’s core',
                    'this flower is white and pink in color with petals that are oval shaped']
        # Sort the text list in descending order of length
        textList = sorted(textList, key=len, reverse=True)
        text2Images(textList, 10, netGen, textEncoder)

    elif cfg.TEST == 3:
        # Model Info
        getModelStats(netGen, netDisc, textEncoder, dataloader)

    else:
        # Training
        count = train(dataloader, netGen, netDisc, textEncoder, optimizerG, optimizerD, saved_epoch+1, batchSize, device)
