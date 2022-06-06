# -----------------------------------------------------------------------------
# Miscs: dataprocessing.py
# -----------------------------------------------------------------------------
# Description: Preprocess data for training and testing
# -----------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
from collections import defaultdict

import numpy as np
import numpy.random as random
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from nltk.tokenize import RegexpTokenizer
from torch.autograd import Variable

from miscs.config import cfg

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import json

# -----------------------------------------------------------------------------
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Function definitions
# -----------------------------------------------------------------------------
def preprocessData(data):
    """
    Function to process the image data and return the final form of data
    """
    images, captions, capLengths, _, keys = data
    # Sort data in decreasing order of caption lengths
    sortedCapLengths, sortedCapIndices = torch.sort(capLengths, 0, True)

    realImages = []
    for i in range(len(images)):
        images[i] = images[i][sortedCapIndices]
        if cfg.CUDA:
            realImages.append(Variable(images[i]).to(device))
        else:
            realImages.append(Variable(images[i]))

    captions = captions[sortedCapIndices].squeeze()
    keys = [keys[i] for i in sortedCapIndices.numpy()]

    if cfg.CUDA:
        captions = Variable(captions).to(device)
        sortedCapLengths = Variable(sortedCapLengths).to(device)
    else:
        captions = Variable(captions)
        sortedCapLengths = Variable(sortedCapLengths)

    return [realImages, captions, sortedCapLengths, keys]


def acquireImages(imagePath, imageSize, bbox=None,
                  transform=None, normalize=None):
    image = Image.open(imagePath).convert('RGB')
    width, height = image.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        image = image.crop([x1, y1, x2, y2])

    if transform is not None:
        image = transform(image)

    returnImage = []
    if cfg.GAN.B_DCGAN:
        returnImage = [normalize(image)]
    else:
        for i in range(cfg.TREE.BRANCH_NUM):
            if i < (cfg.TREE.BRANCH_NUM - 1):
                reImage = transforms.Scale(imageSize[i])(image)
            else:
                reImage = image
            returnImage.append(normalize(reImage))

    return returnImage


class TextProcessing(data.Dataset):
    def __init__(self, pathOfDataDir, split='train',
                 baseSize=64,
                 transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.targetTransform = target_transform
        self.numOfEmbeddings = cfg.TEXT.CAPTIONS_PER_IMAGE

        self.imageSize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imageSize.append(baseSize)
            baseSize = baseSize * 2
        self.data = []
        self.dataDirectory = pathOfDataDir

        self.bbox = None
        splitDirectory = os.path.join(pathOfDataDir, split)

        self.fileNames, self.captions, self.idxToWords, \
        self.wordToIdx, self.numOfWords = self.loadTextData(pathOfDataDir, split)
        self.nameToCaptionDict = {}

        self.IdClass = self.loadClassIdx(splitDirectory, len(self.fileNames['img']))
        self.number_example = len(self.fileNames['img'])

    def loadBoundryBox(self):
        """
        Load bounding box information from file
        :return: bounding box fileNames and bounding box information
        """
        dataDirectory = self.dataDirectory
        boundryBoxPath = os.path.join(dataDirectory, 'bounding_boxes.txt')
        boundryBoxDataframe = pd.read_csv(boundryBoxPath,
                                          delim_whitespace=True,
                                          header=None).astype(int)
        #
        filePath = os.path.join(dataDirectory, 'images.txt')
        dataframeFileNames = \
            pd.read_csv(filePath, delim_whitespace=True, header=None)
        fileNames = dataframeFileNames[1].tolist()
        print('Total number of fileNames: ', len(fileNames), fileNames[0])
        #
        boundryBoxFileName = {fileName[:-4]: [] for fileName in fileNames}
        numOfImages = len(fileNames)
        for i in range(0, numOfImages):
            boundryBox = boundryBoxDataframe.iloc[i][1:].tolist()
            key = fileNames[i][:-4]
            boundryBoxFileName[key] = boundryBox
        return boundryBoxFileName

    def loadCaptionsInfo(self, data_dir, fileNames):
        allCaptions = []
        for i in range(len(fileNames['img'])):
            captionsPath = '%s/%s.txt' % ('/home/yesenmao/dataset/flower/jpg_text/', fileNames['img'][i])
            with open(captionsPath, "r") as f:
                captions = f.read().split('\n')
                count = 0  # count the number of captions
                for caption in captions:
                    if len(caption) == 0:
                        continue
                    caption = caption.replace("\ufffd\ufffd", " ")
                    # Only get alphanumeric characters
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(caption.lower())
                    if len(tokens) == 0:
                        caption = 'this flower'
                        caption = caption.replace("\ufffd\ufffd", " ")
                        tokenizer = RegexpTokenizer(r'\w+')
                        tokens = tokenizer.tokenize(caption.lower())
                        print('caption', caption)

                    newTokens = []
                    for token in tokens:
                        token = token.encode('ascii', 'ignore').decode('ascii')
                        if len(token) > 0:
                            newTokens.append(token)
                    allCaptions.append(newTokens)
                    count += 1
                    if count == self.numOfEmbeddings:
                        break
                if count < self.numOfEmbeddings:
                    print('ERROR MSG: the captions for %s less than %d'
                          % (fileNames['img'][i], count))
        return allCaptions

    def createDictionary(self, trainCaptions, testCaptions):
        """
        Create the dictionary for the captions
        :param trainCaptions: training captions
        :param testCaptions: testing captions
        :return: dictionary
        """
        wordCounts = defaultdict(float)
        captions = trainCaptions + testCaptions
        for caption in captions:
            for word in caption:
                wordCounts[word] += 1

        vocab = [w for w in wordCounts if wordCounts[w] >= 0]

        idxToWord = {}
        idxToWord[0] = '<end>'
        wordToIdx = {}
        wordToIdx['<end>'] = 0
        idx = 1
        for word in vocab:
            wordToIdx[word] = idx
            idxToWord[idx] = word
            idx += 1

        newTrainCaptions = []
        for caption in trainCaptions:
            rev = []
            for word in caption:
                if word in wordToIdx:
                    rev.append(wordToIdx[word])
            newTrainCaptions.append(rev)

        newTestCaptions = []
        for caption in testCaptions:
            rev = []
            for word in caption:
                if word in wordToIdx:
                    rev.append(wordToIdx[word])
            newTestCaptions.append(rev)

        return [newTrainCaptions, newTestCaptions,
                idxToWord, wordToIdx, len(idxToWord)]

    def loadTextData(self, dataDirectory, split):
        """
        Load a dataset that contains text file(s) with pre-processed text sequences.
        :param dataDirectory: string. Directory of the data
        :param split: string. train/test/val
        :return: fileNames
        :return: captions
        :return: idxToWord dictionary
        :return: wordToIdx dictionary
        :return: number of words in dictionary
        """
        filePath = os.path.join(dataDirectory, 'captions.pickle')
        testNames = self.loadFilenames(dataDirectory, 'test')
        trainNames = self.loadFilenames(dataDirectory, 'train')

        if not os.path.isfile(filePath):
            trainCaptions = self.loadCaptionsInfo(dataDirectory, trainNames)
            testCaptions = self.loadCaptionsInfo(dataDirectory, testNames)

            trainCaptions, testCaptions, idxToWord, wordToIdx, numOfWords = \
                self.createDictionary(trainCaptions, testCaptions)
            with open(filePath, 'wb') as f:
                pickle.dump([trainCaptions, testCaptions, idxToWord, wordToIdx], f, protocol=2)
        else:
            with open(filePath, 'rb') as f:
                z = pickle.load(f)
                trainCaptions, testCaptions = z[0], z[1]
                idxToWord, wordToIdx = z[2], z[3]
                del z
                numOfWords = len(idxToWord)
        if split == 'train':
            captions = trainCaptions
            fileNames = trainNames
        else:  # split=='test'
            captions = testCaptions
            fileNames = testNames
        return fileNames, captions, idxToWord, wordToIdx, numOfWords

    def loadClassIdx(self, data_dir, total_num):
        with open('./data/classToName.json', 'r') as f:
            cat_to_name = json.load(f)
        dic_class = []
        dic_classs = {}
        for key, value in cat_to_name.items():
            dic_class.append(value)
        for i in range(len(dic_class)):
            dic_classs[dic_class[i]] = i

        return dic_classs

    def loadFilenames(self, data_dir, split):
        filepath = './data/flowers_dictionary.pkl'
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load fileNames from: %s (%d)' % (filepath, len(filenames['img'])))
        else:
            filenames = []
        return filenames

    def readCaption(self, idx):
        captionSent = np.asarray(self.captions[idx]).astype('int64')
        if (captionSent == 0).sum() > 0:
            print('ERROR: do not need END (0) token', captionSent)
        numOfWords = len(captionSent)
        caption = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        lengthOfCaption = numOfWords
        if numOfWords <= cfg.TEXT.WORDS_NUM:
            caption[:numOfWords, 0] = captionSent
        else:
            idx = list(np.arange(numOfWords))  # 1, 2, 3,..., maxNum
            np.random.shuffle(idx)
            idx = idx[:cfg.TEXT.WORDS_NUM]
            idx = np.sort(idx)
            caption[:, 0] = captionSent[idx]
            lengthOfCaption = cfg.TEXT.WORDS_NUM
        return caption, lengthOfCaption

    def __getitem__(self, index):
        #
        key = self.fileNames['img'][index]
        cat = self.fileNames['cat'][index]
        classId = self.IdClass[cat]
        #
        boundryBox = None

        imageName = '%s/jpg/%s.jpg' % (self.dataDirectory, key)
        images = acquireImages(imageName, self.imageSize,
                               boundryBox, self.transform, normalize=self.norm)

        # random select a sentence
        sentIdx = random.randint(0, self.numOfEmbeddings)
        sentIdxNew = index * self.numOfEmbeddings + sentIdx
        captions, captionLengths = self.readCaption(sentIdxNew)

        return images, captions, captionLengths, classId, key

    def __len__(self):
        return len(self.fileNames['img'])
