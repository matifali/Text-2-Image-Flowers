import errno
import os
from copy import deepcopy

import numpy as np
import skimage.transform
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from miscs.config import cfg

COLOR_DIC = {0: [128, 64, 128], 1: [244, 35, 232],
             2: [70, 70, 70], 3: [102, 102, 156],
             4: [190, 153, 153], 5: [153, 153, 153],
             6: [250, 170, 30], 7: [220, 220, 0],
             8: [107, 142, 35], 9: [152, 251, 152],
             10: [70, 130, 180], 11: [220, 20, 60],
             12: [255, 0, 0], 13: [0, 0, 142],
             14: [119, 11, 32], 15: [0, 60, 100],
             16: [0, 80, 100], 17: [0, 0, 230],
             18: [0, 0, 70], 19: [0, 0, 0]}

# Maximum size of font
FONT_MAX = 50


def drawCaption(canvas, captions, idxToWord, visulaizationSize, offset1=2, offset2=2):
    """
    Draw captions on a canvas.
    :param canvas: PIL.Image object
    :param captions: list of str (caption)
    :param idxToWord: dict {idx: word}
    :param visulaizationSize: tuple (width, height)
    :param offset1: int (offset of X)
    :param offset2: int (offset of Y)
    :return: PIL.Image object (with caption)
    """
    num = captions.size(0)
    imageText = Image.fromarray(canvas)
    fontName = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 50)
    d = ImageDraw.Draw(imageText)
    sentenceList = []
    for i in range(num):
        caption = captions[i].data.cpu().numpy()
        sentence = []
        for j in range(len(caption)):
            if caption[j] == 0:
                break
            word = idxToWord[caption[j]].encode('ascii', 'ignore').decode('ascii')
            d.text(((j + offset1) * (visulaizationSize + offset2), i * FONT_MAX), '%d:%s' % (j, word[:6]),
                   font=fontName, fill=(255, 255, 255, 255))
            sentence.append(word)
        sentenceList.append(sentence)
    return imageText, sentenceList


def buildSuperImage(realImages, captions, idxToWord,
                    attentionMaps, attentionSize, lowResImage=None,
                    batchSize=cfg.TRAIN.BATCH_SIZE,
                    numOfMaxWords=cfg.TEXT.WORDS_NUM):
    """
    Function to build a super image from low resolution image,
    :param realImages: batchSize X 3 X H X W
    :param captions: batchSize X numOfMaxWords
    :param idxToWord: word index to word
    :param attentionMaps: batchSize X numOfMaxWords X H X W
    :param attentionSize: H X W
    :param lowResImage: batchSize X 3 X H X W
    :param batchSize: batch size
    :param numOfMaxWords: max number of words in a caption
    :return: superImage batchSize X 3 X H X W
    """
    nvis = 8
    realImages = realImages[:nvis]
    if lowResImage is not None:
        lowResImage = lowResImage[:nvis]
    if attentionSize == 17:
        visualizationSize = attentionSize * 16
    else:
        visualizationSize = realImages.size(2)

    textCanvas = \
        np.ones([batchSize * FONT_MAX,
                 (numOfMaxWords + 2) * (visualizationSize + 2), 3],
                dtype=np.uint8)

    for i in range(numOfMaxWords):
        startIndex = (i + 2) * (visualizationSize + 2)
        endIndex = (i + 3) * (visualizationSize + 2)
        textCanvas[:, startIndex:endIndex, :] = COLOR_DIC[i]

    realImages = nn.functional.interpolate(realImages, size=(visualizationSize, visualizationSize),
                                           mode='bilinear', align_corners=False)

    # map from [-1, 1] to [0, 1]
    realImages.add_(1).div_(2).mul_(255)
    realImages = realImages.data.numpy()
    realImages = np.transpose(realImages, (0, 2, 3, 1))
    pad_sze = realImages.shape
    paddingMiddle = np.zeros([pad_sze[2], 2, 3])
    post_pad = np.zeros([pad_sze[1], pad_sze[2], 3])
    if lowResImage is not None:
        lowResImage = nn.functional.interpolate(lowResImage, size=(visualizationSize, visualizationSize),
                                                mode='bilinear', align_corners=False)
        # Map from [-1, 1] to [0, 1]
        lowResImage.add_(1).div_(2).mul_(255)
        lowResImage = lowResImage.data.numpy()
        lowResImage = np.transpose(lowResImage, (0, 2, 3, 1))

    sequenceLength = numOfMaxWords
    setofImages = []
    num = nvis

    textMappings, sentences = drawCaption(textCanvas, captions, idxToWord, visualizationSize)
    textMappings = np.asarray(textMappings).astype(np.uint8)

    bUpdate = 1
    for i in range(num):
        attention = attentionMaps[i].cpu().view(1, -1, attentionSize, attentionSize)
        maxAttention = attention.max(dim=1, keepdim=True)
        attention = torch.cat([maxAttention[0], attention], 1)

        attention = attention.view(-1, 1, attentionSize, attentionSize)
        attention = attention.repeat(1, 3, 1, 1).data.numpy()
        attention = np.transpose(attention, (0, 2, 3, 1))
        numOfAttentions = attention.shape[0]
        #
        image = realImages[i]
        if lowResImage is None:
            lowResImage = image
        else:
            lowResImage = lowResImage[i]
        row = [lowResImage, paddingMiddle]
        mergedRow = [image, paddingMiddle]
        rowWithoutNorm = []
        global_minV, global_maxV = 1, 0
        for j in range(numOfAttentions):
            oneMapping = attention[j]
            if (visualizationSize // attentionSize) > 1:
                oneMapping = \
                    skimage.transform.pyramid_expand(oneMapping, sigma=20,
                                                     upscale=visualizationSize // attentionSize,
                                                     multichannel=True)
            rowWithoutNorm.append(oneMapping)
            minV = oneMapping.min()
            maxV = oneMapping.max()
            if global_minV > minV:
                global_minV = minV
            if global_maxV < maxV:
                global_maxV = maxV
        for j in range(sequenceLength + 1):
            if j < numOfAttentions:
                oneMapping = rowWithoutNorm[j]
                # Normalize
                oneMapping = (oneMapping - global_minV) / (global_maxV - global_minV)
                oneMapping *= 255
                # Convert to PIL image
                imagePIL = Image.fromarray(np.uint8(image))
                attentionPIL = Image.fromarray(np.uint8(oneMapping))
                merged = Image.new('RGBA', (visualizationSize, visualizationSize), (0, 0, 0, 0))
                mask = Image.new('L', (visualizationSize, visualizationSize), (210))
                merged.paste(imagePIL, (0, 0))
                merged.paste(attentionPIL, (0, 0), mask)
                merged = np.array(merged)[:, :, :3]
            else:
                oneMapping = post_pad
                merged = post_pad
            row.append(oneMapping)
            row.append(paddingMiddle)
            #
            mergedRow.append(merged)
            mergedRow.append(paddingMiddle)
        row = np.concatenate(row, 1)
        mergedRow = np.concatenate(mergedRow, 1)
        txt = textMappings[i * FONT_MAX: (i + 1) * FONT_MAX]
        if txt.shape[1] != row.shape[1]:
            print('txt', txt.shape, 'row', row.shape)
            bUpdate = 0
            break
        row = np.concatenate([txt, row, mergedRow], 0)
        setofImages.append(row)
    if bUpdate:
        setofImages = np.concatenate(setofImages, 0)
        setofImages = setofImages.astype(np.uint8)
        return setofImages, sentences
    else:
        return None


def weightInitialization(m):
    """
    Initialize the weight of the model
    """
    className = m.__class__.__name__
    # Initialize the weight for Convolution and Deconvolution layers
    if className.find('Conv') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
    # Initialize the weight for BatchNorm layers
    elif className.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    # Initialize the weight for Linear layers
    elif className.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def loadParameters(model, newParameters):
    """
    Transfer new parameters to the model.
    """
    for parameter, newParameter in zip(model.parameters(), newParameters):
        parameter.data.copy_(newParameter)


def copyGeneratorParams(model):
    """
    Copy generator parameters from trained model to newly initialized model.
    """
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def mkdirWithPath(path):
    """
    Create a directory if it doesn't exist
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
