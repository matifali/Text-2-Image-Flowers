from __future__ import division
from __future__ import print_function

import numpy as np
import yaml
from easydict import EasyDict as edict

dummy = edict()
cfg = dummy

dummy.DATASET_NAME = 'flowers'
dummy.CONFIG_NAME = ''
dummy.DATA_DIR = 'data'
dummy.MODELS_DIR = 'models'
dummy.GPU_ID = 0  # set to -1 if using cpu
dummy.CUDA = True  # set to False if using cpu
dummy.WORKERS = 1

dummy.RNN_TYPE = 'LSTM'  # other option is 'GRU'
dummy.TEST = 0        # 0: train, 1: inference, 2: test, 3: model statistics
dummy.loss = 'hinge'  # 'cross_entropy' or 'hinge'
dummy.TREE = edict()  # stores the tree structure of the dataset
dummy.TREE.BRANCH_NUM = 1  # max number of branches
dummy.TREE.BASE_SIZE = 64  # number of features in each node

# Modal Specifications
dummy.GAN = edict()  # stores the specifications of the GAN
dummy.GAN.DF_DIM = 64  # dimensionality of discriminator feature
dummy.GAN.GF_DIM = 128  # dimensionality of generator feature
dummy.GAN.Z_DIM = 100  # dimensionality of latent variable in G
dummy.GAN.CONDITION_DIM = 100  # dimensionality of conditioned variable in D
dummy.GAN.R_NUM = 2  # number of relations in the dataset
dummy.GAN.B_ATTENTION = True  # enable attention or not
dummy.GAN.B_DCGAN = True  # use DCGAN or not

# Default Training parameters
dummy.TRAIN = edict()  # stores the training parameters
dummy.TRAIN.BATCH_SIZE = 64  # batch size
dummy.TRAIN.MAX_EPOCH = 1000  # max epoch to train
dummy.TRAIN.NF = 32  # number of feature maps in first layer of D and G
dummy.TRAIN.DF = 32  # dimensionality of features in D and G
dummy.TRAIN.SNAPSHOT_INTERVAL = 2000  # snapshot interval
dummy.TRAIN.FLAG = True  # flag for training
dummy.TRAIN.NET_E = ''  # path to pretrained encoder
dummy.TRAIN.NET_G = ''  # path to pretrained generator
dummy.TRAIN.B_NET_D = True  # whether to use discriminator or not
dummy.TRAIN.SMOOTH = edict()  # label smoothing
dummy.TRAIN.DISCRIMINATOR_LR = 2e-4  # learning rate of discriminator
dummy.TRAIN.GENERATOR_LR = 2e-4  # learning rate of generator
dummy.TRAIN.ENCODER_LR = 2e-4  # learning rate of encoder
dummy.TRAIN.RNN_GRAD_CLIP = 0.25  # clip for gradient
dummy.TRAIN.SMOOTH.LAMBDA = 1.0  # label smoothing parameter
dummy.TRAIN.SMOOTH.GAMMA1 = 5.0  # label smoothing parameter
dummy.TRAIN.SMOOTH.GAMMA3 = 10.0  # label smoothing parameter
dummy.TRAIN.SMOOTH.GAMMA2 = 5.0  # label smoothing parameter

dummy.TEXT = edict()  # stores the specifications of the text modal
dummy.TEXT.CAPTIONS_PER_IMAGE = 10  # number of captions to sample for each image
dummy.TEXT.EMBEDDING_DIM = 256  # dimensionality of word embedding
dummy.TEXT.WORDS_NUM = 18  # number of words in vocabulary
dummy.TEXT.DAMSM_NAME = ''  # name of the text encoder model


def mergeConfigs(newConfig, defaultConfig):
    """
    Merge new config into the default configuration.
    """
    if type(newConfig) is not edict:
        return

    for key, value in newConfig.items():
        if key not in defaultConfig:
            raise KeyError('{} is not a valid config key'.format(key))

        typeOld = type(defaultConfig[key])
        if typeOld is not type(value):
            if isinstance(defaultConfig[key], np.ndarray):
                value = np.array(value, dtype=defaultConfig[key].dtype)
            else:
                raise ValueError(f'Type mismatch ({type(defaultConfig[key])} vs. {type(value)}) for key: {key}')

        # Merge both dictionaries in a recursive fashion
        if type(value) is edict:
            try:
                mergeConfigs(newConfig[key], defaultConfig[key])
            except:  # If the new config is a different structure than the default config
                print(f'Error under config key: {key}')
                raise
        else:
            defaultConfig[key] = value


def readConfigFile(filename):
    """Load a config file and merge it into the default options."""
    # Read config file
    with open(filename, 'r') as f:
        cfg = edict(yaml.full_load(f))
    # Merge the new config over the default config
    mergeConfigs(cfg, dummy)
