import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import models

from miscs.config import cfg


# Text Encoder Class based on RNN
class TextEncoderRNN(nn.Module):
    def __init__(self, ntoken, ninput=300, drop_prob=0.5, nhidden=128, nlayers=1, bidirectional=True):
        super(TextEncoderRNN, self).__init__()
        self.numSteps = cfg.TEXT.WORDS_NUM
        self.numToken = ntoken  # size of the dictionary
        self.numInput = ninput  # size of each embedding vector
        self.dropProb = drop_prob  # probability of an element to be zeroed
        self.numLayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnnType = cfg.RNN_TYPE
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.defineModule()
        self.initializeWeights()

    def defineModule(self):
        self.encoder = nn.Embedding(self.numToken, self.numInput)
        self.drop = nn.Dropout(self.dropProb)
        if self.rnnType == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.numInput, self.nhidden,
                               self.numLayers, batch_first=True,
                               dropout=self.dropProb,
                               bidirectional=self.bidirectional)
        elif self.rnnType == 'GRU':
            self.rnn = nn.GRU(self.numInput, self.nhidden,
                              self.numLayers, batch_first=True,
                              dropout=self.dropProb,
                              bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def initializeWeights(self):
        initialRange = 0.1
        self.encoder.weight.data.uniform_(-initialRange, initialRange)

    def initializeHidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnnType == 'LSTM':
            return (Variable(weight.new(self.numLayers * self.num_directions,
                                        bsz, self.nhidden).zero_()),
                    Variable(weight.new(self.numLayers * self.num_directions,
                                        bsz, self.nhidden).zero_()))
        else:
            return Variable(weight.new(self.numLayers * self.num_directions,
                                       bsz, self.nhidden).zero_())

    def forward(self, captions, captionLengths, hidden, mask=None):
        # X: torch.LongTensor of size batch X n_steps

        embeddings = self.drop(self.encoder(captions))
        #
        # Returns: a PackedSequence object
        captionLengths = captionLengths.data.tolist()
        embeddings = pack_padded_sequence(embeddings, captionLengths, batch_first=True)
        output, hidden = self.rnn(embeddings, hidden)
        output = pad_packed_sequence(output, batch_first=True)[0]
        wordEmbeddings = output.transpose(1, 2)

        if self.rnnType == 'LSTM':
            sentEmbeddings = hidden[0].transpose(0, 1).contiguous()
        else:
            sentEmbeddings = hidden.transpose(0, 1).contiguous()
        sentEmbeddings = sentEmbeddings.view(-1, self.nhidden * self.num_directions)
        return wordEmbeddings, sentEmbeddings
numOfHiddens = 128
# Custom LSTM layer Class for our Generator
class CustomLSTM(nn.Module):
    def __init__(self, inputSize, hiddenSize):
        super().__init__()
        self.input_sz = inputSize
        self.hidden_size = hiddenSize
        self.W = nn.Parameter(torch.Tensor(inputSize, hiddenSize * 4))
        self.U = nn.Parameter(torch.Tensor(hiddenSize, hiddenSize * 4))
        self.bias = nn.Parameter(torch.Tensor(hiddenSize * 4))
        self.initializeWeights()
        self.noise2h = nn.Linear(100, 256)
        self.noise2c = nn.Linear(100, 256)
        # self.initializeHidden()
        self.hidden_seq = []

    def initializeWeights(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)

    def initializeHidden(self, noise):
        h_t = self.noise2h(noise)
        c_t = self.noise2c(noise)

        self.c_t = c_t
        self.h_t = h_t

    def forward(self, X):
        c_t = self.c_t
        h_t = self.h_t
        HS = self.hidden_size
        #        x_t = X[:, t, :]
        x_t = X
        # Compute full batch as a single matrix multiplication
        gates = x_t @ self.W + h_t @ self.U + self.bias
        i_t, f_t, g_t, o_t = (
            torch.sigmoid(gates[:, :HS]),        # input gate
            torch.sigmoid(gates[:, HS:HS * 2]),  # forget gate
            torch.tanh(gates[:, HS * 2:HS * 3]), # memory cell
            torch.sigmoid(gates[:, HS * 3:]),    # output gate
        )
        c_t = f_t * c_t + i_t * g_t  # update memory cell
        h_t = o_t * torch.tanh(c_t)  # compute hidden state
        self.c_t = c_t
        self.h_t = h_t

        return h_t, c_t
