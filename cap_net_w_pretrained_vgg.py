import torch
import torch.nn as nn
from torch.autograd import Variable
import word_embedding
import pdb
import numpy as np

#load vgg16 with pretrained weights
import torchvision.models as models
our_vgg = models.vgg16(pretrained=True)

#cuda variables
use_gpu = torch.cuda.is_available()

# Input dimensions of VGG16 input image
VGG_IMG_DIM = 224

# Recurrent size must be same as last hidden layer off VGG16
RNN_HIDDEN_SIZE = 4096

# Dimension of word embeddings
WORDVEC_SIZE = 300

# Assume a limited language model consisting of this many words
VOCABULARY_SIZE = 6000

our_vgg.classifier = nn.Sequential(
      nn.Linear(512 * 7 * 7, 4096),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(4096, RNN_HIDDEN_SIZE),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(RNN_HIDDEN_SIZE, num_classes),
    )


class CaptionNet(nn.Module):

  def __init__(self, pretrained_model):
    super(CaptionNet, self).__init__()

    # Make VGG net
    self.pretrained = pretrained_model

    # Recurrent layer
    self.rnn_cell = nn.RNNCell(
      input_size = WORDVEC_SIZE,
      hidden_size = RNN_HIDDEN_SIZE,
      nonlinearity = 'relu',
    )

    # Linear layer to convert hidden layer to word in vocab
    self.hidden_to_vocab = nn.Linear(RNN_HIDDEN_SIZE, VOCABULARY_SIZE)


  def forward(self, img):
    """Forward pass through network
    Input: image tensor
    Output: sequence of words
    """
    hidden = self.pretrained(img)

    # First input is zero vector
    next_input = Variable(torch.zeros(WORDVEC_SIZE)).cuda()
    
    # For now, let's just generate 10 words (should actually generate until end token)
    words = []
    for _ in range(10):
      hidden = self.rnn_cell(next_input, hidden)
      word_class = self.hidden_to_vocab(hidden)
      _, word_ix = torch.max(word_class, 1)
      word_ix = int(word_ix)

      cur_word = word_embedding.get_word_from_index(word_ix)
      words.append(cur_word)

      # Update input to next layer
      next_input = Variable(word_embedding.get_word_embedding(cur_word)).cuda()

    return words


  def forward_perplexity(self, img, words):
    """Given image and ground-truth caption, compute negative log likelihood perplexity"""
    # Todo
    return 0  

