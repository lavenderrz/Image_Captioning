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

if use_gpu:
  our_vgg = our_vgg.cuda()

  

