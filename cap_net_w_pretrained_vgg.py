import torch
import torch.nn as nn
from torch.autograd import Variable
import word_embedding
import pdb
import numpy as np

#load vgg16 with pretrained weights
import torchvision.models as models
vgg16 = models.vgg16(pretrained=True)



