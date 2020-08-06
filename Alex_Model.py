device = 'cuda'
import torch.nn as nn
from collections import OrderedDict
class AlexNet(nn.Module):
  def __init__(self):
    super(AlexNet, self).__init__()
    self.Convolution_Layer = nn.Sequential(
        OrderedDict([('Conv1', nn.Conv2d(3, 96, 11, stride = 4, padding = 2)),
                     ('LRN1', nn.LocalResponseNorm(2)),
                     ('ReLU1', nn.ReLU(inplace = True)),
                     ('Max_Pool1', nn.MaxPool2d(3, stride = 2)),
                     
                     ('Conv2', nn.Conv2d(96, 256, 5, stride = 1, padding = 2)),
                     ('LRN2', nn.LocalResponseNorm(2)),
                     ('ReLU2', nn.ReLU(inplace = True)),
                     ('Max_Pool2', nn.MaxPool2d(3, stride = 2)),
                     
                     ('Conv3', nn.Conv2d(256, 384, 3, stride = 1, padding = 1)),
                     ('ReLU3', nn.ReLU(inplace = True)),
                     
                     ('Conv4', nn.Conv2d(384, 384, 3, stride = 1, padding = 1)),
                     ('ReLU4', nn.ReLU(inplace = True)),
                     
                     ('Conv5', nn.Conv2d(384, 256, 3, stride = 1, padding = 1)), 
                     ('ReLU5', nn.ReLU(inplace = True)),
                     ('Max_Pool3', nn.MaxPool2d(3, stride = 2))]))
    
    self.Fully_Connected_Layer = nn.Sequential(
        OrderedDict([('Dropout1', nn.Dropout()),
                     ('Linear1', nn.Linear(256 * 6 *6, 4096)),
                     ('ReLU1', nn.ReLU(inplace = True)),

                     ('Dropout2', nn.Dropout()),
                     ('Linear2', nn.Linear(4096, 4096)),
                     ('ReLU2', nn.ReLU(inplace = True)),

                     ('Linear3', nn.Linear(4096, 100))]))
    
  def forward(self, Image):
    conv_Image = self.Convolution_Layer(Image)
    CONV_Image = conv_Image.view(-1, 256 * 6 * 6)
    Logits_Image = self.Fully_Connected_Layer(CONV_Image)
    #Logits_Prob = torch.softmax(Logits_Image, dim = 0)
    return Logits_Image

ALEX_Model = AlexNet().to(device)
