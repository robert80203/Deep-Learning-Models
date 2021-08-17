import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *




class D(nn.Module):

  def __init__(self, projectd=False):
    super(D, self).__init__()
    self.base_filter = 512
    self.use_projection = projectd
    self.fromRGB = nn.Sequential(
        nn.Conv2d(3, self.base_filter//16, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(0.2)
    )
    self.conv1 = downblock(self.base_filter//16, self.base_filter//8)
    self.conv2 = downblock(self.base_filter//8, self.base_filter//4)
    self.conv3 = downblock(self.base_filter//4, self.base_filter//2)
    self.conv4 = downblock(self.base_filter//2, self.base_filter)
    
    self.bottleneck = nn.Sequential(
        nn.Conv2d(self.base_filter, self.base_filter, kernel_size = 4,stride = 1),
        nn.LeakyReLU(0.2)
    )
    self.classifier = nn.Sequential(
        nn.Linear(self.base_filter, 24),
        nn.Sigmoid()
    )
    if self.use_projection:
        self.logits = nn.Sequential(
            nn.Linear(self.base_filter, self.base_filter//8),
            nn.ReLU(),
            nn.Linear(self.base_filter//8, 1),
        )
    else:
        self.logits = nn.Sequential(
            nn.Conv2d(self.base_filter,self.base_filter//8,kernel_size=4,stride=1,padding=0,bias=False),
            nn.ReLU(),
            nn.Conv2d(self.base_filter//8,1,kernel_size=1,stride=1,padding=0,bias=False)
        )
    if self.use_projection:
        self.projector = nn.Sequential(
            #nn.Linear(24, self.base_filter),
            #nn.ReLU(),
            #nn.Linear(self.base_filter, self.base_filter),
            nn.Embedding(25,512,padding_idx=24)
        )
    else:
        self.conv5 = nn.Sequential(
            nn.Conv2d(self.base_filter,self.base_filter,kernel_size=4,stride=1,padding=0,bias=False),
            nn.LeakyReLU(0.2)
        )

  def forward(self, x, y=None):
    x = self.fromRGB(x)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    if self.use_projection:
        y = self.projector(y)
        y = torch.sum(y, dim=1)
        x = torch.sum(x, dim=(2, 3))
        aux = self.classifier(x)
        logit = self.logits(x).squeeze(1)
        c = torch.sum(y * x, dim=1)
        logit = logit + c
    else:
        #x = self.conv5(x)
        #x = x.view(-1,self.base_filter)
        logit = self.logits(x)
        #print(logit.size())
        aux = self.bottleneck(x).view(-1,self.base_filter)
        aux = self.classifier(aux)
    return logit, aux
        
class G(nn.Module):

  def __init__(self, condition_mode='embedding'):
    super(G, self).__init__()
    self.base_filter = 512
    self.cm = condition_mode
    if self.cm == 'embedding':
        self.linear = nn.Linear(64+64, self.base_filter*4*4)
    else:
        self.linear = nn.Linear(64+24, self.base_filter*4*4)

    self.activation = nn.ReLU()
    '''self.conv0 = nn.Sequential(
        nn.Conv2d(64+8, self.base_filter, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(self.base_filter),
        nn.ReLU(),
    )'''
    self.conv1 = upblock(self.base_filter, self.base_filter//2)#8x8
    self.conv2 = upblock(self.base_filter//2, self.base_filter//4)#16x16
    self.conv3 = upblock(self.base_filter//4, self.base_filter//8)#32x32
    self.conv4 = upblock(self.base_filter//8, self.base_filter//16)#64x64
    self.toRGB = nn.Sequential(
        nn.Conv2d(self.base_filter//16,3,kernel_size=3,stride=1,padding=1,bias=False),
        #nn.Tanh()
    )
    self.toembed = nn.Embedding(25,64,padding_idx=24)
  def forward(self, x, labels):
    
    if self.cm == 'embedding':
        embed = self.toembed(labels)
        embed = torch.sum(embed, dim=1) / embed.size(1)
        x = torch.cat((x,embed),dim=1)
    else:
        x = torch.cat((x,labels),dim=1)
    #x = x.unsqueeze(2).unsqueeze(3)
    #x = F.interpolate(x, scale_factor=4.0, mode='nearest')
    
    #x = x.repeat(1,1,4,4)
    x = self.linear(x)
    x = x.view(-1,self.base_filter,4,4)

    #x = self.conv0(x)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    output = self.toRGB(x)
    
    return output

