"""
Pytorch implementation of ResNet models
"""
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

    
class ResNet(nn.Module):
    def __init__(self, embedding, classifier):
        super(ResNet, self).__init__()
        self.embed = embedding
        self.classifier = classifier

    def forward(self, x):
        out = self.embed(x)
        out = self.classifier(out)
        return out

    def conf(self,x):
        out = self.embed(x)
        if hasattr(self.classifier,'conf'):
            return self.classifier.conf(out)
        return F.softmax(self.classifier(out),dim=1)



def conv_block(in_channels, out_channels, pool=False, dropout = 0.0):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)

class ResNet9Embed(nn.Module):
    def __init__(self):
        super(ResNet9Embed, self).__init__()
        self.feature = None
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 128, pool=True, dropout=0.0)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True, dropout=0.0)
        self.conv4 = conv_block(256, 512, pool=True, dropout=0.0)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))


    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        self.feature = out.clone().detach()
        return out



def resnet9(classifier, coeff=None, mod=False, **kwargs):
    embed = ResNet9Embed()
    return ResNet(embed,classifier)


