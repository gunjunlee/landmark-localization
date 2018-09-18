import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms

class CustomNet(nn.Module):
    def __init__(self, n_classes=4):
        super(CustomNet, self).__init__()
        
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, 2 * n_classes, bias=True)

    def forward(self, x):
        x = self.resnet(x)
        return x

if __name__ == '__main__':
    net = CostumNet()
    
    print(net(torch.ones((1, 3, 224, 224))))