import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()

        self.backbone = resnet18(weights=None, num_classes=num_classes)
    
    def forward(self, x):
        return self.backbone(x)