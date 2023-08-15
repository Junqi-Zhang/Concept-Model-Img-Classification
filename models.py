import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from collections import OrderedDict


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()

        self.backbone = resnet18(weights=None, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)


class ConceptQuantization(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ConceptQuantization, self).__init__()

        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class TestResNet18(nn.Module):
    def __init__(self, num_classes):
        super(TestResNet18, self).__init__()

        classifier = resnet18(weights=None, num_classes=num_classes)
        self.backbone = nn.Sequential(*list(classifier.children())[:-1])

        self.cq = ConceptQuantization(
            input_dim=512,
            num_classes=num_classes
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.cq(x)
        return x


PROVIDED_MODELS = OrderedDict(
    {
        "ResNet18": ResNet18,
        "TestResNet18": TestResNet18
    }
)
