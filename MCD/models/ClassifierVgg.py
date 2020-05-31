import torch
import torch.nn as nn
import torchvision.models.vgg


class ClassifierVgg(nn.Module):
    def __init__(self, num_classes=31):
        super(ClassifierVgg, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x