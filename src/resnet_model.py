import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class FacialEmotionResNet(nn.Module):
    def __init__(self, num_classes=5, fine_tune=False):
        super(FacialEmotionResNet, self).__init__()

        # Load pre-trained ResNet18 using the latest torchvision syntax
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Freeze the pre-trained layers if not fine-tuning
        if not fine_tune:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Replace final fully connected layer with custom classifier
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.resnet(x)
