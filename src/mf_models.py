import torch
from torch.nn.init import kaiming_normal_
from torchvision import models

use_cuda = torch.cuda.is_available()
if torch.cuda.is_available():
    import torch.cuda as torch
else:
    import torch as torch
import torch.nn as nn

## ResNet fine-tuning
class ResNet50(nn.Module):
    ## We use ResNet weights from PyCaffe.
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        original_model = models.resnet50(pretrained=True)
        # Everything except the last linear layer
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        # Get number of features of last layer
        num_feats = original_model.fc.in_features
        # Plug our classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_feats, num_classes)
        )
        # Init of last layer
        for m in self.classifier:
            kaiming_normal_(m.weight)

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)

        return y