import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FashionModel(nn.Module):
    def __init__(self, output_size=228):
        super(FashionModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(1000, output_size, bias=True)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(in_features=10 * 52 * 52, out_features=32),
            nn.ReLU(True),
            nn.Linear(in_features=32, out_features=3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.cuda.FloatTensor([1, 0, 0, 0, 1, 0]))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 52 * 52)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        x = self.resnet(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        # x = F.sigmoid(x)
        return x