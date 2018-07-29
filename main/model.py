import torch
from torch.nn.init import kaiming_normal_
from torchvision import models

use_cuda = torch.cuda.is_available()
if torch.cuda.is_available():
    import torch.cuda as torch
else:
    import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 48, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(48, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(16 * 16 * 128, 2048)
        self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))  # input: 224*224
        x = F.relu(self.pool2(self.conv2(x)))  # input 112*112
        x = F.relu(self.pool3(self.conv3(x)))  # input 56*56
        x = x.view(-1, 16 * 16 * 128)  # 28*28
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # output: 228
        # x = F.sigmoid(x)
        return x


## ResNet fine-tuning
class ResNet50(nn.Module):
    ## We use ResNet weights from PyCaffe.
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()

        # Loading ResNet arch from PyTorch and weights from Pycaffe
        original_model = models.resnet50(pretrained=False)
        # original_model.load_state_dict(torch.load('./zoo/resnet50.pth'))

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

        # Freeze those weights
        # for p in self.features.parameters():
        #     p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)

        return y

class ResNet152(nn.Module):
    def __init__(self, num_classes):
        super(ResNet152, self).__init__()
        if use_cuda:
            self.net = torchvision.models.resnet152(pretrained=True).cuda()
        else:
            self.net = torchvision.models.resnet152(pretrained=True)
        for param in self.net.parameters():
            param.requires_grad = False
        num_features = self.net.fc.in_features
        self.net.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.net(x)
        # x = F.sigmoid(x)
        return x

class InceptionV3(nn.Module):
    def __init__(self, num_classes):
        super(InceptionV3, self).__init__()
        if use_cuda:
            self.net = torchvision.models.inception_v3(pretrained=True).cuda()
        else:
            self.net = torchvision.models.inception_v3(pretrained=True)
        for param in self.net.parameters():
            param.requires_grad = False
        num_features = self.net.fc.in_features
        self.net.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.net(x)
        # x = F.sigmoid(x)
        return x

class VGG19(nn.Module):
    def __init__(self, num_classes):
        super(VGG19, self).__init__()
        if use_cuda:
            self.net = torchvision.models.vgg19_bn(pretrained=True).cuda()
        else:
            self.net = torchvision.models.vgg19_bn(pretrained=True)
        for param in self.net.parameters():
            param.requires_grad = False
        num_features = self.net.fc.in_features
        self.net.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.net(x)
        # x = F.sigmoid(x)
        return x