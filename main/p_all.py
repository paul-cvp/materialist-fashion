import torch
from torchvision import transforms
from torchvision.transforms import Normalize, ToTensor, Resize, CenterCrop, RandomResizedCrop
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from mf_dataset import MaterialistFashion, TestMaterialistFashion
from tqdm import tqdm
import numpy as np
from sklearn.metrics import fbeta_score

from src.mf_models import ResNet50

image_size = 224
center_crop = CenterCrop(image_size)
random_crop = RandomResizedCrop(image_size)
scale = Resize((image_size, image_size))
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

mf_train = MaterialistFashion('/media/spike/Scoob/materialist_fashion_data/', '../train.json',
                              transforms.Compose([random_crop, ToTensor(), normalize]),
                              100)
mf_val = MaterialistFashion('/media/spike/Scoob/materialist_fashion_val/', '../validation.json',
                            transforms.Compose([scale, ToTensor(), normalize]),
                            100)
total_images = 39706
mf_test = TestMaterialistFashion('/media/spike/Scoob/materialist_fashion_test/',
                                 total_images,
                                 transforms.Compose([scale, ToTensor(), normalize]))

mf_train_loader = torch.utils.data.DataLoader(mf_train, batch_size=26, shuffle=True, num_workers=8)
mf_val_loader = torch.utils.data.DataLoader(mf_val, batch_size=16, shuffle=False, num_workers=8)
mf_test_loader = torch.utils.data.DataLoader(mf_test, batch_size=16, shuffle=False, num_workers=8)


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
        self.conv_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(16 * 16 * 128, 2048)
        self.fc2 = nn.Linear(2048, self.num_classes)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))  # input: 224*224
        x = F.relu(self.pool2(self.conv2(x)))  # input 112*112
        x = F.relu(self.pool3(self.conv_drop(self.conv3(x))))  # input 56*56
        x = x.view(-1, 16 * 16 * 128)  # 28*28
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)  # output: 228
        # x = F.sigmoid(x)
        return x


# model = Net(228).cuda()
model = ResNet50(228).cuda()

# optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0005)
optimizer = optim.Adam(model.parameters())

# criterion = nn.BCELoss().cuda()
criterion = nn.SoftMarginLoss().cuda()

def train(epoch):
    model.train()
    for batch_idx, (data, target, _) in enumerate(mf_train_loader):
        data, target = data.cuda(async=True), target.cuda(async=True)  # On GPU
        data, target = Variable(data), Variable(target)
        # optimizer.zero_grad()
        model.zero_grad()
        output = model(data)
        # loss = F.binary_cross_entropy(output, target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(mf_train_loader.dataset),
                       100. * batch_idx / len(mf_train_loader), loss.data[0]))


def validate(epoch):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    ids = []

    print("Starting Validation")
    for batch_idx, (data, target, id) in enumerate(tqdm(mf_val_loader)):
        true_labels.append(target.cpu().numpy())
        ids.extend(id)
        with torch.no_grad():
            data, target = data.cuda(async=True), target.cuda(async=True).float()
            data, target = Variable(data).cuda(), Variable(target).cuda()

            pred = model(data)
            predictions.append(pred.data.cpu().numpy())

            # total_loss += F.binary_cross_entropy(pred, target).data[0]
            total_loss += criterion(pred, target).data.item()

    avg_loss = total_loss / len(mf_val_loader)

    true_labels = np.vstack(true_labels)
    predictions = np.vstack(predictions)
    threshold = [0.20] * 228
    score = fbeta_score(true_labels, predictions > threshold, beta=2, average='samples')
    predictions[predictions >= 0.2] = 1
    predictions[predictions < 0.2] = 0
    predictions = mf_train.getLabelEncoder().inverse_transform(predictions)
    true_labels = mf_train.getLabelEncoder().inverse_transform(true_labels)
    for idx, predicted, gt in zip(ids, predictions, true_labels):
        print('ID: {0} =>\r\nPredicted: {1} \r\nGround truth: {2}'.format(str(idx),
                                                                          ', '.join(str(k) for k in predicted),
                                                                          ', '.join(str(k) for k in gt)))

    print("===> Validation - Avg. loss: {:.4f}\tF2 Score: {:.4f}".format(avg_loss, score))
    return score, avg_loss, threshold

for epoch in range(1, 3):
    train(epoch)
    torch.save(model.state_dict(), '../resnet50_model{}.pt'.format(epoch))
    validate(epoch)

