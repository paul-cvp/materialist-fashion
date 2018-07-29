import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Normalize, ToTensor, Resize
import matplotlib.pyplot as plt
import numpy as np
import copy
from tqdm import tqdm

# Notes on training: Start by training images to match the label.
# Negative examples are images not part of the label and also not part of labels that have the same of any other labels
# For testing you must loop through all 288 labels and only return the strongest 4-6 labels (above some treshhold)
from main import model
from main import dataset

use_cuda = torch.cuda.is_available()
image_size = 224  # was 128
scale = Resize((image_size, image_size))
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
tforms = torchvision.transforms.Compose([scale, ToTensor(), normalize])
train_percent = 1  # any value from 1 to 100
test_percent = 10
batch = 32
mf_train_set = dataset.MaterialistFashion('/media/spike/Scoob/materialist_fashion_data/', '../train.json', tforms,
                                          train_percent)
mf_train_loader = torch.utils.data.DataLoader(mf_train_set, batch_size=batch, shuffle=True, num_workers=4)
print("Size of train loader: {}".format(len(mf_train_loader)))

mf_test_set = dataset.MaterialistFashion('/media/spike/Scoob/materialist_fashion_val/', '../validation.json', tforms,
                                         test_percent)
mf_test_loader = torch.utils.data.DataLoader(mf_test_set, batch_size=batch, shuffle=False, num_workers=4)
print("Size of test loader: {}".format(len(mf_test_loader)))

mf_labels = np.arange(1, 229)
NUM_CLASSES = len(mf_labels)

# remove last fully-connected layer
# new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
# model.classifier = new_classifier

if use_cuda:
    net = model.ResNet50(NUM_CLASSES).cuda()
    # net = model.Net(NUM_CLASSES).cuda()
else:
    net = model.ResNet50(NUM_CLASSES)
    # net = model.Net(NUM_CLASSES)


def lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


is_training = True
if is_training:
    criterion = nn.MultiLabelSoftMarginLoss()
    # criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(params, lr=0.0001)
    optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0005)
    print('Started Training')

    for epoch in range(10):  # loop over the dataset multiple times
        net.train()
        optimizer = lr_scheduler(optimizer, epoch)

        running_loss = 0.0
        for i, data in enumerate(mf_train_loader, 0):
            # get the inputs
            inputs, labels = data
            # wrap them in Variable
            if use_cuda:
                inputs, labels = Variable(inputs).cuda(), Variable(labels, requires_grad=False).cuda()
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            # print(i)
            if i % 100 == 99:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
            if i % 100 == 0:
                print('Train Epoch: {:03d} [{:05d}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(data), len(mf_train_loader) * len(data),
                           100. * i / len(mf_train_loader), loss.data[0]))

    print('Finished Training')

    torch.save(net.state_dict(), '../mymodel.pt')
else:
    net.load_state_dict(torch.load('../mymodel.pt'))



def validate(epoch, valid_loader, model, loss_func, mlb):
    ## Volatile variables do not save intermediate results and build graphs for backprop, achieving massive memory savings.

    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []

    print("Starting Validation")
    for batch_idx, (data, target) in enumerate(tqdm(valid_loader)):
        true_labels.append(target.cpu().numpy())

        data, target = data.cuda(async=True), target.cuda(async=True)
        data, target = Variable(data, volatile=True), Variable(target, volatile=True)

        pred = model(data)
        predictions.append(F.sigmoid(pred).data.cpu().numpy())

        total_loss += loss_func(pred, target).data[0]

    avg_loss = total_loss / len(valid_loader)

    predictions = np.vstack(predictions)
    true_labels = np.vstack(true_labels)

    score, threshold = best_f2_score(true_labels, predictions)
    print("Corresponding tags\n{}".format(mlb.classes_))

    print("===> Validation - Avg. loss: {:.4f}\tF2 Score: {:.4f}".format(avg_loss, score))


    return score, avg_loss, threshold

def multihot_encoder(labels):
    multihot_labels = []
    for i in range(1, 229):
        if i not in labels:
            multihot_labels.append(0)
        else:
            multihot_labels.append(1)
    return np.array(multihot_labels)


def multihot_decoder(predicted, labels):
    x, y = predicted.shape
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    values = []
    for i in range(0, x):
        values[i] = []
        for j in range(0, y):
            if predicted[i][j] >= 1 / 4:
                values[i].append(j + 1)
                predicted[i][j] = 1
                if labels[i][j] == 1:
                    # print('true positive')
                    tp += 1
                else:
                    # print('false positive')
                    fp += 1
            else:
                predicted[i][j] = 0
                if labels[i][j] == 0:
                    # print('true negative')
                    tn += 1
                else:
                    # print('false negative')
                    fn += 1
    return predicted, tp, fp, tn, fn, values


correct = 0
total = 0
for data in mf_test_loader:
    images, labels = data
    labels = labels.data.cpu().numpy()
    value_labels = labels
    x, y = labels.shape
    temp_labels = []
    for i in range(0, x):
        temp_labels.append(multihot_encoder(labels[i]))
    labels = temp_labels
    if use_cuda:
        outputs = net(Variable(images).cuda())
    else:
        outputs = net(Variable(images))
    _, preds = torch.eq(torch.round(outputs), labels)
    print(preds)
    predicted, tp, fp, tn, fn, values = multihot_decoder(outputs.data.cpu().numpy(), labels)
    total += tp + fp + fn
    correct += tp

print('Accuracy of the network on the test images: %.8f %%' % (100 * correct / total))

# class_correct = list(0. for i in range(0, 228))
# class_total = list(0. for i in range(0, 228))
# for data in mf_test_loader:
#     images, labels = data
#     outputs = net(Variable(images))
#     _, predicted = torch.max(outputs.data, 1)
#     c = (predicted == labels).squeeze()
#     for i in range(4):
#         label = labels[i]
#         class_correct[label] += c[i]
#         class_total[label] += 1
#
# for i in range(229):
#     print('Accuracy of %5s : %2d %%' % (
#         mf_labels[i], 100 * class_correct[i] / class_total[i]))
