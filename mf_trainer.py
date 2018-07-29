from __future__ import print_function, division
import time
import os
import copy
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models, transforms
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from tensorboardX import SummaryWriter

from src.mf_dataset import MaterialistFashion, TestMaterialistFashion
from tensorboard_writer import TensorboardWriter
from model_saver import ModelSaver

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
torch.set_printoptions(threshold=5000)
plt.ion()

verbose = True
spot = False

serialization_dir = 'tmp'
mfb = MultiLabelBinarizer()
mf_labels = np.arange(0, 228)
NUM_CLASSES = len(mf_labels)
# class_names = mf_labels #image_datasets['train'].classes
mfb.fit_transform([mf_labels])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log_dirs = {'train': SummaryWriter(os.path.join(serialization_dir, "log", "train")),
            'val': SummaryWriter(os.path.join(serialization_dir, "log", "validation"))
            }
tensorboard = TensorboardWriter(log_dirs['train'], log_dirs['val'])
summary_interval = 100
model_saver = ModelSaver(serialization_dir)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print('[i] Loading datasets...')

image_datasets = {
    'train': MaterialistFashion('/media/spike/Scoob/materialist_fashion_data/', 'train.json', data_transforms['train'],load_first=16*63),
    'val': MaterialistFashion('/media/spike/Scoob/materialist_fashion_val/', 'validation.json', data_transforms['val'], percent=10)
}

batch_size = 16

dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=False, num_workers=8),
    'val': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=False, num_workers=8)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

print('[i] Done loading datasets.')


def check_label_distribution(dataloader,filter_threshold=100):
    label_bins = {}
    for i in range(0, 228):
        label_bins[i] = 0
    for i, (_, labels, _) in enumerate(dataloader):
        for batch in mfb.inverse_transform(labels):
            for label in batch:
                label_bins[int(label)] += 1
    label_bins = sorted(label_bins.items(), key=lambda x:x[1], reverse=True)
    max = label_bins[0][1]
    rescailing = [1]*228
    filtered_mask = [0]*228
    for key, value in label_bins:
        print("Label ID: {} -> Count: {}".format(key, value))
        if not value == max:
            rescailing[key] = rescailing[key] - value/max
        else:
            rescailing[key] = 1e-10
        if value >=filter_threshold:
            filtered_mask[key] = 1
    return rescailing, filtered_mask


def imshow(inp, title=None):
    """shows a batch of images."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def show_first_batch():
    inputs, labels, image_id = next(iter(dataloaders['train']))
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[x for x in image_id])
    print("Images shown")


# def metrics_to_tensorboard(epoch: int, train_metrics: dict, val_metrics: dict = None) -> None:
#     """
#     Sends all of the train metrics (and validation metrics, if provided) to tensorboard.
#     """
#     metric_names = set(train_metrics.keys())
#     if val_metrics is not None:
#         metric_names.update(val_metrics.keys())
#     val_metrics = val_metrics or {}
#
#     for name in metric_names:
#         train_metric = train_metrics.get(name)
#         if train_metric is not None:
#             tensorboard.add_train_scalar(name, train_metric, epoch)
#         val_metric = val_metrics.get(name)
#         if val_metric is not None:
#             tensorboard.add_validation_scalar(name, val_metric, epoch)


######################################################################
# Training the model
def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0

    model, optimizer, epoch_counter, global_step = model_saver.restore_checkpoint(model, optimizer)
    val_step = global_step

    for epoch in range(epoch_counter, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_f1 = 0.0

            # Iterate over data.
            for n_iter, (inputs, labels, _) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = outputs.ge(0.3).type(torch.cuda.FloatTensor)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        global_step += 1
                        loss.backward()
                        optimizer.step()
                    else:
                        val_step +=1

                # statistics
                if verbose and phase == 'val':
                    for i, (true_label, pred_label) in enumerate(zip(mfb.inverse_transform(labels), mfb.inverse_transform(preds))):
                        print('{} True labels[{}]: {}'.format(phase, i, true_label))
                        print('{} Pred labels[{}]: {}'.format(phase, i, pred_label))

                running_loss += loss.item() * inputs.size(0)
                mf_f1 = f1_score(labels, preds, average='micro')
                running_f1 += mf_f1 * inputs.size(0)  # torch.sum(preds == labels.data)
                spot_loss = (running_loss/n_iter if n_iter > 0 else running_loss)
                if phase == 'train':
                    tensorboard.add_train_scalar('loss', spot_loss, global_step)
                    tensorboard.add_train_scalar('microF1', mf_f1, global_step)
                else:
                    tensorboard.add_validation_scalar('loss', spot_loss, val_step)
                    tensorboard.add_validation_scalar('microF1', mf_f1, val_step)

                if spot:
                    print('{} Spot:  Loss: {:.4f} F1: {:.4f} Step: {}'.format(phase, spot_loss, mf_f1, global_step))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_f1 = running_f1 / dataset_sizes[phase]

            print('{} Loss: {:.4f} F1: {:.4f} <-----------------------------------------------------------------------------------------------'.format(
                phase, epoch_loss, epoch_f1))

            # deep copy the model
            if phase == 'val':
                if epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print('[i] Saving new best F1 {:.4f}'.format(best_f1))
                    model_saver.save_checkpoint(model, epoch, optimizer, global_step, True)

                print("[i] Saving last epoch model.")
                model_saver.save_checkpoint(model, epoch, optimizer, global_step, False)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val F1: {:4f}'.format(best_f1))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


######################################################################
# Visualizing the model predictions
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels, _) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = outputs.ge(0.2).type(torch.cuda.FloatTensor)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(mfb.inverse_transform(preds[j])))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


class FashionModel(nn.Module):
    def __init__(self, output_size=228):
        super(FashionModel, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(1000, output_size, bias=True)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 52 * 52, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
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
        x = self.resnet50(x)
        x = self.fc(self.dropout(x))
        return x

######################################################################

def train_fashion_model():
    model_ft = FashionModel()
    model_ft = model_ft.to(device)
    criterion = nn.MultiLabelSoftMarginLoss()

    # Observe that all parameters are being optimized
    params = list(model_ft.localization.parameters()) + \
             list(model_ft.fc_loc.parameters()) + \
             list(model_ft.resnet50.fc.parameters()) + \
             list(model_ft.fc.parameters())
    optimizer_ft = optim.Adam(params)  # , lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  # Decay LR by a factor of 0.1 every 7 epochs
    return train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=4)

def train_standard_resnet():
    #rescaling_weights, filtered_mask = check_label_distribution(dataloaders['train'])

    model_ft = models.resnet50(pretrained=True)

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 228)
    model_ft = model_ft.to(device)

    criterion = nn.MultiLabelSoftMarginLoss()

    optimizer_ft = optim.Adam(model_ft.fc.parameters()) #, lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  # Decay LR by a factor of 0.1 every 7 epochs
    return train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=4)

def train_extended_standard_resnet():
    #rescaling_weights, filtered_mask = check_label_distribution(dataloaders['train'])

    model_ft = nn.Sequential(models.resnet50(pretrained=True),
                             nn.Dropout(),
                             nn.Linear(1000, 228, bias=True)
                             )

    model_ft = model_ft.to(device)

    criterion = nn.MultiLabelSoftMarginLoss()

    optimizer_ft = optim.Adam(list(model_ft[0].fc.parameters())+list(model_ft[2].parameters()))

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  # Decay LR by a factor of 0.1 every 7 epochs
    return train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=4)


model_ft = train_extended_standard_resnet()
# visualize_model(model_ft)

plt.ioff()
plt.show()

