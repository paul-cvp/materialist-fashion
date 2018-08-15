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
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
import torchvision
from torchvision import models, transforms
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from tensorboardX import SummaryWriter

from mf_dataset import MaterialistFashion
from tensorboard_writer import TensorboardWriter
from model_saver import ModelSaver
from fashion_model import FashionModel
import sys

torch.manual_seed(42)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
torch.set_printoptions(threshold=5000)
plt.ion()


class MfTrainer:
    def __init__(self, base_folder, json_files, batch_size=64):
        self.verbose = False
        self.spot = False

        serialization_dir = 'tmp'
        self.mfb = MultiLabelBinarizer()
        mf_labels = np.arange(0, 228)
        # NUM_CLASSES = len(mf_labels)
        # class_names = mf_labels #image_datasets['train'].classes
        self.mfb.fit_transform([mf_labels])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log_dirs = {'train': SummaryWriter(os.path.join(serialization_dir, "log", "train")),
                    'val': SummaryWriter(os.path.join(serialization_dir, "log", "validation"))
                    }
        self.tensorboard = TensorboardWriter(log_dirs['train'], log_dirs['val'])
        # summary_interval = 100
        self.model_saver = ModelSaver(serialization_dir)

        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        print('[i] Loading datasets...')
        dataset = MaterialistFashion(base_folder, json_files, data_transforms['train'], id_as_path=True, load_first=8 * 139)
        validation_split = .2

        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        shuffle_dataset = True
        random_seed = 42
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        # image_datasets = {
        #     'train': MaterialistFashion(train_folder, train_json, data_transforms['train']),
        #     'val': MaterialistFashion(val_folder, val_json, data_transforms['val'])
        # }

        self.dataloaders = {
            'train': torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8),
            'val': torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=8)
        }

        self.dataset_sizes = {'train': len(train_indices), 'val': len(val_indices)}  # {x: len(image_datasets[x]) for x in ['train', 'val']}

        print('[i] Done loading datasets.')

    def check_label_distribution(self, dataloader, filter_threshold=100):
        label_bins = {}
        for i in range(0, 228):
            label_bins[i] = 0
        for i, (_, labels, _) in enumerate(dataloader):
            for batch in self.mfb.inverse_transform(labels):
                for label in batch:
                    label_bins[int(label)] += 1
        label_bins = sorted(label_bins.items(), key=lambda x: x[1], reverse=True)
        max = label_bins[0][1]
        rescailing = [1] * 228
        filtered_mask = [0] * 228
        for key, value in label_bins:
            print("Label ID: {} -> Count: {}".format(key, value))
            if not value == max:
                rescailing[key] = rescailing[key] - value / max
            else:
                rescailing[key] = 1e-10
            if value >= filter_threshold:
                filtered_mask[key] = 1
        return rescailing, filtered_mask

    def imshow(self, inp, title=None):
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

    def show_first_batch(self):
        inputs, labels, image_id = next(iter(self.dataloaders['train']))
        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)
        self.imshow(out, title=[x for x in image_id])
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
    def train_model(self, model, criterion, optimizer, scheduler, num_epochs=10):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_f1 = 0.0

        model, optimizer, epoch_counter, global_step = self.model_saver.restore_checkpoint(model, optimizer)
        val_step = global_step

        chosen_threshold = 0.2
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
                low_threshold = 0.01
                high_threshold = 0.5
                step_threshold = 0.01
                running_th_f1 = {}
                for threshold in np.arange(low_threshold, high_threshold, step=step_threshold):
                    running_th_f1[threshold] = 0.0

                # Iterate over data.
                for n_iter, (inputs, labels, _) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        soft_out = F.sigmoid(outputs)#F.softmax(outputs, dim=1)
                        # if n_iter % 100 == 0:
                        #     self.imshow(torchvision.utils.make_grid(torch.cat((inputs.detach().cpu(),model.stn(inputs).detach().cpu()))), title='stn')
                        th_selection_preds = {}
                        for threshold in np.arange(low_threshold, high_threshold, step=step_threshold):
                            th_selection_preds[threshold] = soft_out.ge(threshold).type(torch.cuda.FloatTensor)

                        preds = soft_out.ge(chosen_threshold).type(torch.cuda.FloatTensor)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            global_step += 1
                            loss.backward()
                            optimizer.step()
                        else:
                            val_step += 1

                    # statistics
                    if (n_iter == 1 or self.verbose) and phase == 'val':
                        # self.imshow(torchvision.utils.make_grid(torch.cat((inputs.detach().cpu(),model.stn(inputs).detach().cpu()))),title='stn')
                        for i, (true_label, pred_label) in enumerate(zip(self.mfb.inverse_transform(labels), self.mfb.inverse_transform(preds))):
                            true_label_output_probs = [soft_out.cpu().data.numpy()[i][x] for x in true_label]
                            pred_label_output_probs = [soft_out.cpu().data.numpy()[i][x] for x in pred_label]
                            print('{} True labels[{}]: {}'.format(phase, i, true_label))
                            print('{} True probs [{}]: {}'.format(phase, i, true_label_output_probs))
                            print('{} Pred labels[{}]: {}'.format(phase, i, pred_label))
                            print('{} Pred probs [{}]: {}'.format(phase, i, pred_label_output_probs))

                    running_loss += loss.item() * inputs.size(0)
                    mf_f1 = f1_score(labels, preds, average='micro')
                    running_f1 += mf_f1 * inputs.size(0)  # torch.sum(preds == labels.data)

                    for key in th_selection_preds.keys():
                        th_f1 = f1_score(labels, th_selection_preds[key], average='micro')
                        running_th_f1[key] += th_f1 * inputs.size(0)

                    spot_loss = running_loss / (n_iter + 1)
                    if phase == 'train':
                        self.tensorboard.add_train_scalar('loss', spot_loss, global_step)
                        self.tensorboard.add_train_scalar('microF1', mf_f1, global_step)
                    else:
                        self.tensorboard.add_validation_scalar('loss', spot_loss, val_step)
                        self.tensorboard.add_validation_scalar('microF1', mf_f1, val_step)

                    if self.spot:
                        print('{} Spot:  Loss: {:.4f} F1: {:.4f} Step: {}'.format(phase, spot_loss, mf_f1, global_step))

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_f1 = running_f1 / self.dataset_sizes[phase]
                # print('F1 for different thresholds: ')
                sorted_thresholds = sorted(running_th_f1.items(), key=lambda x: x[1], reverse=True)
                if sorted_thresholds[0][1] > running_f1 and phase == 'val':
                    chosen_threshold = (chosen_threshold + sorted_thresholds[0][0]) / 2
                # for item in sorted_thresholds:
                #     print('{} Threshold: {}, F1: {}'.format(phase, item[0], item[1] / self.dataset_sizes[phase]))
                print('{} Loss: {:.4f} F1: {:.4f} Chosen threshold {:.4f} <------------------------------------------------------------------'.format(
                    phase, epoch_loss, epoch_f1, chosen_threshold))

                # deep copy the model
                if phase == 'val':
                    if epoch_f1 > best_f1:
                        best_f1 = epoch_f1
                        best_model_wts = copy.deepcopy(model.state_dict())
                        print('[i] Saving new best F1 {:.4f}'.format(best_f1))
                        self.model_saver.save_checkpoint(model, epoch, optimizer, global_step, True)
                        chosen_threshold = sorted_thresholds[0][0]

                    print("[i] Saving last epoch model.")
                    self.model_saver.save_checkpoint(model, epoch, optimizer, global_step, False)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val F1: {:4f}'.format(best_f1))
        print('Best threshold: {}'.format(chosen_threshold))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    ######################################################################
    # Visualizing the model predictions
    def visualize_model(self, model, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels, _) in enumerate(self.dataloaders['val']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                preds = outputs.ge(0.2).type(torch.cuda.FloatTensor)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images // 2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}'.format(self.mfb.inverse_transform(preds[j])))
                    self.imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)

    ######################################################################

    def train_fashion_model(self, num_epochs=10):
        model_ft = FashionModel()
        model_ft = model_ft.to(self.device)
        criterion = nn.MultiLabelSoftMarginLoss()

        params = list(model_ft.localization.parameters()) + \
                 list(model_ft.fc_loc.parameters()) + \
                 list(model_ft.resnet.fc.parameters()) + \
                 list(model_ft.fc.parameters())
        # params = list(model_ft.resnet.fc.parameters()) + list(model_ft.fc.parameters())
        optimizer_ft = optim.Adam(params)  # , lr=0.001, momentum=0.9)

        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)  # Decay LR by a factor of 0.1 every 7 epochs
        return self.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)

    def train_standard_resnet(self,num_epochs):
        # rescaling_weights, filtered_mask = check_label_distribution(dataloaders['train'])

        model_ft = models.resnet50(pretrained=True)

        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 228)
        model_ft = model_ft.to(self.device)

        criterion = nn.MultiLabelSoftMarginLoss()

        optimizer_ft = optim.Adam(model_ft.parameters())  # , lr=0.001, momentum=0.9)

        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  # Decay LR by a factor of 0.1 every 7 epochs
        return self.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)

    def train_extended_standard_resnet(self, num_epochs):
        # rescaling_weights, filtered_mask = check_label_distribution(dataloaders['train'])

        model_ft = nn.Sequential(models.resnet50(pretrained=True),
                                 nn.Linear(1000, 512, bias=True),
                                 nn.Dropout(),
                                 nn.ReLU(),
                                 nn.Linear(512, 228, bias=True)
                                 )

        model_ft = model_ft.to(self.device)

        criterion = nn.MultiLabelSoftMarginLoss()

        params = list(model_ft[0].fc.parameters()) + \
                 list(model_ft[1].parameters()) + \
                 list(model_ft[4].parameters())
        optimizer_ft = optim.Adam(params)

        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  # Decay LR by a factor of 0.1 every 7 epochs
        return self.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)


if __name__ == "__main__":
    # train_folder, train_json, val_folder, val_json = sys.argv[1:]
    # train_folder = '/media/spike/Scoob/materialist_fashion_data/'
    # train_json = 'train.json'
    # val_folder = '/media/spike/Scoob/materialist_fashion_val/'
    # val_json = 'validation.json'

    # base_folder = '/media/spike/Spike/Ubuntu/imageai/'
    # json_files = ['train_filtered2.json',
    #               'val_filtered2.json']
    base_folder = sys.argv[1]
    json_files = sys.argv[2:]
    batch_size = input('[?] Batch size: ')
    epochs = input('[?] Num epochs: ')
    model_choice = input('[?] Model: ')
    trainer = MfTrainer(base_folder, json_files, int(batch_size))
    # trainer = MfTrainer(train_folder, train_json, val_folder, val_json)
    show_plots = False
    if model_choice.startswith('st'):
        model_ft = trainer.train_standard_resnet(num_epochs=int(epochs))
    elif model_choice.startswith('ex'):
        model_ft = trainer.train_extended_standard_resnet(num_epochs=int(epochs))
    else:
        model_ft = trainer.train_fashion_model(num_epochs=int(epochs))

    if show_plots:
        # visualize_model(model_ft)
        plt.ioff()
        plt.show()
