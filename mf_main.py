import torch
import torchvision
import torch.optim as optim
from torchvision.transforms import Normalize, ToTensor, Resize, CenterCrop, RandomResizedCrop
import numpy as np


## Utilities
import random
from timeit import default_timer as timer
import os

##src
from src.mf_models import ResNet50
from mf_dataset import MaterialistFashion,TestMaterialistFashion
from src.mf_train import train, snapshot
from src.mf_validate import validate
from src.mf_predict import predict, output

############################################################################
## Variables setup
use_cuda = torch.cuda.is_available()

image_size = 224
scale = Resize((image_size, image_size))
center_crop = CenterCrop(image_size)
random_crop = RandomResizedCrop(image_size)
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_percent = 50  # any value from 1 to 100
val_percent = 100

mf_labels = np.arange(1, 229)
NUM_CLASSES = len(mf_labels)
model = ResNet50(NUM_CLASSES).cuda()

epochs = 10
batch_size = 32

# Note, p_training has lr_decay automated
# optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005) # Finetuning whole model
optimizer = optim.Adam(model.parameters())

# Decay LR by a factor of 0.1 every 5 epochs
# lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

criterion = torch.nn.BCEWithLogitsLoss().cuda()
# criterion = torch.nn.MultiLabelSoftMarginLoss().cuda()


if __name__ == "__main__":
    # Initiate timer
    global_timer = timer()

    # Setting random seeds for reproducibility. (Caveat, some CuDNN algorithms are non-deterministic)
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)

    ##############################################################
    ## Loading the dataset
    mf_train = MaterialistFashion('/media/spike/Scoob/materialist_fashion_data/', 'train.json',
                                  torchvision.transforms.Compose([random_crop, ToTensor(), normalize]),
                                  train_percent)
    mf_val = MaterialistFashion('/media/spike/Scoob/materialist_fashion_val/', 'validation.json',
                                torchvision.transforms.Compose([scale, ToTensor(), normalize]),
                                 val_percent)

    mf_train_loader = torch.utils.data.DataLoader(mf_train, batch_size=batch_size, shuffle=True, num_workers=8)
    mf_val_loader = torch.utils.data.DataLoader(mf_val, batch_size=batch_size, shuffle=False, num_workers=8)

    print("Size of train loader: {}".format(len(mf_train_loader)))
    print("Size of val loader: {}".format(len(mf_val_loader)))
    ###########################################################
    ## Start training
    best_score = 0.
    for epoch in range(epochs):
        epoch_timer = timer()

        # Train and validate
        train(epoch, mf_train_loader, model, criterion, optimizer)
        score, loss, threshold = validate(epoch, mf_val_loader, model, criterion, mf_train.getLabelEncoder())
        # Save
        is_best = score > best_score
        best_score = max(score, best_score)
        snapshot(is_best, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': best_score,
            'optimizer': optimizer.state_dict(),
            'threshold': threshold,
            'val_loss': loss
        })

        end_epoch_timer = timer()
        print("#### End epoch {}, elapsed time: {}".format(epoch, end_epoch_timer - epoch_timer))

    ###########################################################
    ## Prediction
    total_images = 39706
    mf_test = TestMaterialistFashion('/media/spike/Scoob/materialist_fashion_test/',
                                     total_images,
                                     torchvision.transforms.Compose([scale, ToTensor(), normalize]))
    test_loader = torch.utils.data.DataLoader(mf_test,
                             batch_size=batch_size,
                             num_workers=8,
                             pin_memory=True)

    # Load model from best iteration
    print('===> loading best model for prediction')
    checkpoint = torch.load(os.path.join('mymodel.pt')
                            )
    model.load_state_dict(checkpoint['state_dict'])

    # Predict
    predictions = predict(test_loader, model)  # TODO load model from the best on disk

    output(predictions,
           checkpoint['threshold'],
           mf_test,
           mf_train.getLabelEncoder(),
           './out',
           'mf',
           checkpoint['best_score'])  # TODO early_stopping and use best_score

    ##########################################################

    end_global_timer = timer()
    print("################## Success #########################")
print("Total elapsed time: %s" % (end_global_timer - global_timer))