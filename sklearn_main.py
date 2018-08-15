import torch
import torchvision
from torchvision.models import resnet50
from torchvision.transforms import Normalize, ToTensor, Resize, CenterCrop, RandomResizedCrop
from torch.autograd import Variable

## Utilities
import random
from timeit import default_timer as timer
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib

from sklearn.preprocessing import MultiLabelBinarizer

##src
from mf_dataset import MaterialistFashion, TestMaterialistFashion

############################################################################
## Global Variables setup
mf_labels = np.arange(1, 229)
NUM_CLASSES = len(mf_labels)
current_classifier_name = 'mf_sklearn_dual0.pkl'
test_file_name = '004.csv'
image_size = 224
scale = Resize((image_size, image_size))
center_crop = CenterCrop(image_size)
random_crop = RandomResizedCrop(image_size)
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

batch_size = 32


def train_images_to_feature_vectors(model):
    mf_train = MaterialistFashion('/media/spike/Scoob/materialist_fashion_data/', 'train.json',
                                  torchvision.transforms.Compose([scale, ToTensor(), normalize]),
                                  10)

    mf_train_loader = torch.utils.data.DataLoader(mf_train, batch_size=batch_size, shuffle=True, num_workers=8)

    X_train = []
    y_train = []
    size = len(mf_train_loader)
    print("Size of train loader: {0}".format(size))
    model.eval()
    for batch_idx, (data, target, _) in enumerate(mf_train_loader):
        batch_idx = batch_idx + 1
        print("Batch {0} of {1} ( {2}% )".format(batch_idx, size, 100. * batch_idx / size))
        data, target = data.cuda(async=True), target.cuda(async=True)
        data, target = Variable(data), Variable(target, requires_grad=False)
        data_fv = model(data).data.cpu().numpy()
        target = target.data.cpu().numpy()
        X_train.extend(data_fv)
        y_train.extend(target)

    print('Length of X_train: {0} |#| Length of y_train {1}'.format(len(X_train), len(y_train)))

    np.save('X_train', np.array(X_train))
    np.save('y_train', np.array(y_train))
    del X_train
    del y_train
    del mf_train
    del mf_train_loader
    del model
    print('Done preparing new training arrays.')


def val_images_to_feature_vectors(model):
    mf_val = MaterialistFashion('/media/spike/Scoob/materialist_fashion_val/', 'validation.json',
                                torchvision.transforms.Compose([scale, ToTensor(), normalize]),
                                100)
    mf_val_loader = torch.utils.data.DataLoader(mf_val, batch_size=batch_size, shuffle=False, num_workers=8)
    X_val = []
    y_val = []
    size = len(mf_val_loader)
    print("Size of val loader: {0}".format(size))
    for batch_idx, (data, target, _) in enumerate(mf_val_loader):
        batch_idx = batch_idx + 1
        print("Batch {0} of {1} ( {2}% )".format(batch_idx, size, 100. * batch_idx / size))
        data, target = data.cuda(async=True), target.cuda(async=True)
        data, target = Variable(data), Variable(target, requires_grad=False)
        data_fv = model(data).data.cpu().numpy()
        target = target.data.cpu().numpy()
        X_val.extend(data_fv)
        y_val.extend(target)

    print('Length of X_val: {0} |#| Length of y_val {1}'.format(len(X_val), len(y_val)))

    np.save('X_val', np.array(X_val))
    np.save('y_val', np.array(y_val))
    del X_val
    del y_val
    del mf_val
    del mf_val_loader


def test_images_to_feature_vectors(model):
    total_images = 39706
    mf_test = TestMaterialistFashion('/media/spike/Scoob/materialist_fashion_test/',
                                     total_images,
                                     torchvision.transforms.Compose([scale, ToTensor(), normalize]))
    mf_test_loader = torch.utils.data.DataLoader(mf_test, batch_size=batch_size, shuffle=False, num_workers=8)
    X_test = []
    size = len(mf_test_loader)
    print("Size of test loader: {0}".format(size))
    for batch_idx, data in enumerate(mf_test_loader):
        batch_idx = batch_idx + 1
        print("Batch {0} of {1} ( {2}% )".format(batch_idx, size, 100. * batch_idx / size))
        data = data.cuda(async=True)
        data = Variable(data)
        data_fv = model(data).data.cpu().numpy()
        X_test.extend(data_fv)

    print('Length of X_test {0}'.format(len(X_test)))

    np.save('X_test', np.array(X_test))
    del X_test
    del mf_test
    del mf_test_loader


def run():
    ##############################################################
    ## local variable setup
    mlb = MultiLabelBinarizer(classes=mf_labels)
    mlb.fit(mf_labels)
    is_training = True
    is_validating = True
    is_testing = True

    # lm = LinearSVC(dual=True, class_weight='balanced', random_state=0, verbose=2, max_iter=500, tol=1e-4)
    lm = LinearSVC(dual=False, class_weight='balanced', verbose=2)
    classifier = OneVsRestClassifier(lm, n_jobs=-1)

    ###########################################################
    ## Start training
    if is_training:
        X = np.load('X_train.npy', mmap_mode='r')
        Y = np.load('y_train.npy', mmap_mode='r')
        print("### Classifier fitting X size: {0} y size: {1}".format(len(X), len(Y)))
        classifier.fit(X, Y)
        # joblib.dump(classifier, current_classifier_name)
    ###########################################################
    ## Prediction
    if is_validating:
        print("## Prediction")
        # classifier = joblib.load(current_classifier_name)
        X_val = np.load('X_val.npy', mmap_mode='r')
        y_val = np.load('y_val.npy', mmap_mode='r')
        X_id = np.arange(1, 9897)
        predicted = classifier.predict(X_val)
        pred_labels = mlb.inverse_transform(predicted)
        groud_truth = mlb.inverse_transform(y_val)
        for id, predicted, gt in zip(X_id, pred_labels, groud_truth):
            print('ID: {0} =>\r\nPredicted: {1} \r\nGround truth: {2}'.format(str(id),
                                                                              ', '.join(str(k) for k in predicted),
                                                                              ', '.join(str(k) for k in gt)))
    ###########################################################
    ## Test
    if is_testing:
        print('## Test')
        # classifier = joblib.load(current_classifier_name)
        X_test = np.load('X_test.npy', mmap_mode='r')
        predicted = classifier.predict(X_test)
        pred_labels = mlb.inverse_transform(predicted)
        to_write = 'image_id,label_id\r\n'
        for idx, predicted in enumerate(pred_labels):
            i = idx + 1
            to_write += '{0},{1}\r\n'.format(str(i), ' '.join(str(k) for k in predicted))
        f = open(test_file_name, 'w')
        f.write(to_write)
        f.close()
    return classifier


if __name__ == "__main__":
    # multiprocessing.set_start_method('forkserver')
    # Initiate timer
    global_timer = timer()

    # Setting random seeds for reproducibility. (Caveat, some CuDNN algorithms are non-deterministic)
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)

    # images_to_feature_vectors()
    model = resnet50(pretrained=True).cuda()
    train_images_to_feature_vectors(model)
    classifier = run()
    joblib.dump(classifier, current_classifier_name)
    end_global_timer = timer()
    print("################## Success #########################")
print("Total elapsed time: %s" % (end_global_timer - global_timer))
