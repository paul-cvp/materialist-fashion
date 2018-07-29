import json
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_multilabel_classification
from skmultilearn.adapt import MLkNN
import os.path


def load_data(filename, image_folder, percent=100):
    X = []
    y = []
    with open(filename, 'r') as f:
        data = json.load(f)
        for annotation in data['annotations']:
            if randint(0, 100) <= percent:
                image_id = int(annotation['imageId'])
                im_data = io.imread(os.path.join(image_folder, '{}.jpg'.format(image_id)))
                im_data = resize(im_data, (32, 32))
                X.append(np.array(im_data).flatten())
                labels = []
                multihot_labels = []
                for l in annotation['labelId']:
                    labels.append(int(l))
                for i in range(1, 229):
                    if i not in labels:
                        multihot_labels.append(0)
                    else:
                        multihot_labels.append(1)
                y.append(multihot_labels)

    X = np.matrix(X)
    y = np.matrix(y)
    return X, y


# filename = "../train.json"
# image_folder = '/media/spike/Scoob/materialist_fashion_data/'
# print('[i] Loading train data please wait')
# X_train, y_train = load_data(filename, image_folder, 1)
#
# filename = "../validation.json"
# image_folder = '/media/spike/Scoob/materialist_fashion_val/'
# print('[i] Loading validation data please wait')
# X_test, y_test = load_data(filename, image_folder, 10)

X_train, y_train = make_multilabel_classification(sparse=True, n_labels=228, return_indicator='sparse',
                                                  allow_unlabeled=False)

X_test, y_test = make_multilabel_classification(sparse=True, n_labels=228, return_indicator='sparse',
                                                allow_unlabeled=False)

classifier = MLkNN(k=228)

# train
print('[i] Training')
classifier.fit(X_train, y_train)

# predict
print('[i] Predicting')
predictions = classifier.predict(X_test)

score = accuracy_score(y_test, predictions)

print('[i] Done! Accuracy score: {}'.format(score))
