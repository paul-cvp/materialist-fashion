import os
import os.path
import numpy as np

import json
import pandas as pd
from pandas.io.json import json_normalize
import torch

if torch.cuda.is_available():
    import torch as torch
else:
    import torch as torch
import torch.utils.data as data

from PIL import Image
from random import randint
from sklearn.preprocessing import MultiLabelBinarizer


class TestMaterialistFashion(data.Dataset):

    def __init__(self, image_folder, total_images, transform=None):
        self.transform = transform
        self.image_id = []
        self.image_folder = image_folder
        for i in range(1, total_images + 1):
            self.image_id.append(i)

    def __getitem__(self, index):
        image_data = self.default_loader(os.path.join(self.image_folder, '{}{}'.format(self.image_id[index],'.jpg')))
        if self.transform:
            image_data = self.transform(image_data)
        return image_data

    def __len__(self):
        return len(self.image_id)

    def default_loader(self, path):
        return Image.open(path).convert('RGB')


class MaterialistFashion(data.Dataset):

    def __init__(self, image_folder, json_file, transform=None, percent=100, load_first=-1):
        self.mlb = MultiLabelBinarizer()
        self.transform = transform
        self.image_id = []
        self.annotations = []
        self.image_folder = image_folder
        with open(json_file, 'r') as f:
            data = pd.DataFrame.from_dict(json_normalize(json.load(f)['annotations']), orient='columns')
            self.annotations = self.mlb.fit_transform(data['labelId']).astype(np.float32)
            x, y = self.annotations.shape
            if y != 228:
                to_concat = 228-y
                self.annotations = np.concatenate((self.annotations, np.zeros((x, to_concat))), axis=1)

            self.image_id = data['imageId']
            if percent < 100:
                temp_annotations = []
                temp_image_id = []
                for i in range(0, len(self.image_id)):
                    if randint(0, 100) <= percent:
                        temp_annotations.append(self.annotations[i])
                        temp_image_id.append(self.image_id[i])
                self.annotations = temp_annotations
                self.image_id = temp_image_id
                self.length = len(self.image_id)
            elif load_first > 0:
                self.image_id = self.image_id[:load_first]
                self.annotations = self.annotations[:load_first]
                self.length = len(self.image_id)
            else:
                self.length = len(self.image_id.index)

    def __getitem__(self, index):
        image_data = self.default_loader(os.path.join(self.image_folder, self.image_id[index] + '.jpg'))
        if self.transform:
            image_data = self.transform(image_data)
        labels = self.annotations[index]
        multihot_label = torch.from_numpy(labels)
        #multihot_label = multihot_label.type(torch.LongTensor)
        return image_data, multihot_label, self.image_id[index]

    def __len__(self):
        return self.length

    def default_loader(self, path):
        return Image.open(path).convert('RGB')

    def getLabelEncoder(self):
        return self.mlb

