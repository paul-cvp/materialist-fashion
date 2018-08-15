import numpy as np
import torch

from PIL import Image
if torch.cuda.is_available():
    import torch as torch
else:
    import torch as torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import Normalize, ToTensor, Resize
from fashion_model import FashionModel
from sklearn.preprocessing import MultiLabelBinarizer

torch.manual_seed(42)
plt.ion()


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

if __name__ == "__main__":
    model_path = input("[?] Model path: ")

    run = True
    while run:
        image_path = input("[?] Image path: ")
        if image_path != 'e':
            image = Image.open(image_path).convert('RGB')
            use_cuda = torch.cuda.is_available()
            image_size = 224
            scale = Resize((image_size, image_size))
            normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            tforms = torchvision.transforms.Compose([scale, ToTensor(), normalize])
            image_data = tforms(image)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = FashionModel()
            model_state = torch.load(model_path)
            model.load_state_dict(model_state)
            batch_number = 0
            mfb = MultiLabelBinarizer()
            mf_labels = np.arange(0, 228)
            mfb.fit_transform([mf_labels])
            outputs = model(image_data)
            predicted = outputs.ge(0.2)
            labels = mfb.inverse_transform(predicted)
            print("[i] Labels: {}".format(labels))
            preview_stn = model.stn(image_data)
            imshow(torchvision.utils.make_grid(torch.cat((image_data,preview_stn))),title=labels)
        else:
            run = False
    plt.ioff()
    plt.show()
    print("[i] Done.")