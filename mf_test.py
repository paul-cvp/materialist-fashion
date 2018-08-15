import numpy as np
import torch

if torch.cuda.is_available():
    import torch as torch
else:
    import torch as torch
import torchvision
from torchvision.transforms import Normalize, ToTensor, Resize
from main import dataset
from fashion_model import FashionModel
from sklearn.preprocessing import MultiLabelBinarizer

torch.manual_seed(42)

if __name__ == "__main__":
    test_folder = '/media/spike/Scoob/materialist_fashion_test/'
    cvs_filename = input("[?] Output csv name: ") # '001'
    use_cuda = torch.cuda.is_available()
    image_size = 224
    scale = Resize((image_size, image_size))
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tforms = torchvision.transforms.Compose([scale, ToTensor(), normalize])
    batch = 32
    total_images = 39706
    mf_test_set = dataset.TestMaterialistFashion(test_folder, total_images, tforms)
    mf_test_loader = torch.utils.data.DataLoader(mf_test_set, batch_size=batch, shuffle=False, num_workers=8)
    print("Size of test loader: {}".format(len(mf_test_loader)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FashionModel()
    model_state = torch.load('tmp/model_state_best.th')
    model.load_state_dict(model_state)
    batch_number = 0
    idx = 1
    mfb = MultiLabelBinarizer()
    mf_labels = np.arange(0, 228)
    mfb.fit_transform([mf_labels])
    to_write = 'image_id,label_id\r\n'
    for data in mf_test_loader:
        outputs = model(data)
        predicted = outputs.ge(0.2)
        _, y = predicted.shape
        for _, preds in enumerate(mfb.inverse_transform(predicted)):
            labels = ''
            for x in preds:
                labels += '{} '.format(x)
            line = '{},{}\r\n'.format(idx, labels.strip())
            to_write += line
            idx += 1
            print('{}'.format(line))

    with open('{}.csv'.format(cvs_filename), 'w') as f:
        f.write(to_write)
        f.close()
    print("[i] Done.")