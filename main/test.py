import torch

if torch.cuda.is_available():
    import torch as torch
else:
    import torch as torch
import torchvision
from torch.autograd import Variable
from torchvision.transforms import Normalize, ToTensor, Resize,CenterCrop
from main import dataset, model
import torch.nn as nn
from torchvision import models

use_cuda = torch.cuda.is_available()
image_size = 224
scale = Resize((image_size, image_size))
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
tforms = torchvision.transforms.Compose([Resize(256),CenterCrop(224),ToTensor(),normalize])
# tforms = torchvision.transforms.Compose([scale, ToTensor(), normalize])
batch = 1
total_images = 39706
mf_test_set = dataset.TestMaterialistFashion('/media/spike/Scoob/materialist_fashion_test/', total_images, tforms)
mf_test_loader = torch.utils.data.DataLoader(mf_test_set, batch_size=batch, shuffle=False, num_workers=8)
print("Size of test loader: {}".format(len(mf_test_loader)))

if use_cuda:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 228)
    model_ft = model_ft.to(device)
    net = model_ft
    # net = model.Net(228).cuda()
else:
    print('not here')
    # net = model.Net(228)

net.load_state_dict(torch.load('../mymodel.pt'))
batch_number = 0
idx = 1
to_write = 'image_id,label_id\r\n'
for data in mf_test_loader:
    images = data
    if use_cuda:
        outputs = net(Variable(images).cuda())
    else:
        outputs = net(Variable(images))
    predicted = outputs.data.cpu().numpy()
    _, y = predicted.shape
    labels = ''
    for j in range(0, y):
        if predicted[0][j] >= 1 / 2:
            labels += '{} '.format(j + 1)
    line = '{},{}\r\n'.format(idx, labels.strip())
    to_write += line
    idx += 1
    if idx % 100 == 99:
        print('{}'.format(line))
print(to_write)
f = open('../001.csv', 'w')
f.write(to_write)
f.close()
