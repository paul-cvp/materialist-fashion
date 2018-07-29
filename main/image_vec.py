import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image

class ImageVector():

    def __init__(self):
        # Load the pretrained model
        self.model = models.resnet18(pretrained=True)

        # Use the model object to select the desired layer
        self.layer = self.model._modules.get('avgpool')

        # Set model to evaluation mode
        self.model.eval()

        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def get_vector(self,image_name):
        # 1. Load the image with Pillow library
        img = Image.open(image_name)

        # 2. Create a PyTorch Variable with the transformed image
        t_img = Variable(self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0))

        # 3. Create a vector of zeros that will hold our feature vector
        #    The 'avgpool' layer has an output size of 512
        my_embedding = torch.zeros(512)

        # 4. Define a function that will copy the output of a layer
        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        # 5. Attach that function to our selected layer
        h = self.layer.register_forward_hook(copy_data)

        # 6. Run the model on our transformed image
        self.model(t_img)

        # 7. Detach our copy function from the layer
        h.remove()

        # 8. Return the feature vector
        return my_embedding

