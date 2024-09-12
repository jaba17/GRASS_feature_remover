import torch
from torchvision import transforms
from PIL import Image
from .networks import define_G
from collections import OrderedDict
import torch
import torch.nn as nn


class Pix2Pix(nn.Module):

    def __init__(self, config):
        super(Pix2Pix, self).__init__()
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model_url = config["model_url"]
        self.model = self.prepare_model(self.model_url)

    def prepare_model(self, model_url):
        model_dict = torch.load(model_url)
        new_dict = OrderedDict()    
        for k, v in model_dict.items():
            print(k)
            new_dict["module." + k] = v

        generator_model = define_G(input_nc=3
                                    ,output_nc=3,ngf=64,netG="unet_256",
                                    norm="batch",use_dropout=True,init_gain=0.02,gpu_ids=[0])
        print(generator_model)
        generator_model.load_state_dict(new_dict)
        generator_model.to(self.device)

        return generator_model
    
    # Function to perform inference
    def infer(self, path):
        image_tensor = self.load_image_as_tensor(path)
        self.model.eval()  # Set the model to evaluation mode
        output = self.forward(image_tensor)

        return output

    def __crop(self, img, pos=(0, 0), size=256):
        ow, oh = img.size
        x1, y1 = pos
        tw = th = size
        if (ow > tw or oh > th):
            return img.crop((x1, y1, x1 + tw, y1 + th))
        
        return img

    def forward(self, image_tensor):
        with torch.no_grad():  # Disable gradient calculation for inference
            output = self.model(image_tensor).detach().cpu()

        return output

    # Function to load an image and convert it to a tensor
    def load_image_as_tensor(self, image_path):

        transform = transforms.Compose([
            transforms.Resize((256, 256), antialias=True),
            transforms.Lambda(lambda img: self.__crop(img)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        # Open the image file
        image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
        
        # Apply the transformations
        image_tensor = transform(image)
        image_tensor = image_tensor.to(self.device)

        # Add a batch dimension
        image_tensor = image_tensor.unsqueeze(0)  # Shape becomes [1, 3, 224, 224]
        
        return image_tensor




