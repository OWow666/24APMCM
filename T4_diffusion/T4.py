from time import sleep

import numpy as np
import torch
import os
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from T4_train import UNet, DiffusionModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target_imgs_path = os.path.join(os.getcwd(), 'Attachment', "Attachment_2")
base_output_imgs_path = os.path.join(os.getcwd(), 'imgs_output')
if not os.path.exists(base_output_imgs_path):
    os.makedirs(base_output_imgs_path)
print("Images have been prepared!")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()])
img_size = (256,256)
timesteps = 1000

model = UNet(in_channels=3, out_channels=3).to(device)
model.load_state_dict(torch.load("./diffusion_model.pth"))
model.eval()
diffusion = DiffusionModel(model, img_size, timesteps, device)
print("Model has been prepared!")

def tensor_to_img(tensor):
    max_val = tensor.max()
    min_val = tensor.min()
    tensor_normalized = (tensor - min_val) / (max_val - min_val)
    tensor_normalized = tensor_normalized.cpu().detach().numpy().transpose(1, 2, 0)
    tensor_normalized = (tensor_normalized * 255).clip(0, 255).astype(np.uint8)
    img = transforms.ToPILImage()(tensor_normalized)
    return img

for img in os.listdir(target_imgs_path):
    img_path = os.path.join(target_imgs_path, img)
    img_name = img.split('.')[0]
    img_extension = img.split('.')[1]
    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = Image.open(img_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            generated_img_tensor = diffusion.generate(batch_size=1)
            #print(generated_img_tensor)
            generated_img = tensor_to_img(generated_img_tensor.squeeze(0))
        #print(generated_img.size())

        output_imgs_path = os.path.join(base_output_imgs_path, f"generated_{img_name}.{img_extension}")
        #print(generated_img.dtype)
        generated_img.save(output_imgs_path)
        #save_image(generated_img, output_imgs_path)
        
        print(f'generated_{img_name}.{img_extension} has been generated!')
print("Images have all been generated!")