import os

import torch
from PIL import Image
from torchvision import transforms

from T4_train import UNet

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

input_imgs_path = "./Attachment/Attachment_2"
output_imgs_path = "./imgs_output"
if not os.path.exists(output_imgs_path):
    os.makedirs(output_imgs_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./UNet_model.pth"
model = UNet(3, 3, 64).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

def tensor_to_img(tensor):
    tensor = tensor.squeeze(0).cpu().detach().numpy().transpose((1, 2, 0))
    tensor = (tensor * 255).clip(0, 255).astype('uint8')
    img = Image.fromarray(tensor)
    return img

for img in os.listdir(input_imgs_path):
    img_path = os.path.join(input_imgs_path, img)
    input_img = Image.open(img_path).convert('RGB')
    input_img_tensor = transform(input_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output_img_tensor = model(input_img_tensor)
        output_img = tensor_to_img(output_img_tensor)
    output_img.save(os.path.join(output_imgs_path, f'enhanced_{img}'))

print("Done!")