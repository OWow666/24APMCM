import torch
import torch.nn as nn
from sympy.printing.cxx import reserved
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = os.path.join(os.getcwd(), 'EUVP', 'Paired', 'underwater_dark')
imgs_path = os.path.join(dataset_path, 'trainA')
clear_imgs_path = os.path.join(dataset_path, 'trainB')

class UnderWaterDataset(Dataset):
    def __init__(self, origin_path, res_path, transform=None):
        self.origin_path = origin_path
        self.res_path = res_path
        self.transform = transform
        self.origin_imgs = os.listdir(self.origin_path)
        self.res_imgs = os.listdir(self.res_path)
    
    def __len__(self):
        return len(self.origin_imgs)
    
    def __getitem__(self, idx):
        origin_img_path = os.path.join(self.origin_path, self.origin_imgs[idx])
        res_img_path = os.path.join(self.res_path, self.res_imgs[idx])
        origin_img = Image.open(origin_img_path).convert('RGB')
        res_img = Image.open(res_img_path).convert('RGB')
        if self.transform:
            origin_img = self.transform(origin_img)
            res_img = self.transform(res_img)
        return origin_img, res_img

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()])
train_dataset = UnderWaterDataset(imgs_path, clear_imgs_path, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
print("Data has been Prepared!")


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64):
        super(UNet, self).__init__()

        self.enc1 = self.conv_block(in_channels, base_channels)  # 3 -> 64
        self.enc2 = self.conv_block(base_channels, base_channels * 2)  # 64 -> 128
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)  # 128 -> 256
        #self.enc4 = self.conv_block(base_channels * 4, base_channels * 8)  # 256 -> 512

        self.bottleneck = self.conv_block(base_channels * 4, base_channels * 8)  # 512 -> 1024 / 256->512

        #self.up3 = self.upsample_block(base_channels * 16, base_channels * 8)  # 1024 -> 512
        self.up2 = self.upsample_block(base_channels * 8, base_channels * 4)  # 512 -> 256
        self.up1 = self.upsample_block(base_channels * 4, base_channels * 2)  # 256 -> 128
        self.up0 = self.upsample_block(base_channels * 2, base_channels)  # 128 -> 64

        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, t):
        enc1 = self.enc1(x)  # 64
        enc2 = self.enc2(self.pool(enc1))  # 128
        enc3 = self.enc3(self.pool(enc2))  # 256
        #enc4 = self.enc4(self.pool(enc3))  # 512

        bottleneck = self.bottleneck(self.pool(enc3))  # 1024

        #dec3 = self.up3(bottleneck) + enc4  # 512
        dec2 = self.up2(bottleneck) + enc3  # 256
        dec1 = self.up1(dec2) + enc2  # 128
        dec0 = self.up0(dec1) + enc1  # 64

        out = self.final_conv(dec0)
        return out

import torch.nn.functional as F
class DiffusionModel:
    def __init__(self, model, img_size, timesteps, device):
        self.model = model
        self.img_size = img_size
        self.timesteps = timesteps
        self.device = device
        self.beta = torch.linspace(0.0001, 0.02, timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)
    
    def forward_diffusion(self, x0, t):
        noise = torch.randn_like(x0).to(self.device)
        sqrt_alpha_cumprod_t = torch.sqrt(self.alpha_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1-self.alpha_cumprod[t])[:, None, None, None]
        xt = sqrt_one_minus_alpha_cumprod_t * x0 + sqrt_alpha_cumprod_t * noise
        return xt, noise
    
    def reverse_diffusion(self, xt, t):
        pred_noise = self.model(xt,t)
        beta_t = self.beta[t][:, None, None, None]
        alpha_t = self.alpha[t][:, None, None, None]
        alpha_cumprod_t = self.alpha_cumprod[t][:, None, None, None]
        alpha_cumprod_prev_t = self.alpha_cumprod_prev[t][:, None, None, None]
        
        mean = 1 / torch.sqrt(alpha_t) * (xt-beta_t / torch.sqrt(1-alpha_cumprod_t) * pred_noise)
        var = torch.sqrt(1 - alpha_cumprod_prev_t)
        
        z = torch.randn_like(xt).to(self.device) if t > 0 else 0
        xt_prev = mean + var * z
        return xt_prev
    
    def generate(self, batch_size):
        xt = torch.randn(batch_size, 3, *self.img_size).to(self.device)
        for t in reserved(range(self.timesteps)):
            xt = self.reverse_diffusion(xt, torch.full((batch_size,), t, dtype=torch.long).to(self.device))
        return xt

def train_model(model, diffusion, dataloader, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            noisy_imgs, clean_imgs = batch
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            
            t = torch.randint(0, diffusion.timesteps, (noisy_imgs.shape[0],)).to(device)
            xt, noise = diffusion.forward_diffusion(clean_imgs, t)
            
            optimizer.zero_grad()
            pred_noise = model(xt, t)
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
if __name__ == "__main__":
    img_size = (256,256)
    timesteps = 1000

    model = UNet(in_channels=3, out_channels=3, base_channels=64).to(device)
    diffusion = DiffusionModel(model, img_size, timesteps, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("Start Training!")
    train_model(model, diffusion, train_loader, optimizer,epochs=10, device=device)

    torch.save(model.state_dict(), './diffusion_model.pth')
    print("Done! Model has been saved!")