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
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64):
        super(UNet, self).__init__()
        
        self.enc1 = self.conv_block(in_channels, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels * 2)
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)
        
        self.bottleneck = self.conv_block(base_channels * 4, base_channels * 8)
        
        self.dec3 = self.conv_block(base_channels * 8, base_channels * 4)
        self.dec2 = self.conv_block(base_channels * 4, base_channels * 2)
        self.dec1 = self.conv_block(base_channels * 2, base_channels)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, kernel_size=2, stride=2)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        
        bottleneck = self.bottleneck(self.pool(enc3))
        
        dec3 = self.upsample(bottleneck) + enc3
        dec3 = self.dec3(dec3)
        dec2 = self.upsample(dec3) + enc2
        dec2 = self.dec2(dec2)
        dec1 = self.upsample(dec2) + enc1
        dec1 = self.dec1(dec1)
        
        return dec1

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

img_size = (256,256)
timesteps = 1000

model = UNet(in_channels=3, out_channels=3).to(device)
diffsion = DiffusionModel(model, img_size, timesteps, device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
train_model(model, diffsion, train_loader, optimizer,epochs=10, device=device)

torch.save(model.state_dict(), './diffusion_model.pth')