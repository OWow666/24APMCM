import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from torchvision import transforms, models
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt


# UNet 模型定义
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64, time_emb_dim=128):
        super(UNet, self).__init__()

        # 时间步嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.enc1 = self.conv_block(in_channels + time_emb_dim, base_channels)  # 3 + time_emb_dim -> 64
        self.enc2 = self.conv_block(base_channels, base_channels * 2)  # 64 -> 128
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)  # 128 -> 256

        self.bottleneck = self.conv_block(base_channels * 4, base_channels * 8)  # 256 -> 512

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
        # 计算时间步嵌入并将其扩展到与图像相同的维度
        if isinstance(t, float) or isinstance(t, int):
            t = torch.tensor([t], dtype=torch.float32, device=x.device)
        t = t.float() / 1000.0  # 正则化时间步，使得值在合理范围内
        t_embed = self.time_mlp(t.view(-1, 1)).unsqueeze(-1).unsqueeze(-1)
        t_embed = t_embed.expand(-1, -1, x.shape[2], x.shape[3])

        # 将时间嵌入与输入图像连接
        x = torch.cat([x, t_embed], dim=1)

        enc1 = self.enc1(x)  # 64
        enc2 = self.enc2(self.pool(enc1))  # 128
        enc3 = self.enc3(self.pool(enc2))  # 256

        bottleneck = self.bottleneck(self.pool(enc3))  # 512

        dec2 = self.up2(bottleneck) + enc3  # 256
        dec1 = self.up1(dec2) + enc2  # 128
        dec0 = self.up0(dec1) + enc1  # 64

        out = self.final_conv(dec0)
        return out


# DDPM 扩散模型定义
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.tensor(np.clip(betas, 0, 0.999), dtype=torch.float32)


class DDPM:
    def __init__(self, model, img_size, timesteps, device):
        self.model = model
        self.img_size = img_size
        self.timesteps = timesteps
        self.device = device
        self.beta = cosine_beta_schedule(timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod).to(device)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod).to(device)

    def forward_diffusion(self, x0, t):
        noise = torch.randn_like(x0).to(self.device)
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        xt = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        return xt, noise

    def reverse_diffusion(self, xt, t):
        t_embed = torch.tensor([t], dtype=torch.float32).to(self.device)
        pred_noise = self.model(xt, t_embed)

        beta_t = self.beta[t].view(-1, 1, 1, 1)
        alpha_t = self.alpha[t].view(-1, 1, 1, 1)
        alpha_cumprod_t = self.alpha_cumprod[t].view(-1, 1, 1, 1)
        alpha_cumprod_prev_t = self.alpha_cumprod_prev[t].view(-1, 1, 1, 1)

        mean = (1 / torch.sqrt(alpha_t)) * (xt - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * pred_noise)

        # 在最后几个时间步减少噪声的加入
        if t < 50:  # 在最后20个时间步减少噪声
            var = torch.zeros_like(beta_t)
        else:
            var = torch.sqrt(beta_t)

        z = torch.randn_like(xt).to(self.device) if t > 0 else 0
        xt_prev = mean + var * z
        return xt_prev

    def generate(self, batch_size):
        xt = torch.randn(batch_size, 3, *self.img_size).to(self.device)
        for t in reversed(range(self.timesteps)):
            xt = self.reverse_diffusion(xt, t)
        return xt


vgg = models.vgg16(pretrained=True).features.to('cuda' if torch.cuda.is_available() else 'cpu').eval()

def perceptual_loss(pred, target):
    pred_features = vgg(pred)
    target_features = vgg(target)
    loss = F.mse_loss(pred_features, target_features)
    return loss

# 模型训练定义
def train_model(model, diffusion, dataloader, optimizer, epochs, device):
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            clean_imgs = batch[0].to(device)
            optimizer.zero_grad()

            t = torch.randint(0, diffusion.timesteps, (clean_imgs.shape[0],), device=device)
            xt, noise = diffusion.forward_diffusion(clean_imgs, t)
            pred_noise = model(xt, t)

            mse_loss = F.mse_loss(pred_noise, noise)
            percep_loss = perceptual_loss(xt, clean_imgs)
            loss = mse_loss + 0.1 * percep_loss  # 结合感知损失
            loss.backward()
            optimizer.step()

        scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")


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

# 加载数据
dataset_path = os.path.join(os.getcwd(), 'EUVP', 'Paired', 'underwater_dark')
imgs_path = os.path.join(dataset_path, 'trainA')
clear_imgs_path = os.path.join(dataset_path, 'trainB')
transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()])
train_dataset = UnderWaterDataset(imgs_path, clear_imgs_path, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
print("Data has been Prepared!")

# 模型和优化器定义
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = (256, 256)
timesteps = 1000
model = UNet(in_channels=3, out_channels=3).to(device)
diffusion = DDPM(model, img_size, timesteps, device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 模型训练
train_model(model, diffusion, train_loader, optimizer, epochs=10, device=device)

# 保存模型
torch.save(model.state_dict(), "enhanced_ddpm_model.pth")
print("Model has been saved!")

# 使用模型对目标图像进行增强
def enhance_image(model, diffusion, img_path, output_path):
    model.eval()
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        enhanced_img_tensor = diffusion.generate(batch_size=1)
        enhanced_img = enhanced_img_tensor.squeeze(0).cpu().clamp(0, 1)
        save_image(enhanced_img, output_path)

    print(f"Image saved to {output_path}")


# 增强图像
input_img_path = "./Attachment/Attachment_2/test_001.png"  # 替换为目标图片路径
output_img_path = "./test_001_enhanced_image.jpg"
enhance_image(model, diffusion, input_img_path, output_img_path)

print("All tasks completed.")
