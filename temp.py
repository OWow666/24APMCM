import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.utils import save_image


# --------------------------------------------
# 1. UNet模型（扩散模型的核心）
# --------------------------------------------
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels * 2)
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)

        # Bottleneck
        self.bottleneck = self.conv_block(base_channels * 4, base_channels * 8)

        # Decoder
        self.dec3 = self.conv_block(base_channels * 8, base_channels * 4)
        self.dec2 = self.conv_block(base_channels * 4, base_channels * 2)
        self.dec1 = self.conv_block(base_channels * 2, out_channels)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, kernel_size=2, stride=2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))

        # Decoder
        dec3 = self.upsample(bottleneck) + enc3
        dec3 = self.dec3(dec3)
        dec2 = self.upsample(dec3) + enc2
        dec2 = self.dec2(dec2)
        dec1 = self.upsample(dec2) + enc1
        dec1 = self.dec1(dec1)

        return dec1


# --------------------------------------------
# 2. 扩散过程定义
# --------------------------------------------
class DiffusionModel:
    def __init__(self, model, img_size, timesteps, device):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        self.img_size = img_size
        self.beta = torch.linspace(0.0001, 0.02, timesteps).to(device)  # 噪声调度
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)

    def forward_diffusion(self, x0, t):
        """前向扩散过程：添加噪声"""
        noise = torch.randn_like(x0).to(self.device)
        sqrt_alpha_cumprod_t = torch.sqrt(self.alpha_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - self.alpha_cumprod[t])[:, None, None, None]
        xt = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        return xt, noise

    def reverse_diffusion(self, xt, t):
        """反向扩散过程：移除噪声"""
        pred_noise = self.model(xt, t)
        beta_t = self.beta[t][:, None, None, None]
        alpha_t = self.alpha[t][:, None, None, None]
        alpha_cumprod_t = self.alpha_cumprod[t][:, None, None, None]
        alpha_cumprod_prev_t = self.alpha_cumprod_prev[t][:, None, None, None]

        mean = 1 / torch.sqrt(alpha_t) * (xt - beta_t / torch.sqrt(1 - alpha_cumprod_t) * pred_noise)
        variance = torch.sqrt(1 - alpha_cumprod_prev_t)

        z = torch.randn_like(xt).to(self.device) if t > 0 else 0  # 最后一层不需要噪声
        xt_prev = mean + variance * z
        return xt_prev

    def generate(self, batch_size):
        """从噪声生成清晰图像"""
        xt = torch.randn(batch_size, 3, *self.img_size).to(self.device)
        for t in reversed(range(self.timesteps)):
            xt = self.reverse_diffusion(xt, torch.full((batch_size,), t, dtype=torch.long).to(self.device))
        return xt


# --------------------------------------------
# 3. 数据集定义
# --------------------------------------------
class UnderwaterDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = ...  # 读取水下图像
        clear_img = ...  # 对应的清晰图像
        if self.transform:
            img = self.transform(img)
            clear_img = self.transform(clear_img)
        return img, clear_img


# --------------------------------------------
# 4. 训练过程
# --------------------------------------------
def train_model(model, diffusion, dataloader, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            noisy_imgs, clean_imgs = batch
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)

            t = torch.randint(0, diffusion.timesteps, (noisy_imgs.size(0),)).to(device)
            xt, noise = diffusion.forward_diffusion(clean_imgs, t)

            optimizer.zero_grad()
            pred_noise = model(xt, t)
            loss = F.mse_loss(pred_noise, noise)  # 损失函数
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")


# --------------------------------------------
# 5. 使用示例
# --------------------------------------------
if __name__ == "__main__":
    # 模型参数
    img_size = (128, 128)  # 输入图像大小
    timesteps = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
    transform = T.Compose([T.Resize(img_size), T.ToTensor()])
    dataset = UnderwaterDataset(img_paths=["path/to/images"], transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 初始化模型和扩散过程
    model = UNet(in_channels=3, out_channels=3).to(device)
    diffusion = DiffusionModel(model, img_size, timesteps, device)

    # 训练模型
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_model(model, diffusion, dataloader, optimizer, epochs=10, device=device)

    # 推理生成
    generated_imgs = diffusion.generate(batch_size=16)
    save_image(generated_imgs, "generated_images.png")
