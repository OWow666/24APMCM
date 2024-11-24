import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# UNet 模型定义
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64):
        super(UNet, self).__init__()

        self.enc1 = self.conv_block(in_channels, base_channels)  # 3 -> 64
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

    def forward(self, x):
        enc1 = self.enc1(x)  # 64
        enc2 = self.enc2(self.pool(enc1))  # 128
        enc3 = self.enc3(self.pool(enc2))  # 256

        bottleneck = self.bottleneck(self.pool(enc3))  # 512

        dec2 = self.up2(bottleneck) + enc3  # 256
        dec1 = self.up1(dec2) + enc2  # 128
        dec0 = self.up0(dec1) + enc1  # 64

        out = self.final_conv(dec0)
        return out

class UnderWaterDataset(Dataset):
    def __init__(self, dataset_paths, transform=None):
        self.transform = transform
        self.img_pairs = []
        for path in dataset_paths:
            trainA_path = os.path.join(path, 'trainA')
            trainB_path = os.path.join(path, 'trainB')
            trainA_imgs = os.listdir(trainA_path)
            trainB_imgs = os.listdir(trainB_path)
            trainA_imgs.sort()
            trainB_imgs.sort()

            for img_name in trainA_imgs:
                self.img_pairs.append((os.path.join(trainA_path, img_name), os.path.join(trainB_path, img_name)))

    def __len__(self):
        return len(self.img_pairs)

    def __getitem__(self, idx):
        origin_img_path, res_img_path = self.img_pairs[idx]
        origin_img = Image.open(origin_img_path).convert('RGB')
        res_img = Image.open(res_img_path).convert('RGB')
        if self.transform:
            origin_img = self.transform(origin_img)
            res_img = self.transform(res_img)
        return origin_img, res_img

if __name__ == "__main__":
    dataset_paths = [
        os.path.join(os.getcwd(), 'EUVP', 'Paired', 'underwater_dark'),
        os.path.join(os.getcwd(), 'EUVP', 'Paired', 'underwater_imagenet'),
        os.path.join(os.getcwd(), 'EUVP', 'Paired', 'underwater_scenes')
    ]
    transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()])
    train_dataset = UnderWaterDataset(dataset_paths, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    print("Data has been Prepared!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(3, 3, 32).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 10
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}')

    torch.save(model.state_dict(), "./UNet_model.pth")
    print("Model has been Saved!")