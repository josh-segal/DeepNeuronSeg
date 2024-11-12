import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(image_path).convert('L')
        mask_dir = data_dir.replace('COCO_train_X', 'COCO_train_y')
        mask_path = os.path.join(mask_dir, self.image_files[idx])
        mask = Image.open(mask_path).convert('L')
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(1, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        
        self.up1 = DoubleConv(512 + 256, 256)
        self.up2 = DoubleConv(256 + 128, 128)
        self.up3 = DoubleConv(128 + 64, 64)
        self.up4 = DoubleConv(64, 64)
        
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # layers = []
        # layers.append(('original', x))

        c1 = self.down1(x)
        # layers.append(('down1', c1))
        x = self.pool(c1)
        
        c2 = self.down2(x)
        # layers.append(('down2', c2))
        x = self.pool(c2)
        
        c3 = self.down3(x)
        # layers.append(('down3', c3))
        x = self.pool(c3)
        
        c4 = self.down4(x)
        # layers.append(('down4', c4))
        
        x = self.upsample(c4)
        x = torch.cat([x, c3], dim=1)
        x = self.up1(x)
        # layers.append(('up1', x))
        
        x = self.upsample(x)
        x = torch.cat([x, c2], dim=1)
        x = self.up2(x)
        # layers.append(('up2', x))
        
        x = self.upsample(x)
        x = torch.cat([x, c1], dim=1)
        x = self.up3(x)
        # layers.append(('up3', x))
        
        x = self.up4(x)
        # layers.append(('up4', x))
        
        output = self.final(x)
        # layers.append(('final', output))
        
        # return torch.sigmoid(output), layers
        return torch.sigmoid(output)

    def trainer(num_epochs=3):
        # Train the model
        model = UNet()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            model.train()
            c = 1
            for images, masks in dataloader:
                # Forward pass
                # outputs, layers = model(images)
                outputs = model(images)
                loss = criterion(outputs, masks)
                print("finished iter ", c)
                c += 1

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # plot_images_in_grid(layers)


            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

            # Save the model
            torch.save(model.state_dict(), 'models/denoise_model.pth')