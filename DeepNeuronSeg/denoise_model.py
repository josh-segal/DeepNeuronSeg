import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm
from utils import create_yaml, copy_files
import shutil
from tinydb import Query

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


class DenoiseModel:
    def __init__(self, dataset_path, model_path=None):
        self.model = None
        self.dataset_path = dataset_path
        self.images_path = os.path.join(dataset_path, 'images') if os.path.exists(os.path.join(dataset_path, 'images')) else dataset_path 
        self.model_path = model_path

    def unet_trainer(self, num_epochs=3, batch_size=4):
        # build the dataset
        denoise_path = os.path.join(self.dataset_path, "denoise_model.pth")
        os.makedirs(denoise_path, exist_ok=True)
        self.model_path = os.path.abspath(denoise_path)
        model = UNet()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # define transforms
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

        # create dataset

        dataset = ImageDataset(self.images_path, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # define loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # training loop
        for epoch in range(num_epochs):
            model.train()

            with tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
                for images, masks in pbar:
                    images = images.to(device)
                    masks = masks.to(device)

                    # Forward pass
                    # outputs, layers = model(images)
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.set_postfix({'loss': loss.item()})

            # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Save the model
        torch.save(model.state_dict(), self.model_path)

    def load_model(self):
        if self.model is None:
            self.model = UNet()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.load_state_dict(torch.load(self.model_path, map_location=device))
            self.model.eval()

        return self.model

    def create_dn_shuffle(self):
        dn_dir = os.path.join(self.dataset_path, 'denoised')
        dn_images_path = os.path.join(dn_dir, 'images')
        os.makedirs(dn_images_path, exist_ok=True)

        for image in os.listdir(self.images_path):
            image_path = os.path.join(self.images_path, image)
            save_path = os.path.join(dn_images_path, image)
            print(image_path)
            image = Image.open(image_path).convert('L')
            image = self.denoise_image(image)
            image.save(save_path)
            print(f"Saved denoised image to: {save_path}")

        train_images_dir = os.path.join("train", "images")
        val_images_dir = os.path.join("val", "images")

        copy_files(os.path.join(self.dataset_path, "train", "images"), os.path.join(self.dataset_path, "denoised", "images"), os.path.join(self.dataset_path, "denoised", "train", "images"))
        copy_files(os.path.join(self.dataset_path, "val", "images"), os.path.join(self.dataset_path, "denoised", "images"), os.path.join(self.dataset_path, "denoised", "val", "images"))

        
        shutil.copytree(os.path.join(self.dataset_path, "train", "labels"), os.path.join(self.dataset_path, "denoised", "train", "labels"), dirs_exist_ok=True)
        shutil.copytree(os.path.join(self.dataset_path, "train", "masks"), os.path.join(self.dataset_path, "denoised", "train", "masks"), dirs_exist_ok=True)

        shutil.copytree(os.path.join(self.dataset_path, "val", "labels"), os.path.join(self.dataset_path, "denoised", "val", "labels"), dirs_exist_ok=True)
        shutil.copytree(os.path.join(self.dataset_path, "val", "masks"), os.path.join(self.dataset_path, "denoised", "val", "masks"), dirs_exist_ok=True)

        create_yaml(os.path.join(dn_dir, "data.yaml"), train_images_dir, val_images_dir)

        return os.path.abspath(dn_dir)

    def denoise_image(self, image):
        model = self.load_model()
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            denoised_image = model(image)
        denoised_image = transforms.ToPILImage()(denoised_image.squeeze(0)) 

        return denoised_image
