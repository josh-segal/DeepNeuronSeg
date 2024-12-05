import os
from torchvision import transforms
from torch.utils.data import DataLoader
from DeepNeuronSeg.views.widgets.image_display import ImageDataset
from DeepNeuronSeg.models.denoise_model import DenoiseModel

def load_dataset(self, dataset_path):
        not_temp = os.path.exists(os.path.join(dataset_path, 'images'))
        if not_temp:
            image_path = os.path.join(dataset_path, 'images')
            dataset = ImageDataset(root_dir=image_path)
        elif self.denoised:
            if os.path.exists(os.path.join(dataset_path, 'denoise_model.pth')):
                model_path = os.abspath(os.path.join(dataset_path, 'denoise_model.pth'))
                print(model_path)
                dn_model = DenoiseModel(dataset_path, model_path=model_path)
                
            else:
                dn_model = DenoiseModel(dataset_path)

            transform = transforms.Compose([
                    transforms.Lambda(lambda image: dn_model.denoise_image(image)),
                    transforms.Resize((512, 512)),
                    transforms.Lambda(lambda image: image.convert("RGB")),
                    transforms.ToTensor(),
                ])
            dataset = ImageDataset(root_dir=dataset_path, transform=transform)
        else:
            dataset = ImageDataset(root_dir=dataset_path)

        self.batch_size = 4
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader