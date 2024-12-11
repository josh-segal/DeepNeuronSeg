from itertools import chain
from tinydb import Query
import os
from DeepNeuronSeg.models.denoise_model import DenoiseModel
from pathlib import Path


class TrainingModel:
    def __init__(self, db):
        super().__init__()
        self.db = db

        self.default_augmentations = {
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'auto_augment': 'randaugment',
            'erasing': 0.4,
            'crop_fraction': 1.0
        }

        self.no_augmentations = {
            'hsv_h': 0.0,
            'hsv_s': 0.0,
            'hsv_v': 0.0,
            'degrees': 0.0,
            'translate': 0.0,
            'scale': 0.0,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.0,
            'mosaic': 0.0,
            'mixup': 0.0,
            'auto_augment': 'randaugment',
            'erasing': 0.0,
            'crop_fraction': 1.0
        }

        self.augmentations = self.default_augmentations.copy()

    def update_dataset_selector(self):
        return chain(
                *map(
                    lambda dataset: [dataset['dataset_name']] + 
                                    ([f"{dataset['dataset_name']} (denoised)"] if 'denoise_dataset_path' in dataset and dataset['denoise_dataset_path'] else []),
                    self.db.load_datasets()
                )
            )

    def set_augmentations(self, checked):
        if checked:
            self.augmentations = self.default_augmentations.copy()
            print(self.augmentations)
        else:
            self.augmentations = self.no_augmentations.copy()
            print(self.augmentations)
        return self.augmentations.copy()

    def trainer(self, model_name, base_model, dataset_name, denoise, denoise_base, epochs, batch_size):
        if not model_name:
            print("Model name required")
            return
        
        dataset = self.db.dataset_table.get(Query().dataset_name == dataset_name)
        dataset_path = dataset["dataset_path"]

        model_name_exists = self.db.model_table.contains(Query()["model_name"] == model_name)
        if model_name_exists:
            print("Model name already exists, please choose a different name")
            return

        if denoise:
            print("Training denoising network")
            denoise_path = os.path.join(dataset_path, "denoise_model.pth")
            os.makedirs(denoise_path, exist_ok=True)

            dn_model = DenoiseModel(dataset_path=dataset_path, model_path=denoise_path)
            dn_model.unet_trainer(num_epochs=epochs, batch_size=batch_size)
            dn_dataset_path = dn_model.create_dn_shuffle()

            dataset_data = Query()
            self.db.dataset_table.update(
                {"denoise_dataset_path": os.path.abspath(dn_dataset_path)}, 
                dataset_data.dataset_path == os.path.abspath(dataset_path)
            )

            print(f"Denoising images in {os.path.abspath(dataset_path)} and saving to {os.path.abspath(dn_dataset_path)}")

            dataset_path = os.path.abspath(dataset_path)

        elif denoise_base:
            print("Using pretrained denoising network")
            denoise_path = (Path(__file__).resolve().parents[2] / "ml" / "denoise_model.pth").resolve()
            print('denoise path: ', denoise_path)
            # denoise_path = os.path.abspath("ml/denoise_model.pth")
            dn_model = DenoiseModel(dataset_path=dataset_path)
            dn_dataset_path = dn_model.create_dn_shuffle()

            dataset_data = Query()
            self.db.dataset_table.update(
                {"denoise_dataset_path": os.path.abspath(dn_dataset_path)}, 
                dataset_data.dataset_path == os.path.abspath(dataset_path)
            )

            print(f"Denoising images in {os.path.abspath(dataset_path)} and saving to {os.path.abspath(dn_dataset_path)}")

            dataset_path = os.path.abspath(dataset_path)
        else:
            denoise_path = None

        if base_model == "YOLOv8n-seg":
            # offset program load times by loading model here
            from ultralytics import YOLO
            print("Training YOLOv8n-seg")

            

            self.model = YOLO((Path(__file__).resolve().parents[2] / "ml" / "yolov8n-seg.pt").resolve())
            # self.model = YOLO("ml/yolov8n-seg.pt")
            self.model.train(
                #TODO: if denoised use denoised data dir, recreate yaml (?)
                data = os.path.abspath(f'{dataset_path}/data.yaml'),
                project = f'{dataset_path}/results',
                name = model_name,
                epochs = epochs,
                patience = 0,
                batch = batch_size,
                imgsz = 1024,
                hsv_h=self.augmentations['hsv_h'], 
                hsv_s=self.augmentations['hsv_s'], 
                hsv_v=self.augmentations['hsv_v'], 
                degrees=self.augmentations['degrees'], 
                translate=self.augmentations['translate'], 
                scale=self.augmentations['scale'], 
                shear=self.augmentations['shear'], 
                perspective=self.augmentations['perspective'], 
                flipud=self.augmentations['flipud'], 
                fliplr=self.augmentations['fliplr'], 
                mosaic=self.augmentations['mosaic'], 
                mixup=self.augmentations['mixup'],
                auto_augment=self.augmentations['auto_augment'],
                erasing=self.augmentations['erasing'],
                crop_fraction=self.augmentations['crop_fraction']
            )

        self.db.model_table.insert({
            "model_name": model_name,
            "model_path": f'{dataset_path}/results/{model_name}/weights/best.pt',
            "denoise_path": str(denoise_path)
        })

    def update(self):
        self.dataset.clear()
        self.dataset.addItems(
            chain(
                *map(
                    lambda dataset: [dataset['dataset_name']] + 
                                    ([f"{dataset['dataset_name']} (denoised)"] if 'denoise_dataset_path' in dataset and dataset['denoise_dataset_path'] else []),
                    self.db.load_datasets()
                )
            )
        )
