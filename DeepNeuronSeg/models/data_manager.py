import os
from tinydb import TinyDB, Query
from pathlib import Path
from PyQt5.QtWidgets import QMessageBox

class DataManager:
    def __init__(self, db_path='data/db.json'):
        self.db = TinyDB(db_path)
        self.image_table = self.db.table('images')
        self.dataset_table = self.db.table('datasets')
        self.model_table = self.db.table('models')

        if not self.model_table.search(Query()["model_name"] == "DeepNeuronSegBaseModel"):
            self.model_table.insert({
                "model_name": "DeepNeuronSegBaseModel",
                "model_path": str((Path(__file__).resolve().parents[1] / "ml" / "yolov8n-largedata-70-best.pt").resolve()),
                # os.path.abspath("ml/yolov8n-largedata-70-best.pt"),
                "denoise": ""
                # str((Path(__file__).resolve().parents[1] / "ml" / "denoise_model.pth").resolve()) 
                # os.path.abspath("ml/denoise_model.pth")
            })

    def load_images(self):
        images = self.image_table.all()
        uploaded_files = [(image['file_path'], str(i)) for i, image in enumerate(images) if 'file_path' in image]
        if not uploaded_files:
            # QMessageBox.warning(self, "No Images", "No images found")
            print('No images found')
            return []
        else:
            return uploaded_files

    def load_labels(self):
        images = self.image_table.all()

        labels = [image['labels'] for image in images if 'labels' in image]
        if not labels:
            # QMessageBox.warning(self, "No Labels", "No labels found")
            print('No labels found')
            return []
        else:
            return labels

    def load_masks(self):
        items = self.image_table.all()

        masks = [(item['mask_data']['mask_path'], str(i)) for i, item in enumerate(items) if 'mask_data' in item]
        if not masks:
            # QMessageBox.warning(self, "No Masks", "No masks found")
            print('No masks found')
            return []
        else:
            return masks

    def load_datasets(self):
        return self.dataset_table.all()

    def load_models(self):
        return self.model_table.all()
    
    def get_models(self):
        return map(lambda model: model['model_name'], self.load_models())
