import os
from PIL import Image
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import pyqtSignal, QObject
from tinydb import Query
from ultralytics import YOLO
from DeepNeuronSeg.models.denoise_model import DenoiseModel
import pandas as pd
from PyQt5.QtWidgets import QMessageBox

class ModelZooModel(QObject):

    display_image_signal = pyqtSignal(str, object, int, int)

    def __init__(self, db):
        super().__init__()
        self.db = db
        
        self.current_index = 0
        # Model selection

    def download_data(self):
        if not hasattr(self, 'inference_result') or not hasattr(self, 'uploaded_files'):
            # QMessageBox.warning(self, "No Inference Results", "No inference results to download")
            return
        # Create data for each image
        data = []
        for file, result in zip(self.uploaded_files, self.inference_result):
            basename = os.path.basename(file)
            num_dets = len(result.boxes)
            conf_mean = float(result.boxes.conf.mean()) if num_dets > 0 else 0
            
            data.append({
                'image': basename,
                'num_detections': num_dets,
                'confidence_mean': conf_mean
            })

        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        save_path, _ = QFileDialog.getSaveFileName(None, "Save Metrics", 
                                                 "inference_metrics.csv",
                                                 "CSV Files (*.csv)")
        if save_path:
            df.to_csv(save_path, index=False)
            # QMessageBox.information(self, "Metrics Saved", f"Metrics saved to {save_path}")

    def inference_images(self, name_of_model, uploaded_files, confidence):
        self.uploaded_files = uploaded_files
        model_path = self.db.model_table.get(Query().model_name == name_of_model).get('model_path')
        model_denoise = self.db.model_table.get(Query().model_name == name_of_model).get('denoise')
        model = YOLO(model_path)
        self.inference_dir = os.path.join('data', 'inferences', name_of_model)
        os.makedirs(self.inference_dir, exist_ok=True)
        if not uploaded_files:
            # QMessageBox.warning(self, "No Images", "No images selected")
            return  
        if model_denoise:
            dn_model = DenoiseModel(dataset_path='idc update to not need', model_path=model_denoise)
            uploaded_images = [
                dn_model.denoise_image(Image.open(image_path).convert('L')) 
                for image_path in uploaded_files
            ]
        else:
            uploaded_images = [Image.open(image_path) for image_path in uploaded_files]
            
        self.inference_result = model.predict(uploaded_images, conf=confidence, visualize=False, save=False, show_labels=False, max_det=1000, verbose=False)

        self.display_images()

    def display_images(self):
        self.display_image_signal.emit(self.uploaded_files[self.current_index], self.inference_result[self.current_index], self.current_index, len(self.uploaded_files))

    def next_image(self):
        self.current_index = (self.current_index + 1) % len(self.uploaded_files)
        self.display_images()
            

    def save_inferences(self):
        for file, result in zip(self.uploaded_files, self.inference_result):
                masks = result.masks
                mask_num = len(masks)
                default_file_name = f"{os.path.splitext(os.path.basename(file))[0]}_{mask_num}.png"
                save_path, _ = QFileDialog.getSaveFileName(None, "Save Inference", default_file_name, "Images (*.png)")
                if not save_path:
                    continue
                mask_image = result.plot(labels=False, conf=False, boxes=False)
                mask_image = Image.fromarray(mask_image)
                mask_image.save(save_path)

    def update_index(self, index):
        self.current_index = index

    def update(self):
        pass
