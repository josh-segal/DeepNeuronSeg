import os
import tempfile
from PIL import Image
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel, QFileDialog
from PyQt5.QtCore import pyqtSignal, QObject
from tinydb import Query
from DeepNeuronSeg.views.widgets.image_display import ImageDisplay

class ModelZooModel(QObject):

    display_image_signal = pyqtSignal(list, list, int)

    def __init__(self, db):
        super().__init__()
        self.db = db
        
        self.current_index = 0
        # Model selection

    def inference_images(self, name_of_model, uploaded_files):
        print("model inferencing images")
        from ultralytics import YOLO
        self.uploaded_files = uploaded_files
        self.model_path = self.db.model_table.get(Query().model_name == name_of_model).get('model_path')
        self.model_denoise = self.db.model_table.get(Query().model_name == name_of_model).get('denoise')
        self.model = YOLO(self.model_path)
        self.inference_dir = os.path.join('data', 'inferences', name_of_model)
        os.makedirs(self.inference_dir, exist_ok=True)
        if not uploaded_files:
            print("No images selected")
            return  
        if self.model_denoise:
            #TODO: NEED TO TEST WITH A REAL MODEL
            dn_model = DenoiseModel(dataset_path='idc update to not need', model_path=self.model_denoise)
            uploaded_images = [
                dn_model.denoise_image(Image.open(image_path).convert('L')) 
                for image_path in uploaded_files
            ]
        else:
            uploaded_images = [Image.open(image_path) for image_path in uploaded_files]
            
        self.inference_result = self.model.predict(uploaded_images, conf=0.3, visualize=False, save=False, show_labels=False, max_det=1000, verbose=False)

        self.display_images()

    def display_images(self):
        self.display_image_signal.emit(self.uploaded_files, self.inference_result, self.current_index)
            

    def save_inferences(self):
        for file, result in zip(self.uploaded_files, self.inference_result):
                masks = result.masks
                mask_num = len(masks)
                # print(file, '------------')
                default_file_name = f"{os.path.splitext(os.path.basename(file))[0]}_{mask_num}.png"
                save_path, _ = QFileDialog.getSaveFileName(self, "Save Inference", default_file_name, "Images (*.png)")
                if not save_path:
                    continue
                mask_image = result.plot(labels=False, conf=False, boxes=False)
                mask_image = Image.fromarray(mask_image)
                # print(save_path)
                mask_image.save(save_path)

    def update_index(self, index):
        self.current_index = index

    def update(self):
        pass
