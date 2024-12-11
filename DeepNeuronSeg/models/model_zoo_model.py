import os
import tempfile
from PIL import Image
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel, QFileDialog
from tinydb import Query
from DeepNeuronSeg.views.widgets.image_display import ImageDisplay

class ModelZooModel:
    def __init__(self, db):
        super().__init__()
        self.db = db
        
        self.current_index = 0
        # Model selection
        
    def select_images(self):
        pass

    def inference_images(self):

        from ultralytics import YOLO

        self.model_name = self.model_selector.currentText()
        self.model_path = self.db.model_table.get(Query().model_name == self.model_name).get('model_path')
        self.model = YOLO(self.model_path)
        self.inference_dir = os.path.join('data', 'data_inference')
        os.makedirs(self.inference_dir, exist_ok=True)
        if not self.uploaded_files:
            print("No images selected")
            return  
        #TODO: if denoise trained model pass through denoise model first
        self.inference_result = self.model.predict(self.uploaded_files, conf=0.3, visualize=False, save=False, show_labels=False, max_det=1000, verbose=False)

        self.display_images()

    def display_images(self):
        if len(self.inference_result) > 0 and len(self.uploaded_files) > 0:
            
            self.left_image._display_image(self.uploaded_files[self.current_index], self.current_index + 1, len(self.uploaded_files))
            self._show_pred()

            self.current_index = (self.current_index + 1) % len(self.uploaded_files)
            

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

    def _show_pred(self):
        result = self.inference_result[self.current_index]
        mask_image = result.plot(labels=False, conf=False, boxes=False)
        mask_image = Image.fromarray(mask_image)
        temp_image_path = os.path.join(tempfile.gettempdir(), "temp_mask_image.png")
        mask_image.save(temp_image_path, format='PNG')
        self.right_image._display_image(temp_image_path, self.current_index + 1, len(self.inference_result))

    def update(self):
        pass
