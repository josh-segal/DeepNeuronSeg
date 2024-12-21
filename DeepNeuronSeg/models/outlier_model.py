from PyQt5.QtCore import QObject, pyqtSignal
from DeepNeuronSeg.models.image_manager import ImageManager
import tempfile
import shutil
import os
from tinydb import Query

class OutlierModel(QObject):

    update_signal = pyqtSignal()

    def __init__(self, db):
        super().__init__()
        self.db = db
        self.outlier_threshold = 3
        self.dataset_path = tempfile.mkdtemp()
        self.image_manager = ImageManager(self.db, dataset_path=self.dataset_path)

    def update_outlier_threshold(self, value):
        self.outlier_threshold = value

    def receive_outlier_data(self, data, inference_dir, blinded=False):
        self.inference_dir = inference_dir
        outlier_dict = {}
        for item in data:
            for file, score in item.items():
                if score > self.outlier_threshold:
                    outlier_dict[file] = score
                    dst_path = os.path.join(self.dataset_path, os.path.basename(file[0]))
                    shutil.copy(file[0], dst_path)
        return outlier_dict, blinded
    
    def remove_outlier(self):
        new_dataset_path = tempfile.mkdtemp()
        item, _, _, _ = self.image_manager.get_item()
        file_to_remove = os.path.basename(item[0])
        for file in os.listdir(self.dataset_path):
            if file_to_remove != file:
                shutil.copy(os.path.join(self.dataset_path, file), os.path.join(new_dataset_path, file))
        self.dataset_path = new_dataset_path
        self.image_manager.set_dataset_path(self.dataset_path)
        self.update_signal.emit()

    def relabel_outlier(self):
        item, _, _, _ = self.image_manager.get_item()
        image_path = item[0]
        image_name = os.path.basename(image_path)
        
        image_data = Query()
        if not self.db.image_table.get(image_data.file_path == image_name):
            
            dst_path = os.path.join('data', 'data_images', image_name)
            shutil.copy(image_path, dst_path)

            self.db.image_table.insert({
                "file_path": dst_path,
                "labels": []
            })
        self.remove_outlier()

    def get_inference_result(self, path):
        if not os.path.exists(self.inference_dir):
            return None
            
        image_name = os.path.basename(path)
        inference_path = os.path.join(self.inference_dir, image_name)
        
        if not os.path.exists(inference_path):
            return None
            
        return inference_path

    def update(self):
        pass