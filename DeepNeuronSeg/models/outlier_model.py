from PyQt5.QtCore import QObject
from DeepNeuronSeg.models.image_manager import ImageManager
import tempfile
import shutil
import os

class OutlierModel(QObject):
    def __init__(self, db):
        super().__init__()
        self.db = db
        self.outlier_threshold = 3
        self.dataset_path = tempfile.mkdtemp()
        self.image_manager = ImageManager(dataset_path=self.dataset_path)

    def update_outlier_threshold(self, value):
        self.outlier_threshold = value

    def receive_outlier_data(self, data):
        outlier_dict = {}
        for item in data:
            for file, score in item.items():
                if score > self.outlier_threshold:
                    outlier_dict[file] = score
                    dst_path = os.path.join(self.dataset_path, os.path.basename(file))
                    shutil.copy(file, dst_path)
        return outlier_dict

    def update(self):
        pass