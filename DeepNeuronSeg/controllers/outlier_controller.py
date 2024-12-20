from PyQt5.QtCore import QObject
import os
class OutlierController(QObject):
    def __init__(self, model, view):
        super().__init__()
        self.model = model
        self.view = view
        self._blinded = False

        self.view.update_outlier_threshold_signal.connect(self.model.update_outlier_threshold)
        self.view.update_signal.connect(self.update)
        self.view.next_image_signal.connect(self.next_image)
        self.view.remove_outlier_signal.connect(self.remove_outlier)

    def set_blinded(self, value):
        self._blinded = value
        self.update()
    
    def receive_outlier_data(self, data):
        outlier_dict = self.model.receive_outlier_data(data)
        self.view.update_outliers(outlier_dict)
    
    def update(self):
        self.view.update_response()

    def next_image(self):
        self.model.image_manager.next_image()
        item, index, total, points = self.model.image_manager.get_item()
        self.view.display_outlier_image(item, index, total, points)

    def remove_outlier(self):
        index = self.model.image_manager.get_index()
        self.view.remove_outlier(index)
