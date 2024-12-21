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
        self.view.remove_outlier_signal.connect(self.model.remove_outlier)
        self.view.load_image_signal.connect(self.load_image)
        self.model.update_signal.connect(self.update)
        self.view.relabel_outlier_signal.connect(self.model.relabel_outlier)

    def set_blinded(self, value):
        self._blinded = value
        self.update()
    
    def receive_outlier_data(self, data, inference_dir):
        outlier_dict, blinded = self.model.receive_outlier_data(data, inference_dir, self._blinded)
        self.view.update_outliers(outlier_dict, blinded)
    
    def update(self):
        images = self.model.image_manager.get_images()
        self.view.update_response(images, self._blinded)
        self.curr_image()

    def load_image(self, index):
        self.model.image_manager.set_index(index)
        self.curr_image()

    def next_image(self):
        self.model.image_manager.next_image()
        self.curr_image()

    def curr_image(self):
        item, index, total, points = self.model.image_manager.get_item()
        if item:
            inference_result = self.model.get_inference_result(item[0])
            if inference_result is None:
                self.view.image_display.clear()
                self.view.image_display.text_label.setText("No inference result found")
                return
            self.view.image_display.display_frame((inference_result, 0), index, total, points)
        else:
            self.view.image_display.clear()
            self.view.image_display.text_label.setText("No image found")
