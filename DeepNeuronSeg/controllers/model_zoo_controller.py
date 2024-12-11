from PyQt5.QtCore import QObject

class ModelZooController(QObject):
    def __init__(self, model, view):
        super().__init__()
        self.model = model
        self.view = view

        self.view.inference_images_signal.connect(self.model.inference_images)
        self.view.save_inferences_signal.connect(self.model.save_inferences)
        self.model.display_image_signal.connect(self.view.display_images)
        self.view.update_index_signal.connect(self.model.update_index)