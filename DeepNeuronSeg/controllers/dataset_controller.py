from PyQt5.QtCore import QObject

class DatasetController(QObject):
    def __init__(self, model, view):
        super().__init__()
        self.model = model
        self.view = view
        self._blinded = False

        self.view.update_signal.connect(self.update)
        self.view.create_dataset_signal.connect(self.create_dataset)

        self.update_model_selector()

    def set_blinded(self, value):
        self._blinded = value
        self.update()

    def update(self):
        items = self.model.load_images()
        self.view.update_response(items, self._blinded)

    def create_dataset(self, selected_images, dataset_name, train_split):
        self.model.create_dataset(selected_images, dataset_name, train_split)

    def update_model_selector(self):
        items = self.model.db.load_images()
        self.view.update_image_list(items)