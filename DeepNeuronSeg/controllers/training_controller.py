from PyQt5.QtCore import QObject

class TrainingController(QObject):
    def __init__(self, model, view):
        super().__init__()
        self.model = model
        self.view = view
        self._blinded = False

        self.view.update_signal.connect(self.update)
        self.view.set_augmentations_signal.connect(self.set_augmentations)
        self.view.train_signal.connect(self.trainer)

        self.update_dataset_selector()

    def set_blinded(self, value):
        self._blinded = value
        self.update()
    
    def set_augmentations(self, checked):
        augmentations = self.model.set_augmentations(checked)
        self.view.update_augmentations(augmentations)

    def update_dataset_selector(self):
        datasets = self.model.update_dataset_selector()
        self.view.update_dataset_selector(datasets)

    def update(self):
        datasets = self.model.update_dataset_selector()
        self.view.update_response(datasets)

    def trainer(self, model_name, base_model, dataset_name, denoise, denoise_base, epochs, batch_size):
        self.model.trainer(model_name, base_model, dataset_name, denoise, denoise_base, epochs, batch_size)