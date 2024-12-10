

class DatasetController:
    def __init__(self, model, view):
        self.model = model
        self.view = view

        self.view.update_signal.connect(self.update)
        self.view.set_augmentations_signal.connect(self.set_augmentations)

        self.update_dataset_selector()

    def set_augmentations(self, checked):
        augmentations = self.model.set_augmentations(checked)
        self.view.update_augmentations(augmentations)

    def update_dataset_selector(self):
        datasets = self.model.update_dataset_selector()
        self.view.update_dataset_selector(datasets)

    def update(self):
        datasets = self.model.update_dataset_selector()
        self.update_response(datasets)