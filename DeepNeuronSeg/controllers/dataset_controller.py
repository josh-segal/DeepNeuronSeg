

class DatasetController:
    def __init__(self, model, view):
        self.model = model
        self.view = view

        self.view.update_signal.connect(self.update)
        self.view.create_dataset_signal.connect(self.create_dataset)

        self.update_model_selector()

    def update(self):
        items = self.model.load_images()
        self.view.update_response(items)

    def create_dataset(self, images, dataset_name, train_split):
        self.model.create_dataset(images, dataset_name, train_split)

    def update_model_selector(self):
        items = self.model.db.load_images()
        self.view.update_image_list(items)