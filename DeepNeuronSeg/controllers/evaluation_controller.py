from PyQt5.QtCore import QObject

class EvaluationController(QObject):
    def __init__(self, model, view):
        super().__init__()
        self.model = model
        self.view = view
        self._blinded = False

        self.view.calculate_metrics_signal.connect(self.calculate_metrics)
        self.view.display_graph_signal.connect(self.display_graph)
        self.view.download_data_signal.connect(self.download_data)
        self.view.update_signal.connect(self.update)
        self.view.next_image_signal.connect(self.next_image)
        self.view.dataset_changed_signal.connect(self.on_dataset_changed)
        self.view.load_image_signal.connect(self.load_image)
        self.view.update_confidence_signal.connect(self.update_confidence)

        self.update()

    def set_blinded(self, value):
        self._blinded = value
        self.update()

    def update_confidence(self, value):
        self.model.set_confidence(value)

    def on_dataset_changed(self, dataset_name):
        dataset_path = self.model.get_dataset_path(dataset_name)
        self.model.image_manager.set_dataset_path(dataset_path)
        item, index, total, points = self.model.image_manager.get_item(subdir='images')
        if item:
            self.view.image_display.display_frame(item, index, total, points)
        else:
            self.view.image_display.clear()
            self.view.image_display.text_label.setText("No image found")

    def next_image(self):
        self.model.image_manager.next_image(subdir='images')
        item, index, total, points = self.model.image_manager.get_item(subdir='images')
        if item:
            self.view.image_display.display_frame(item, index, total, points)
        else:
            self.view.image_display.clear()
            self.view.image_display.text_label.setText("No image found")

    def calculate_metrics(self, model_name, dataset_name):
        dataset_metrics = self.model.calculate_metrics(model_name, dataset_name)
        self.view.update_metrics_labels(dataset_metrics)
        checked = self.view.display_graph_checkbox.isChecked()
        self.display_graph(checked)

    def display_graph(self, checked):
        if checked:
            sorted_num_dets, sorted_conf_mean = self.model.display_graph()
            self.view.handle_graph_display(sorted_num_dets, sorted_conf_mean)
        else:
            self.view.handle_image_display()

    def download_data(self, dataset_name):
        self.model.download_data(dataset_name)

    def load_image(self, index):
        self.model.image_manager.set_index(index)
        item, index, total, points = self.model.image_manager.get_item(subdir='images')
        if item:
            self.view.image_display.display_frame(item, index, total, points)
        else:
            self.view.image_display.clear()
            self.view.image_display.text_label.setText("No image found")

    def update(self):
        self.model.set_first_dataset_path()
        images = self.model.image_manager.get_images(subdir='images')
        if self._blinded:
            images = [img[1] for img in images]
        else:
            images = [img[0] for img in images]
        models = self.model.get_models()
        datasets = self.model.get_datasets()
        self.view.update_response(models, datasets, images)
        if self.model.image_manager.dataset_path is not None:
            item, index, total, points = self.model.image_manager.get_item(subdir='images')
            if item:
                self.view.image_display.display_frame(item, index, total, points)
            else:
                self.view.image_display.clear()
                self.view.image_display.text_label.setText("No image found")
        else:
            self.view.image_display.clear()
            self.view.image_display.text_label.setText("No dataset found")
