from PyQt5.QtCore import QObject

class EvaluationController(QObject):
    def __init__(self, model, view):
        super().__init__()
        self.model = model
        self.view = view

        self.view.calculate_metrics_signal.connect(self.calculate_metrics)
        self.view.display_graph_signal.connect(self.display_graph)
        self.view.download_data_signal.connect(self.download_data)
        self.view.update_signal.connect(self.update)

        self.update()

    def calculate_metrics(self, model_name, dataset_name):
        dataset_metrics = self.model.calculate_metrics(model_name, dataset_name)
        self.view.update_metrics_labels(dataset_metrics)

    def display_graph(self, checked):
        if checked:
            sorted_num_dets, sorted_conf_mean = self.model.display_graph()
            self.view.handle_graph_display(sorted_num_dets, sorted_conf_mean)
        else:
            self.view.handle_image_display()

    def download_data(self, dataset_name):
        self.model.download_data(dataset_name)

    def update(self):
        models = self.model.get_models()
        datasets = self.model.get_datasets()
        self.view.update_response(models, datasets)