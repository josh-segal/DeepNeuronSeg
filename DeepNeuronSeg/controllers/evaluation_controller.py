

class EvaluationController:
    def __init__(self, model, view):
        self.model = model
        self.view = view

        self.view.calculate_metrics_signal.connect(self.calculate_metrics)
        self.view.display_graph_signal.connect(self.display_graph)
        self.view.download_data_signal.connect(self.download_data)
        self.view.update_signal.connect(self.update)

        self.update()

    def calculate_metrics(self, model_name, dataset_name, display_graph):
        metrics = self.model.calculate_metrics(model_name, dataset_name, display_graph)
        self.view.update_metrics_labels(metrics)

    def display_graph(self, checked):
        sorted_num_dets, sorted_conf_mean = self.model.display_graph(checked)
        if sorted_num_dets is not None and sorted_conf_mean is not None:
            self.view.update_graph(sorted_num_dets, sorted_conf_mean)

    def download_data(self):
        self.model.download_data()

    def update(self):
        models = self.model.get_models()
        datasets = self.model.get_datasets()
        self.view.update_response(models, datasets)