from PyQt5.QtCore import QObject


class AnalysisController(QObject):
    def __init__(self, model, view):
        super().__init__()
        self.model = model
        self.view = view

        self.view.inference_images_signal.connect(self.inference_images)
        self.view.save_inferences_signal.connect(self.save_inferences)
        self.view.download_data_signal.connect(self.download_data)
        self.model.dataset_metrics_signal.connect(self.update_dataset_metrics)
        self.model.analysis_metrics_signal.connect(self.update_analysis_metrics)
        self.view.update_signal.connect(self.update_model_selector)
        self.view.display_graph_signal.connect(self.display_graph)

        self.update_model_selector()

    def display_graph(self, checked):
        if checked:
            sorted_all_num_detections, sorted_all_conf_mean, colors = self.model.display_graph()
            if sorted_all_num_detections is not None and sorted_all_conf_mean is not None and colors is not None:
                self.view.update_graph(sorted_all_num_detections, sorted_all_conf_mean, colors)
            else:
                print("No metrics to display, please calculate metrics first.")
        else:
            self.view.clear_graph()


    def inference_images(self, model_name, uploaded_files):
        self.model.inference_images(model_name, uploaded_files)
        print("controller recieved signal",)

    def save_inferences(self):
        self.model.save_inferences()

    def download_data(self):
        self.model.download_data()

    def update_model_selector(self):
        models = map(lambda model: model['model_name'], self.model.load_models())
        self.view.update_response(models)

    def receive_dataset_metrics(self, dataset_metrics, variance_baselines, model_path):
            self.model.receive_dataset_metrics(dataset_metrics, variance_baselines, model_path)

    def update_dataset_metrics(self, dataset_metrics):
        self.view.update_dataset_metrics(dataset_metrics)

    def update_analysis_metrics(self, analysis_metrics):
        self.view.update_analysis_metrics(analysis_metrics)