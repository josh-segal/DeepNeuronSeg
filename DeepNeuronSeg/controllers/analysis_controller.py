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
        self.view.update_signal.connect(self.update)
        self.view.display_graph_signal.connect(self.display_graph)
        self.view.curr_image_signal.connect(self.curr_image)
        self.view.load_image_signal.connect(self.load_image)
        self.view.next_image_signal.connect(self.next_image)
        self.model.update_images_signal.connect(self.update)

        self.update()

    def next_image(self):
        self.model.image_manager.next_image()
        self.curr_image()

    def load_image(self, index):
        self.model.image_manager.set_index(index)
        self.curr_image()

    def curr_image(self):
        item, index, total, points = self.model.image_manager.get_item()
        if item:
            self.view.image_display.display_frame(item, index, total, points)

    def display_graph(self, checked):
        if checked:
            sorted_all_num_detections, sorted_all_conf_mean, colors = self.model.display_graph()
            self.view.handle_graph_display(sorted_all_num_detections, sorted_all_conf_mean, colors)
        else:
            self.view.handle_image_display()

    def inference_images(self, model_name, uploaded_files):
        self.model.inference_images(model_name, uploaded_files)

    def save_inferences(self):
        self.model.save_inferences()

    def download_data(self):
        self.model.download_data()

    def update(self):
        models = map(lambda model: model['model_name'], self.model.load_models())
        images = self.model.image_manager.get_images()
        self.view.update_response(models, images)
        self.curr_image()

    def receive_dataset_metrics(self, dataset_metrics, analysis_metrics, variance_baselines, model_path):
            self.model.receive_dataset_metrics(dataset_metrics, analysis_metrics, variance_baselines, model_path)

    def update_dataset_metrics(self, dataset_metrics):
        self.view.update_dataset_metrics(dataset_metrics)

    def update_analysis_metrics(self, analysis_metrics):
        self.view.update_analysis_metrics(analysis_metrics)