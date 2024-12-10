from itertools import chain
from tinydb import Query
from DeepNeuronSeg.models.qa_metrics import DetectionQAMetrics

class EvaluationModel:

    calculated_dataset_metrics_signal = pyqtSignal(dict)
    update_metrics_labels_signal = pyqtSignal(dict)

    def __init__(self, db):
        super().__init__()
        self.db = db
        self.metrics = None

    def calculate_metrics(self, model_name, dataset_name):
        # TODO: abstract
        self.model_path = self.db.model_table.get(Query().model_name == model_name)
        self.model_path = self.model_path["model_path"]
        # print(self.model_path, '<----------------')
        if " (denoised)" in dataset_name:
            dn_dataset_name = dataset_name.replace(" (denoised)", "")
            self.dataset_path = self.db.dataset_table.get(Query().dataset_name == dn_dataset_name).get('denoise_dataset_path')
        else:
            self.dataset_path = self.db.dataset_table.get(Query().dataset_name == dataset_name).get('dataset_path')
        # print(self.dataset_path, '<----------------')

        self.metrics = DetectionQAMetrics(self.model_path, self.dataset_path)
        self.update_metrics_labels_signal.emit(self.metrics.dataset_metrics_mean_std)
    
    def display_graph(self, checked):
        if checked:
            if self.metrics is not None:
                sorted_num_dets, sorted_conf_mean = self.sort_metrics()
                return sorted_num_dets, sorted_conf_mean
            else:
                print("No metrics to display, please calculate metrics first.")
                return None, None
        else:
            if self.metrics is not None:
                #TODO: unhide individual image preds scrollable
                return None, None

    def sort_metrics(self):
        metrics = self.metrics.dataset_metrics
        metrics_mean_std = self.metrics.dataset_metrics_mean_std

        # Sort by num_detections and apply the same order to confidence_mean
        sorted_indices = sorted(range(len(metrics["num_detections"])), key=lambda i: metrics["num_detections"][i])

        sorted_num_detections = [metrics["num_detections"][i] for i in sorted_indices]
        sorted_conf_mean = [metrics["confidence_mean"][i] for i in sorted_indices]

        return sorted_num_detections, sorted_conf_mean

    def download_data(self):
        if self.metrics is not None:
            self.metrics.export_image_metrics_to_csv(filename=f'{self.dataset_name}_image_metrics.csv')
        else:
            print("No metrics to download, please calculate metrics first.")

    def get_models(self):
        return map(lambda model: model['model_name'], self.db.load_models())

    def get_datasets(self):
        return chain(
                *map(
                    lambda dataset: [dataset['dataset_name']] + 
                                    ([f"{dataset['dataset_name']} (denoised)"] if 'denoise_dataset_path' in dataset and dataset['denoise_dataset_path'] else []),
                    self.db.load_datasets()
                )
            )