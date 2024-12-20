from itertools import chain
from tinydb import Query
from PyQt5.QtCore import pyqtSignal, QObject
from DeepNeuronSeg.models.qa_metrics import DetectionQAMetrics
from DeepNeuronSeg.models.image_manager import ImageManager
from PyQt5.QtWidgets import QMessageBox
from torch.utils.data import DataLoader
from DeepNeuronSeg.utils.data_loader import ImageDataset
import os
from tqdm import tqdm
from PIL import Image

class EvaluationModel(QObject):

    calculated_dataset_metrics_signal = pyqtSignal(dict)
    update_metrics_labels_signal = pyqtSignal(dict, dict, dict, str, float)

    def __init__(self, db):
        super().__init__()
        self.db = db
        self.metrics = None
        # get first dataset path
        self.image_manager = ImageManager(self.db)
        self.set_first_dataset_path()
        self.confidence = 0.3

    def set_confidence(self, value):
        self.confidence = value

    def set_first_dataset_path(self):
        dataset_name = next(map(lambda dataset: dataset['dataset_name'], self.db.load_datasets()), None)
        self.dataset_path = self.get_dataset_path(dataset_name) if dataset_name else None
        if self.dataset_path is not None:
            self.image_manager.set_dataset_path(self.dataset_path)

    def get_dataset_path(self, dataset_name):
        if " (denoised)" in dataset_name:
            dn_dataset_name = dataset_name.replace(" (denoised)", "")
            return self.db.dataset_table.get(Query().dataset_name == dn_dataset_name).get('denoise_dataset_path')
        else:
            return self.db.dataset_table.get(Query().dataset_name == dataset_name).get('dataset_path')
        
    def inference_images(self, model_name):
        images = self.image_manager.get_images(subdir='images')
        images = [img[0] for img in images]
        model = self.load_model(self.model_path)
        self.inference_dir = os.path.join('data', 'inferences', model_name)
        os.makedirs(self.inference_dir, exist_ok=True)
        
        self.inference_result = model.predict(images, conf=self.confidence, visualize=False, save=False, show_labels=False, max_det=1000, verbose=False)

        self.save_inferences(images)
        
        
    def save_inferences(self, images):
        if not images:
            return
        for image, result in zip(images, self.inference_result):
            save_path = os.path.join(self.inference_dir, f'{os.path.splitext(os.path.basename(image))[0]}.png')
            mask_image = result.plot(labels=False, conf=False, boxes=False)
            mask_image = Image.fromarray(mask_image)
            mask_image.save(save_path)

    def get_inference_result(self, path):
        if not os.path.exists(self.inference_dir):
            return None
            
        image_name = os.path.basename(path)
        inference_path = os.path.join(self.inference_dir, image_name)
        
        if not os.path.exists(inference_path):
            return None
            
        return inference_path
            
    def calculate_metrics(self, model_name, dataset_name):
        self.model_path = self.db.model_table.get(Query().model_name == model_name)
        self.model_path = self.model_path["model_path"]
        self.dataset_path = self.get_dataset_path(dataset_name)

        self.image_manager.set_dataset_path(self.dataset_path)
        self.metrics = DetectionQAMetrics(self.model_path, self.dataset_path, self.confidence)
        self.update_metrics_labels_signal.emit(self.metrics.dataset_metrics_mean_std, self.metrics.dataset_metrics, self.metrics.get_analysis_metrics(), self.metrics.model_path, self.confidence)
        return self.metrics.dataset_metrics_mean_std 
    
    def display_graph(self):
        if self.metrics is not None:
            sorted_num_dets, sorted_conf_mean = self.sort_metrics()
            return sorted_num_dets, sorted_conf_mean
        else:
            return None, None

    def sort_metrics(self):
        metrics = self.metrics.dataset_metrics

        # Sort by num_detections and apply the same order to confidence_mean
        sorted_indices = sorted(range(len(metrics["num_detections"])), key=lambda i: metrics["num_detections"][i])

        sorted_num_detections = [metrics["num_detections"][i] for i in sorted_indices]
        sorted_conf_mean = [metrics["confidence_mean"][i] for i in sorted_indices]

        return sorted_num_detections, sorted_conf_mean

    def download_data(self, dataset_name):
        if self.metrics is not None:
            self.metrics.export_image_metrics_to_csv(filename=f'{dataset_name}_image_metrics.csv')
        else:
            # QMessageBox.warning(self, "No Metrics", "No metrics to download, please calculate metrics first.")
            something = 1

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
    
    def load_dataset(self, dataset_path):
        if os.path.exists(os.path.join(dataset_path, 'images')):
            image_path = os.path.join(dataset_path, 'images')
            dataset = ImageDataset(root_dir=image_path)
            self.batch_size = 4
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            return dataloader
        else:
            return None
        
    def load_model(self, model_path):
        # Load the model from the specified path
        from ultralytics import YOLO
        model = YOLO(model_path)
        return model