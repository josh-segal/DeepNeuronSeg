import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

@dataclass
class ImageMetrics:
    """Class to store computed metrics for a single image"""
    mean_confidence: float
    confidence_std: float
    num_detections: int
    avg_area: float
    spatial_coverage: float
    mask_complexity: float
    edge_proximity: float
    overlap_ratio: float

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image

class DetectionQAMetrics:

    def __init__(self, model_path, dataset_path):
        self.model = load_model(model_path)
        self.dataset = load_dataset(dataset_path)
        self.metrics = []

    def compute_metrics(self):
        for images in self.dataset:
            # Get the predictions
            preds = self.model.predict(image)
            # Format the predictions
            confs, boxes = format_predictions(preds)

            # Compute metrics
            image_metrics = compute_single_metric(confs, boxes)

            # Add predictions to dataset metrics
            self.metrics.append(image_metrics)

    def format_predictions(self, predictions):
        confidences = []
        bbox_bounds = []
        for pred in predictions:
            conf = pred.boxes.conf
            boxes = pred.boxes.xyxy

            confidences.append(conf)
            bbox_bounds.append(boxes)

        return confidences, bbox_bounds



    def compute_single_metric(self, confidences, bbox_bounds):
        mean_confidence = np.mean(confidences)
        confidence_std = np.std(confidences)
        num_detections = len(confidences)
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bbox_bounds]
        avg_area = np.mean(areas)
        area_std = np.std(areas)
        


    def load_model(self, model_path):
        # Load the model from the specified path
        from ultralytics import YOLO
        model = YOLO(model_path)
        return model

    def load_dataset(self, dataset_path):
        dataset = ImageDataset(root_dir=dataset_path)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        return dataloader
        