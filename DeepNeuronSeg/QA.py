import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
from tqdm import tqdm


# @dataclass
# class ImageMetrics:
#     """Class to store computed metrics for a single image"""
#     mean_confidence: float
#     confidence_std: float
#     num_detections: int
#     avg_area: float
#     area_std: float
#     overlap_ratio: float


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.ToTensor()
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]

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
        self.model = self.load_model(model_path)
        print('loaded model')
        self.dataset = self.load_dataset(dataset_path)
        print('loaded dataset')
        self.dataset_metrics = {
            'confidence_mean': [],
            'confidence_std': [],
            'num_detections': [],
            'area_mean': [],
            'area_std': [],
            'overlap_ratio': []
        }
        self.confidences = []
        self.bbox_bounds = []

        self.compute_metrics()

    def compute_metrics(self):
        for images in tqdm(self.dataset, desc="Processing Images", unit=f"{self.batch_size} image(s)"):
            # Get the predictions
            preds = self.model.predict(images, conf=0.3, max_det=1000, verbose=False)
            # print('got predictions from batch')
            # Format the predictions
            self.format_predictions(preds)

        for img_conf, img_bboxes in  tqdm(zip(self.confidences, self.bbox_bounds), total=len(self.confidences), desc="Computing Metrics", unit="image"):
            # print("computing metrics for image")
            img_metrics = self.compute_image_metrics(img_conf, img_bboxes)

            self.dataset_metrics["confidence_mean"].append(img_metrics["confidence_mean"])
            self.dataset_metrics["confidence_std"].append(img_metrics["confidence_std"])
            self.dataset_metrics["num_detections"].append(img_metrics["num_detections"])
            self.dataset_metrics["area_mean"].append(img_metrics["area_mean"])
            self.dataset_metrics["area_std"].append(img_metrics["area_std"])
            self.dataset_metrics["overlap_ratio"].append(img_metrics["overlap_ratio"])
            
        # print("computing dataset metrics")
        self.dataset_metrics_mean_std = {
            'confidence_mean_mean': np.mean(self.dataset_metrics['confidence_mean']),
            'confidence_mean_std': np.std(self.dataset_metrics['confidence_mean']),
            'confidence_std_mean': np.mean(self.dataset_metrics['confidence_std']),
            'confidence_std_std': np.std(self.dataset_metrics['confidence_std']),
            'num_detections_mean': np.mean(self.dataset_metrics['num_detections']),
            'num_detections_std': np.std(self.dataset_metrics['num_detections']),
            'area_mean_mean': np.mean(self.dataset_metrics['area_mean']),
            'area_mean_std': np.std(self.dataset_metrics['area_mean']),
            'area_std_mean': np.mean(self.dataset_metrics['area_std']),
            'area_std_std': np.std(self.dataset_metrics['area_std']),
            'overlap_ratio_mean': np.mean(self.dataset_metrics['overlap_ratio']),
            'overlap_ratio_std': np.std(self.dataset_metrics['overlap_ratio'])
        }
        # print('done computing image and dataset metrics')

    def format_predictions(self, predictions):
        for pred in predictions:
            conf = pred.boxes.conf
            boxes = pred.boxes.xyxy

            self.confidences.append(conf)
            self.bbox_bounds.append(boxes)

    def change_last_to_std(var_name):
        parts = var_name.split('_')
        parts[-1] = 'std'
        return '_'.join(parts)

    def compute_image_metrics(self, confs, boxes):
        confidence_mean = np.mean(confs.numpy())
        confidence_std = np.std(confs.numpy())
        num_dets = len(confs.numpy())
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        area_mean = np.mean(areas)
        area_std = np.std(areas)
        overlap_ratio = self.compute_overlap(boxes)
        
        return {
            'confidence_mean': confidence_mean,
            'confidence_std': confidence_std,
            'num_detections': num_dets,
            'area_mean': area_mean,
            'area_std': area_std,
            'overlap_ratio': overlap_ratio
        }
    
    def compute_overlap(self, boxes):
        """ Calculate the overlap ratio between bounding boxes: 1 means 100% overlap, 0 means no overlap """
        intersection_areas = []
        box_areas = []
    
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                A = boxes[i]
                B = boxes[j]
                
                # Calculate intersection area
                x1_intersection = max(A[0], B[0])
                y1_intersection = max(A[1], B[1])
                x2_intersection = min(A[2], B[2])
                y2_intersection = min(A[3], B[3])
                intersection_area = max(0, x2_intersection - x1_intersection) * max(0, y2_intersection - y1_intersection)
                intersection_areas.append(intersection_area)
                
                # Calculate box areas
                box_area_A = (A[2] - A[0]) * (A[3] - A[1])
                box_area_B = (B[2] - B[0]) * (B[3] - B[1])
                box_areas.append(box_area_A)
                box_areas.append(box_area_B)
        
        total_area = sum(box_areas)
        if total_area == 0:
            return 0
        else:
            return sum(intersection_areas) / total_area

    def compute_variance(self, analysis_metrics):
        """ Compute the variance of the computed metrics """
        metric_variance = {}
        for metric, value in self.dataset_metrics_mean_std.items():
            analysis_metric = "analysis_" + metric
            if analysis_metric in analysis_metrics:
                variance_metric = "variance_" + metric
                std_metric = change_last_to_std(metric)

                metric_variance[variance_metric] = (analysis_metrics[analysis_metric] - value) / self.dataset_metrics_mean_std[std_metric]
        
        return metric_variance

    def compute_quality_score(self, variancs):
        return np.mean(variances.values())

    def load_model(self, model_path):
        # Load the model from the specified path
        from ultralytics import YOLO
        model = YOLO(model_path)
        return model

    def load_dataset(self, dataset_path):
        dataset = ImageDataset(root_dir=dataset_path)
        self.batch_size = 4
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader
        