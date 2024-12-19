import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from DeepNeuronSeg.utils.data_loader import ImageDataset
from DeepNeuronSeg.models.denoise_model import DenoiseModel
from PyQt5.QtWidgets import QMessageBox

class DetectionQAMetrics:
    def __init__(self, model_path, dataset_path, confidence):
        self.model_path = model_path
        self.model = self.load_model(model_path)
        self.denoised = os.path.basename(dataset_path) == 'denoised'
        self.dataset = self.load_dataset(dataset_path)
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
        self.confidence = confidence

        self.compute_metrics()

    def compute_metrics(self):
        for images, image_names in tqdm(self.dataset, desc="Processing Images", unit=f"{self.batch_size} image(s)"):
            self.image_names = image_names
            # Get the predictions
            preds = self.model.predict(images, conf=self.confidence, max_det=1000, verbose=False)
            # Format the predictions
            self.format_predictions(preds)

        self.img_level_metrics = []
        for img_conf, img_bboxes, img_name in tqdm(zip(self.confidences, self.bbox_bounds, image_names), total=len(self.confidences), desc="Computing Metrics", unit="image"):
            img_metrics = self.compute_image_metrics(img_conf, img_bboxes)

            self.img_level_metrics.append(img_metrics)

            for key, value in img_metrics.items():
                self.dataset_metrics[key].append(value)
            
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

    def export_image_metrics_to_csv(self, filename='inference_image_metrics.csv'):
        if not self.img_level_metrics:
            QMessageBox.warning(None, "No Metrics", "No metrics to export.")
            return
        
        df = pd.DataFrame(self.img_level_metrics, index=self.image_names)
        df = df.rename_axis("image_names", axis="index")
        df.to_csv(filename)
        
        QMessageBox.information(None, "Metrics Exported", f"Metrics exported to {filename}")

    def format_predictions(self, predictions):
        for pred in predictions:
            conf = pred.boxes.conf
            boxes = pred.boxes.xyxy

            self.confidences.append(conf)
            self.bbox_bounds.append(boxes)

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

    def get_analysis_metrics(self):
        return {
            'analysis_confidence_mean_mean': self.dataset_metrics['confidence_mean'],
            'analysis_confidence_std_std': self.dataset_metrics['confidence_std'],
            'analysis_num_detections_mean': self.dataset_metrics['num_detections'],
            'analysis_area_mean_mean': self.dataset_metrics['area_mean'],
            'analysis_area_std_std': self.dataset_metrics['area_std'],
            'analysis_overlap_ratio_mean': self.dataset_metrics['overlap_ratio']
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
            return float(sum(intersection_areas) / total_area)

    def load_model(self, model_path):
        # Load the model from the specified path
        from ultralytics import YOLO
        model = YOLO(model_path)
        return model

    def load_dataset(self, dataset_path):
        not_temp = os.path.exists(os.path.join(dataset_path, 'images'))
        if not_temp:
            image_path = os.path.join(dataset_path, 'images')
            dataset = ImageDataset(root_dir=image_path)
        elif self.denoised:
            if os.path.exists(os.path.join(dataset_path, 'denoise_model.pth')):
                model_path = os.abspath(os.path.join(dataset_path, 'denoise_model.pth'))
                dn_model = DenoiseModel(dataset_path, model_path=model_path)
                
            else:
                dn_model = DenoiseModel(dataset_path)

            transform = transforms.Compose([
                    transforms.Lambda(lambda image: dn_model.denoise_image(image)),
                    transforms.Resize((512, 512)),
                    transforms.Lambda(lambda image: image.convert("RGB")),
                    transforms.ToTensor(),
                ])
            dataset = ImageDataset(root_dir=dataset_path, transform=transform)
        else:
            dataset = ImageDataset(root_dir=dataset_path)

        self.batch_size = 4
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    
        