from DeepNeuronSeg.models.denoise_model import DenoiseModel
from DeepNeuronSeg.models.qa_metrics import DetectionQAMetrics
import shutil
import tempfile
import numpy as np
import os
from PIL import Image
from PyQt5.QtCore import pyqtSignal, QObject
from tinydb import Query


class AnalysisModel(QObject):

    calculated_outlier_data = pyqtSignal(dict)
    dataset_metrics_signal = pyqtSignal(dict)
    analysis_metrics_signal = pyqtSignal(dict)

    def __init__(self, db):
        super().__init__()
        self.db = db
        self.uploaded_files = []
        self.dataset_metrics = None
        self.sorted_all_num_detections, self.sorted_all_conf_mean, self.colors = None, None, None

    def load_models(self):
        return self.db.load_models()

    def inference_images(self, name_of_model, uploaded_files):
        print("model inferencing images")
        from ultralytics import YOLO
        self.uploaded_files = uploaded_files
        self.model_path = self.db.model_table.get(Query().model_name == name_of_model).get('model_path')
        self.model_denoise = self.db.model_table.get(Query().model_name == name_of_model).get('denoise')
        self.model = YOLO(self.model_path)
        self.inference_dir = os.path.join('data', 'inferences', name_of_model)
        os.makedirs(self.inference_dir, exist_ok=True)
        if not uploaded_files:
            print("No images selected")
            return  
        if self.model_denoise:
            #TODO: NEED TO TEST WITH A REAL MODEL
            dn_model = DenoiseModel(dataset_path='idc update to not need', model_path=self.model_denoise)
            uploaded_images = [
                dn_model.denoise_image(Image.open(image_path).convert('L')) 
                for image_path in uploaded_files
            ]
        else:
            uploaded_images = [Image.open(image_path) for image_path in uploaded_files]
            
        self.inference_result = self.model.predict(uploaded_images, conf=0.3, visualize=False, save=False, show_labels=False, max_det=1000, verbose=False)

        self.update_metrics_labels()

        self.sorted_all_num_detections, self.sorted_all_conf_mean, self.colors = self.compute_analysis()

    def display_graph(self):
        if self.sorted_all_num_detections is not None and self.sorted_all_conf_mean is not None and self.colors is not None:
            return self.sorted_all_num_detections, self.sorted_all_conf_mean, self.colors
        else:
            return None, None, None

    def receive_dataset_metrics(self, dataset_metrics, variance_baselines, model_path):
        self.dataset_metrics = dataset_metrics
        self.variance_baselines = variance_baselines
        self.analysis_model_path = model_path

    def update_metrics_labels(self):
        if self.dataset_metrics is not None:
            self.dataset_metrics_signal.emit(self.dataset_metrics)
            self.update_analysis_metrics_labels()
        else:
            print("No dataset metrics, please calculate metrics first in Evaluation Tab.")

    def update_analysis_metrics_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy the selected files to the temporary directory
            if not self.uploaded_files:
                print("No uploaded files to process.")
                return
            for file in self.uploaded_files:
                shutil.copy(file, temp_dir)
            
            # Use the temporary directory as the dataset path
            #TODO: get rid of evaluation tab instance
            self.analysis_metrics = DetectionQAMetrics(self.analysis_model_path, temp_dir)
            self.analysis_metrics_signal.emit(self.analysis_metrics.dataset_metrics_mean_std)

    def download_data(self):
        if self.analysis_metrics is not None:
            self.analysis_metrics.export_image_metrics_to_csv()
        else:
            print("No metrics to download, please calculate metrics first.")

    def save_inferences(self):
        for file, result in zip(self.uploaded_files, self.inference_result):
                masks = result.masks
                # boxes = result.boxes.xyxy
                confs = result.boxes.conf
                mask_num = len(masks)
                # print(file, '------------')
                save_path = os.path.join(self.inference_dir, f'{os.path.splitext(os.path.basename(file))[0]}_{mask_num}.png')
                mask_image = result.plot(labels=False, conf=False, boxes=False)
                mask_image = Image.fromarray(mask_image)
                # print(save_path)
                mask_image.save(save_path)
                #TODO: save name, masks, and confs to db

    def compute_analysis(self):
        sorted_num_detections, sorted_conf_mean = self.sort_metrics()
        sorted_additional_num_detections, sorted_additiona_conf_mean = self.sort_additions_metrics()

        sorted_all_num_detections, sorted_all_conf_mean, merged_additional_indices = self.merge_sorted_metrics(sorted_num_detections, sorted_conf_mean, sorted_additional_num_detections, sorted_additiona_conf_mean)
        colors = self.find_colors(merged_additional_indices, sorted_all_conf_mean)

        return sorted_all_num_detections, sorted_all_conf_mean, colors

    def sort_metrics(self):
        sorted_indices = sorted(range(len(self.variance_baselines["num_detections"])), key=lambda i: self.variance_baselines["num_detections"][i])
        
        sorted_num_detections = [self.variance_baselines["num_detections"][i] for i in sorted_indices]
        sorted_conf_mean = [self.variance_baselines["confidence_mean"][i] for i in sorted_indices]
    
        return sorted_num_detections, sorted_conf_mean

    def sort_additions_metrics(self):
        additional_num_detections, additional_conf_mean = self.format_preds(self.inference_result)
        additional_sorted_indices = sorted(range(len(additional_num_detections)), key=lambda i: additional_num_detections[i], reverse=True)
        
        sorted_additional_num_detections = [additional_num_detections[i] for i in additional_sorted_indices]
        sorted_additiona_conf_mean = [additional_conf_mean[i] for i in additional_sorted_indices]

        return sorted_additional_num_detections, sorted_additiona_conf_mean

    def merge_sorted_metrics(self, sorted_num_detections, sorted_conf_mean, sorted_additional_num_detections, sorted_additiona_conf_mean):
        merged_additional_indices = []
        for num in sorted_additional_num_detections:
            if num > sorted_num_detections[-1]:
                merged_additional_indices.append(len(sorted_num_detections))
                continue
            for i, sorted_num in enumerate(sorted_num_detections):
                if num <= sorted_num:
                    merged_additional_indices.append(i)
                    break

        for i, num in enumerate(merged_additional_indices):
            sorted_num_detections.insert(num, sorted_additional_num_detections[i])
            sorted_conf_mean.insert(num, sorted_additiona_conf_mean[i])

        return sorted_num_detections, sorted_conf_mean, merged_additional_indices

    def find_colors(self, merged_additional_indices, sorted_conf_mean):
        indicies_to_color = [(len(merged_additional_indices) - (index + 1)) + value for index, value in enumerate(merged_additional_indices)]
        colors = ['skyblue' if i in indicies_to_color else 'salmon' for i in range(len(sorted_conf_mean))]

        return colors

    def calculate_variance(self):
        reshaped_analysis_list_of_list = [dict(zip(variance_baselines.keys(), values)) for values in zip(*variance_baselines.values())]

        variance_list_of_list = []
        quality_score_list = []
        for i, image in enumerate(reshaped_analysis_list_of_list):
            # print(f"Image {i+1} metrics: {image}")
            # print('-'*50)
            variance_list = self.compute_variance(image)
            variance_list_of_list.append(variance_list)
            # print(f"Image {i+1} variance: {variance_list}")
            # print('-'*50)
            quality_score = self.compute_quality_score(variance_list)
            quality_score_list.append({self.uploaded_files[i]: quality_score})
            # print(f"Image {i+1} quality score: {quality_score} from {self.uploaded_files[i]}")
            # print('-'*50)
        
        self.calculated_outlier_data.emit(quality_score_list)


    def compute_variance(self, analysis_metrics):
        """ Compute the variance of the computed metrics """
        metric_variance = {}
        for metric, value in self.dataset_metrics.items():
            analysis_metric = "analysis_" + metric
            if analysis_metric in analysis_metrics:
                variance_metric = "variance_" + metric
                std_metric = self.change_last_to_std(metric)

                metric_variance[variance_metric] = (analysis_metrics[analysis_metric] - value) / self.dataset_metrics[std_metric]
        
        return metric_variance

    def compute_quality_score(self, variances):
        return np.mean(np.array(list(variances.values())))

    def change_last_to_std(var_name):
        parts = var_name.split('_')
        parts[-1] = 'std'
        return '_'.join(parts)
        

    def format_preds(self, predictions):
        # print("formatting preds")
        num_detections_list = []
        mean_confidence_list = []
        for pred in predictions:
            conf = pred.boxes.conf
            cell_num = len(conf)
            num_detections_list.append(cell_num)
            mean_confidence_list.append(np.mean(conf.numpy()))

        return num_detections_list, mean_confidence_list