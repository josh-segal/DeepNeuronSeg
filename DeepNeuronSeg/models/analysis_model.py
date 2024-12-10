from DeepNeuronSeg.models.denoise_model import DenoiseModel
from DeepNeuronSeg.models.qa_metrics import DetectionQAMetrics
import shutil
import tempfile
import numpy as np
import os
from PIL import Image
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QLabel, QComboBox, QPushButton, QFileDialog, QCheckBox
from PyQt5.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from tinydb import Query


class AnalysisModel:

    calculated_outlier_data = pyqtSignal(dict)

    def __init__(self, db):
        super().__init__()
        self.db = db
        self.uploaded_files = []

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
        if self.model_denoise is not None:
            #TODO: NEED TO TEST WITH A REAL MODEL
            dn_model = DenoiseModel(dataset_path='idc update to not need', model_path=self.model_denoise)
            uploaded_images = [
                dn_model.denoise_image(Image.open(image_path).convert('L')) 
                for image_path in uploaded_files
            ]
        self.inference_result = self.model.predict(uploaded_images, conf=0.3, visualize=False, save=False, show_labels=False, max_det=1000, verbose=False)

        self.update_metrics_labels()

        self.update_analysis_metrics_labels()

        if True:
            self.save_inferences()
        if self.display_graph_checkbox.isChecked():
            self.plot_inferences_against_dataset()

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
        
    def plot_inferences_against_dataset(self):
        self.canvas.figure.clf()
        ax1, ax2 = self.canvas.figure.subplots(1, 2)

        #TODO: add evaluation tab data to shared data to reference here without passing instance of evaluation tab here
        # Sort by num_detections and apply the same order to confidence_mean
        sorted_indices = sorted(range(len(self.dataset_metrics_model.dataset_metrics["num_detections"])), key=lambda i: self.evaluation_tab.metrics.dataset_metrics["num_detections"][i])
        
        sorted_num_detections = [self.dataset_metrics_model.dataset_metrics["num_detections"][i] for i in sorted_indices]
        sorted_conf_mean = [self.dataset_metrics_model.dataset_metrics["confidence_mean"][i] for i in sorted_indices]

        # print("sorted_num_detections", sorted_num_detections)
        # print("sorted_conf_mean", sorted_conf_mean)

        additional_num_detections, additional_conf_mean = self.format_preds(self.inference_result)
        additional_sorted_indices = sorted(range(len(additional_num_detections)), key=lambda i: additional_num_detections[i], reverse=True)
        
        sorted_additional_num_detections = [additional_num_detections[i] for i in additional_sorted_indices]
        sorted_additiona_conf_mean = [additional_conf_mean[i] for i in additional_sorted_indices]

        # print("sorted_additional_num_detections", sorted_additional_num_detections)
        # print("sorted_additiona_conf_mean", sorted_additiona_conf_mean)

        merged_additional_indices = []
        for num in sorted_additional_num_detections:
            if num > sorted_num_detections[-1]:
                merged_additional_indices.append(len(sorted_num_detections))
                continue
            for i, sorted_num in enumerate(sorted_num_detections):
                if num <= sorted_num:
                    merged_additional_indices.append(i)
                    break

        # print("merged_additional_indices", merged_additional_indices)

        for i, num in enumerate(merged_additional_indices):
            sorted_num_detections.insert(num, sorted_additional_num_detections[i])
            sorted_conf_mean.insert(num, sorted_additiona_conf_mean[i])

        # print("merged_sorted_num_detections", sorted_num_detections)
        # print("merged_sorted_conf_mean", sorted_conf_mean)

    
        indicies_to_color = [(len(merged_additional_indices) - (index + 1)) + value for index, value in enumerate(merged_additional_indices)]

        # print("indicies_to_color", indicies_to_color)

        # Plotting histograms
        colors = ['skyblue' if i in indicies_to_color else 'salmon' for i in range(len(sorted_conf_mean))]

        # each list is a metric where values in list are individual image values
        # restructure to be a list of list of image metrics each list contains all metrics for a single image


    def diff_function_for_plotting(self):
        analysis_list_of_list = self.analysis_metrics.get_analysis_metrics()
        reshaped_analysis_list_of_list = [dict(zip(analysis_list_of_list.keys(), values)) for values in zip(*analysis_list_of_list.values())]

        variance_list_of_list = []
        quality_score_list = []
        for i, image in enumerate(reshaped_analysis_list_of_list):
            # print(f"Image {i+1} metrics: {image}")
            # print('-'*50)
            variance_list = self.dataset_metrics_model.compute_variance(image)
            variance_list_of_list.append(variance_list)
            # print(f"Image {i+1} variance: {variance_list}")
            # print('-'*50)
            quality_score = self.dataset_metrics_model.compute_quality_score(variance_list)
            quality_score_list.append({self.uploaded_files[i]: quality_score})
            # print(f"Image {i+1} quality score: {quality_score} from {self.uploaded_files[i]}")
            # print('-'*50)
        
        self.calculated_outlier_data.emit(quality_score_list)
        

    def format_preds(self, predictions):
        print("formatting preds")
        num_detections_list = []
        mean_confidence_list = []
        for pred in predictions:
            conf = pred.boxes.conf
            cell_num = len(conf)
            num_detections_list.append(cell_num)
            mean_confidence_list.append(np.mean(conf.numpy()))

        # print("num_detections_list", num_detections_list)
        # print("mean_confidence_list", mean_confidence_list)

        return num_detections_list, mean_confidence_list