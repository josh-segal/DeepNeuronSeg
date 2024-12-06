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


class AnalysisTab(QWidget):

    calculated_outlier_data = pyqtSignal(dict) #TODO: implement this and switch all pyqt signals to class level instead of instance level

    def __init__(self, db):
        super().__init__()
        self.db = db
        layout = QVBoxLayout()
        metrics_layout = QGridLayout()
        self.uploaded_files = []
        
        # Model selection
        self.model_selector = QComboBox()
        self.model_selector.addItems(map(lambda model: model['model_name'], self.db.load_models()))
        
        # Image upload/selection
        self.select_btn = QPushButton("Select Images")
        self.inference_btn = QPushButton("Inference Images")
        self.save_btn = QPushButton("Save Inferences")
        self.download_btn = QPushButton("Download Data")  

        self.display_graph_checkbox = QCheckBox("Display Graph")

        self.confidence_mean_mean_label = QLabel("Average Confidence Score: N/A")
        self.confidence_mean_mean_label.setToolTip("""
        Average confidence of predictions per image, averaged across all images.
        """)
        self.confidence_mean_std_label = QLabel("Confidence Score Variability: N/A")
        self.confidence_mean_std_label.setToolTip("""
        Average confidence of predictions per image, standard deviation across all images.
        """)
        self.confidence_std_mean_label = QLabel("Average Confidence Spread: N/A")
        self.confidence_std_mean_label.setToolTip("""
        Standard deviation of confidence of predictions per image, averaged across all images.
        """)
        self.confidence_std_std_label = QLabel("Confidence Spread Variability: N/A")
        self.confidence_std_std_label.setToolTip("""
        Standard deviation of confidence of predictions per image, standard deviation across all images.
        """)
        self.num_detections_mean_label = QLabel("Average Number of Detections: N/A")
        self.num_detections_mean_label.setToolTip("""
        Average number of detections per image.
        """)
        self.num_detections_std_label = QLabel("Detection Count Variability: N/A")
        self.num_detections_std_label.setToolTip("""
        Standard deviation of number of detections per image.
        """)
        self.area_mean_mean_label = QLabel("Average Detection Area: N/A")
        self.area_mean_mean_label.setToolTip("""
        Average area of detections per image, averaged across all images.
        """)
        self.area_mean_std_label = QLabel("Detection Area Variability: N/A")
        self.area_mean_std_label.setToolTip("""
        Average area of detections per image, standard deviation across all images.
        """)
        self.area_std_mean_label = QLabel("Average Area Spread: N/A")
        self.area_std_mean_label.setToolTip("""
        Standard deviation of area of detections per image, averaged across all images.
        """)
        self.area_std_std_label = QLabel("Area Spread Variability: N/A")
        self.area_std_std_label.setToolTip("""
        Standard deviation of area of detections per image, standard deviation across all images.
        """)
        self.overlap_ratio_mean_label = QLabel("Average Overlap Rati: N/A")
        self.overlap_ratio_mean_label.setToolTip("""
        Average overlap ratio of detections per image.
        """)
        self.overlap_ratio_std_label = QLabel("Overlap Ratio Variability: N/A")
        self.overlap_ratio_std_label.setToolTip("""
        Standard deviation of overlap ratio of detections per image.
        """)

        self.analysis_confidence_mean_mean_label = QLabel("Analysis Average Confidence Score: N/A")
        self.analysis_confidence_mean_mean_label.setToolTip("""
        Average confidence of predictions per image, averaged across all images.
        """)
        self.analysis_confidence_mean_std_label = QLabel("Analysis Confidence Score Variability: N/A")
        self.analysis_confidence_mean_std_label.setToolTip("""
        Average confidence of predictions per image, standard deviation across all images.
        """)
        self.analysis_confidence_std_mean_label = QLabel("Analysis Average Confidence Spread: N/A")
        self.analysis_confidence_std_mean_label.setToolTip("""
        Standard deviation of confidence of predictions per image, averaged across all images.
        """)
        self.analysis_confidence_std_std_label = QLabel("Analysis Confidence Spread Variability: N/A")
        self.analysis_confidence_std_std_label.setToolTip("""
        Standard deviation of confidence of predictions per image, standard deviation across all images.
        """)
        self.analysis_num_detections_mean_label = QLabel("Analysis Average Number of Detections: N/A")
        self.analysis_num_detections_mean_label.setToolTip("""
        Average number of detections per image.
        """)
        self.analysis_num_detections_std_label = QLabel("Analysis Detection Count Variability: N/A")
        self.analysis_num_detections_std_label.setToolTip("""
        Standard deviation of number of detections per image.
        """)
        self.analysis_area_mean_mean_label = QLabel("Analysis Average Detection Area: N/A")
        self.analysis_area_mean_mean_label.setToolTip("""
        Average area of detections per image, averaged across all images.
        """)
        self.analysis_area_mean_std_label = QLabel("Analysis Detection Area Variability: N/A")
        self.analysis_area_mean_std_label.setToolTip("""
        Average area of detections per image, standard deviation across all images.
        """)
        self.analysis_area_std_mean_label = QLabel("Analysis Average Area Spread: N/A")
        self.analysis_area_std_mean_label.setToolTip("""
        Standard deviation of area of detections per image, averaged across all images.
        """)
        self.analysis_area_std_std_label = QLabel("Analysis Area Spread Variability: N/A")
        self.analysis_area_std_std_label.setToolTip("""
        Standard deviation of area of detections per image, standard deviation across all images.
        """)
        self.analysis_overlap_ratio_mean_label = QLabel("Analysis Average Overlap Rati: N/A")
        self.analysis_overlap_ratio_mean_label.setToolTip("""
        Average overlap ratio of detections per image.
        """)
        self.analysis_overlap_ratio_std_label = QLabel("Analysis Overlap Ratio Variability: N/A")
        self.analysis_overlap_ratio_std_label.setToolTip("""
        Standard deviation of overlap ratio of detections per image.
        """)
        
        self.canvas = FigureCanvas(Figure(figsize=(12, 5)))

        self.select_btn.clicked.connect(self.select_images)
        self.inference_btn.clicked.connect(self.inference_images)
        self.save_btn.clicked.connect(self.save_inferences)
        self.download_btn.clicked.connect(self.download_data)
        
        layout.addWidget(QLabel("Trained Model:"))
        layout.addWidget(self.model_selector)
        layout.addWidget(self.select_btn)
        layout.addWidget(self.inference_btn)
        layout.addWidget(self.display_graph_checkbox)
        layout.addWidget(self.save_btn)
        layout.addWidget(self.canvas)

        metrics_layout.addWidget(self.confidence_mean_mean_label, 0, 0)
        metrics_layout.addWidget(self.confidence_mean_std_label, 0, 1)
        metrics_layout.addWidget(self.confidence_std_mean_label, 0, 2)
        metrics_layout.addWidget(self.confidence_std_std_label, 0, 3)
        metrics_layout.addWidget(self.num_detections_mean_label, 0, 4)
        metrics_layout.addWidget(self.num_detections_std_label, 0, 5)

        metrics_layout.addWidget(self.analysis_confidence_mean_mean_label, 1, 0)
        metrics_layout.addWidget(self.analysis_confidence_mean_std_label, 1, 1)
        metrics_layout.addWidget(self.analysis_confidence_std_mean_label, 1, 2)
        metrics_layout.addWidget(self.analysis_confidence_std_std_label, 1, 3)
        metrics_layout.addWidget(self.analysis_num_detections_mean_label, 1, 4)
        metrics_layout.addWidget(self.analysis_num_detections_std_label, 1, 5)

        metrics_layout.addWidget(self.area_mean_mean_label, 2, 0)
        metrics_layout.addWidget(self.area_mean_std_label, 2, 1)
        metrics_layout.addWidget(self.area_std_mean_label, 2, 2)
        metrics_layout.addWidget(self.area_std_std_label, 2, 3)
        metrics_layout.addWidget(self.overlap_ratio_mean_label, 2, 4)
        metrics_layout.addWidget(self.overlap_ratio_std_label, 2, 5)

        metrics_layout.addWidget(self.analysis_area_mean_mean_label, 3, 0)
        metrics_layout.addWidget(self.analysis_area_mean_std_label, 3, 1)
        metrics_layout.addWidget(self.analysis_area_std_mean_label, 3, 2)
        metrics_layout.addWidget(self.analysis_area_std_std_label, 3, 3)
        metrics_layout.addWidget(self.analysis_overlap_ratio_mean_label, 3, 4)
        metrics_layout.addWidget(self.analysis_overlap_ratio_std_label, 3, 5)

        layout.addLayout(metrics_layout)
        self.setLayout(layout)

    # TODO: select multiple models and compare results / ensemble ?
    def select_images(self):
        self.uploaded_files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png)")

    def inference_images(self):
        from ultralytics import YOLO
        self.model_name = self.model_selector.currentText()
        self.model_path = self.db.model_table.get(Query().model_name == self.model_name).get('model_path')
        self.model_denoise = self.db.model_table.get(Query().model_name == self.model_name).get('denoise')
        self.model = YOLO(self.model_path)
        self.inference_dir = os.path.join('data/datasets/dataset_0/results/testModel', 'inference')
        os.makedirs(self.inference_dir, exist_ok=True)
        if not self.uploaded_files:
            print("No images selected")
            return  
        if self.model_denoise is not None:
            #TODO: NEED TO TEST WITH A REAL MODEL
            dn_model = DenoiseModel(dataset_path='idc update to not need', model_path=self.model_denoise)
            uploaded_images = [
                dn_model.denoise_image(Image.open(image_path).convert('L')) 
                for image_path in self.uploaded_files
            ]
        self.inference_result = self.model.predict(uploaded_images, conf=0.3, visualize=False, save=False, show_labels=False, max_det=1000, verbose=False)

        self.update_metrics_labels()

        self.update_analysis_metrics_labels()

        if True:
            self.save_inferences()
        if self.display_graph_checkbox.isChecked():
            self.plot_inferences_against_dataset()

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

    def download_data(self):
        if self.analysis_metrics is not None:
            self.analysis_metrics.export_image_metrics_to_csv()
        else:
            print("No metrics to download, please calculate metrics first.")

    def receive_dataset_metrics(self, dataset_metrics_model: DetectionQAMetrics):
        print("received dataset_metrics_model", dataset_metrics_model)
        self.dataset_metrics_model = dataset_metrics_model
        

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

        # print("colors", colors)

        ax1.bar(range(len(sorted_conf_mean)), sorted_conf_mean, color=colors, edgecolor='black', label='_nolegend_')
        ax1.bar(0, 0, width=0, color='salmon', edgecolor='black', label='Original Data')
        ax1.bar(0, 0, width=0, color='skyblue', edgecolor='black', label='New Data')
        ax1.set_title("Mean Confidence of Predictions Per Image")
        ax1.set_xlabel("Image")
        ax1.set_ylabel("Mean Confidence")
        ax1.legend()

        ax2.bar(range(len(sorted_num_detections)), sorted_num_detections, color=colors, edgecolor='black', label='_nolegend_')
        ax2.bar(0, 0, width=0, color='salmon', edgecolor='black', label='Original Data')
        ax2.bar(0, 0, width=0, color='skyblue', edgecolor='black', label='New Data')
        ax2.set_title("Number of Detections Per Image")
        ax2.set_xlabel("Image")
        ax2.set_ylabel("Number of Detections")
        ax2.legend()

        # Adjust layout and render
        self.canvas.figure.tight_layout()
        self.canvas.draw()
        # print("plotted")

        # each list is a metric where values in list are individual image values
        # restructure to be a list of list of image metrics each list contains all metrics for a single image
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

    def update_metrics_labels(self):
        self.metrics_mean_std = self.dataset_metrics_model.dataset_metrics_mean_std
        

        #TODO: how to know what variance is acceptable, n-fold cross variation as baseline, how to calculate?
        self.confidence_mean_mean_label.setText(f"Average Confidence Score: {self.metrics_mean_std['confidence_mean_mean']:.2f}")
        self.confidence_mean_std_label.setText(f"Confidence Score Variability: {self.metrics_mean_std['confidence_mean_std']:.2f}")
        self.confidence_std_mean_label.setText(f"Average Confidence Spread: {self.metrics_mean_std['confidence_std_mean']:.2f}")
        self.confidence_std_std_label.setText(f"Confidence Spread Variability: {self.metrics_mean_std['confidence_std_std']:.2f}")
        self.num_detections_mean_label.setText(f"Average Number of Detections: {self.metrics_mean_std['num_detections_mean']:.2f}")
        self.num_detections_std_label.setText(f"Detection Count Variability: {self.metrics_mean_std['num_detections_std']:.2f}")
        self.area_mean_mean_label.setText(f"Average Detection Area: {self.metrics_mean_std['area_mean_mean']:.2f}")
        self.area_mean_std_label.setText(f"Detection Area Variability: {self.metrics_mean_std['area_mean_std']:.2f}")
        self.area_std_mean_label.setText(f"Average Area Spread: {self.metrics_mean_std['area_std_mean']:.2f}")
        self.area_std_std_label.setText(f"Area Spread Variability: {self.metrics_mean_std['area_std_std']:.2f}")
        self.overlap_ratio_mean_label.setText(f"Average Overlap Ratio: {self.metrics_mean_std['overlap_ratio_mean']:.2f}")
        self.overlap_ratio_std_label.setText(f"Overlap Ratio Variability: {self.metrics_mean_std['overlap_ratio_std']:.2f}")
    
    def update_analysis_metrics_labels(self):
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy the selected files to the temporary directory
            for file in self.uploaded_files:
                shutil.copy(file, temp_dir)
            
            # Use the temporary directory as the dataset path
            #TODO: get rid of evaluation tab instance
            self.analysis_metrics = DetectionQAMetrics(self.metrics.model_path, temp_dir)

        self.analysis_confidence_mean_mean_label.setText(f"Analysis Average Confidence Score: {self.analysis_metrics.dataset_metrics_mean_std['confidence_mean_mean']:.2f}")
        self.analysis_confidence_mean_std_label.setText(f"Analysis Confidence Score Variability: {self.analysis_metrics.dataset_metrics_mean_std['confidence_mean_std']:.2f}")
        self.analysis_confidence_std_mean_label.setText(f"Analysis Average Confidence Spread: {self.analysis_metrics.dataset_metrics_mean_std['confidence_std_mean']:.2f}")
        self.analysis_confidence_std_std_label.setText(f"Analysis Confidence Spread Variability: {self.analysis_metrics.dataset_metrics_mean_std['confidence_std_std']:.2f}")
        self.analysis_num_detections_mean_label.setText(f"Analysis Average Number of Detections: {self.analysis_metrics.dataset_metrics_mean_std['num_detections_mean']:.2f}")
        self.analysis_num_detections_std_label.setText(f"Analysis Detection Count Variability: {self.analysis_metrics.dataset_metrics_mean_std['num_detections_std']:.2f}")
        self.analysis_area_mean_mean_label.setText(f"Analysis Average Detection Area: {self.analysis_metrics.dataset_metrics_mean_std['area_mean_mean']:.2f}")
        self.analysis_area_mean_std_label.setText(f"Analysis Detection Area Variability: {self.analysis_metrics.dataset_metrics_mean_std['area_mean_std']:.2f}")
        self.analysis_area_std_mean_label.setText(f"Analysis Average Area Spread: {self.analysis_metrics.dataset_metrics_mean_std['area_std_mean']:.2f}")
        self.analysis_area_std_std_label.setText(f"Analysis Area Spread Variability: {self.analysis_metrics.dataset_metrics_mean_std['area_std_std']:.2f}")
        self.analysis_overlap_ratio_mean_label.setText(f"Analysis Average Overlap Ratio: {self.analysis_metrics.dataset_metrics_mean_std['overlap_ratio_mean']:.2f}")
        self.analysis_overlap_ratio_std_label.setText(f"Analysis Overlap Ratio Variability: {self.analysis_metrics.dataset_metrics_mean_std['overlap_ratio_std']:.2f}")

    def update(self):
        pass