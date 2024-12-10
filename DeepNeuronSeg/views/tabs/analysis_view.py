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


class AnalysisView(QWidget):

    inference_images_signal = pyqtSignal(str, list)
    save_inferences_signal = pyqtSignal()
    download_data_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        metrics_layout = QGridLayout()
        self.uploaded_files = []
        
        # Model selection
        self.model_selector = QComboBox()
        # self.model_selector.addItems(map(lambda model: model['model_name'], self.db.load_models()))
        
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

    def inference_images(self):
        self.model_name = self.model_selector.currentText()
        self.inference_images_signal.emit(self.model_name, self.uploaded_files)
        print("view inference_images_signal emitted")

    def save_inferences(self):
        self.save_inferences_signal.emit()

    def download_data(self):
        self.download_data_signal.emit()

    def set_model_names(self, models):
        self.model_selector.clear()
        self.model_selector.addItems(models)

    # TODO: select multiple models and compare results / ensemble ?
    def select_images(self):
        self.uploaded_files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png)")
        
    def plot_inferences_against_dataset(self, sorted_conf_mean, sorted_num_detections, colors):
        self.canvas.figure.clf()
        ax1, ax2 = self.canvas.figure.subplots(1, 2)

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

    def update_metrics_labels(self, metrics):
        # self.metrics_mean_std = self.dataset_metrics_model.dataset_metrics_mean_std
        

        #TODO: how to know what variance is acceptable, n-fold cross variation as baseline, how to calculate?
        self.confidence_mean_mean_label.setText(f"Average Confidence Score: {self.metrics['confidence_mean_mean']:.2f}")
        self.confidence_mean_std_label.setText(f"Confidence Score Variability: {self.metrics['confidence_mean_std']:.2f}")
        self.confidence_std_mean_label.setText(f"Average Confidence Spread: {self.metrics['confidence_std_mean']:.2f}")
        self.confidence_std_std_label.setText(f"Confidence Spread Variability: {self.metrics['confidence_std_std']:.2f}")
        self.num_detections_mean_label.setText(f"Average Number of Detections: {self.metrics['num_detections_mean']:.2f}")
        self.num_detections_std_label.setText(f"Detection Count Variability: {self.metrics['num_detections_std']:.2f}")
        self.area_mean_mean_label.setText(f"Average Detection Area: {self.metrics['area_mean_mean']:.2f}")
        self.area_mean_std_label.setText(f"Detection Area Variability: {self.metrics['area_mean_std']:.2f}")
        self.area_std_mean_label.setText(f"Average Area Spread: {self.metrics['area_std_mean']:.2f}")
        self.area_std_std_label.setText(f"Area Spread Variability: {self.metrics['area_std_std']:.2f}")
        self.overlap_ratio_mean_label.setText(f"Average Overlap Ratio: {self.metrics['overlap_ratio_mean']:.2f}")
        self.overlap_ratio_std_label.setText(f"Overlap Ratio Variability: {self.metrics['overlap_ratio_std']:.2f}")
    
    def update_analysis_metrics_labels(self, metrics):
        # # Create a temporary directory
        # with tempfile.TemporaryDirectory() as temp_dir:
        #     # Copy the selected files to the temporary directory
        #     for file in self.uploaded_files:
        #         shutil.copy(file, temp_dir)
            
        #     # Use the temporary directory as the dataset path
        #     #TODO: get rid of evaluation tab instance
        #     self.analysis_metrics = DetectionQAMetrics(self.metrics.model_path, temp_dir)

        self.analysis_confidence_mean_mean_label.setText(f"Analysis Average Confidence Score: {metrics['confidence_mean_mean']:.2f}")
        self.analysis_confidence_mean_std_label.setText(f"Analysis Confidence Score Variability: {metrics['confidence_mean_std']:.2f}")
        self.analysis_confidence_std_mean_label.setText(f"Analysis Average Confidence Spread: {metrics['confidence_std_mean']:.2f}")
        self.analysis_confidence_std_std_label.setText(f"Analysis Confidence Spread Variability: {metrics['confidence_std_std']:.2f}")
        self.analysis_num_detections_mean_label.setText(f"Analysis Average Number of Detections: {metrics['num_detections_mean']:.2f}")
        self.analysis_num_detections_std_label.setText(f"Analysis Detection Count Variability: {metrics['num_detections_std']:.2f}")
        self.analysis_area_mean_mean_label.setText(f"Analysis Average Detection Area: {metrics['area_mean_mean']:.2f}")
        self.analysis_area_mean_std_label.setText(f"Analysis Detection Area Variability: {metrics['area_mean_std']:.2f}")
        self.analysis_area_std_mean_label.setText(f"Analysis Average Area Spread: {metrics['area_std_mean']:.2f}")
        self.analysis_area_std_std_label.setText(f"Analysis Area Spread Variability: {metrics['area_std_std']:.2f}")
        self.analysis_overlap_ratio_mean_label.setText(f"Analysis Average Overlap Ratio: {metrics['overlap_ratio_mean']:.2f}")
        self.analysis_overlap_ratio_std_label.setText(f"Analysis Overlap Ratio Variability: {metrics['overlap_ratio_std']:.2f}")

    def update(self):
        pass