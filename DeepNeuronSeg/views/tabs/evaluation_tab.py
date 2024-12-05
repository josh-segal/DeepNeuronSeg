from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QComboBox, QPushButton, QLabel
from PyQt5.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from itertools import chain
from tinydb import Query
from DeepNeuronSeg.models.qa_metrics import DetectionQAMetrics

class EvaluationTab(QWidget):

    calculated_dataset_metrics = pyqtSignal(object)

    def __init__(self, db):
        super().__init__()
        self.db = db
        layout = QVBoxLayout()
        metrics_layout = QGridLayout()
        self.metrics = None
        
        # Model selection
        self.model_selector = QComboBox()
        # model_dict = get_data(file_path='ml/model_metadata.json')
        # if model_dict:
        #     model_list = list(model_dict.keys())
        #     self.model_selector.addItems(model_list)

        self.model_selector.addItems(map(lambda model: model['model_name'], self.db.load_models()))

        self.dataset_selector = QComboBox()
        # dataset_dict = get_data(file_path='data/datasets/dataset_metadata.json')
        # if dataset_dict:
        #     dataset_list = list(dataset_dict.keys())
        #     self.dataset_selector.addItems(dataset_list)

        self.dataset_selector.addItems(
            chain(
                *map(
                    lambda dataset: [dataset['dataset_name']] + 
                                    ([f"{dataset['dataset_name']} (denoised)"] if 'denoise_dataset_path' in dataset and dataset['denoise_dataset_path'] else []),
                    self.db.load_datasets()
                )
            )
        )
        
        # Visualization area (placeholder for distribution plots)
        self.canvas = FigureCanvas(Figure(figsize=(12, 5)))

        self.calculate_metrics_btn = QPushButton("Calculate Metrics")
        self.calculate_metrics_btn.clicked.connect(self.calculate_metrics)

        self.downoad_data_btn = QPushButton("Download Data")
        self.downoad_data_btn.clicked.connect(self.download_data)

         # Labels for metrics
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

        layout.addWidget(QLabel("Trained Model:"))
        layout.addWidget(self.model_selector)
        layout.addWidget(QLabel("Dataset:"))
        layout.addWidget(self.dataset_selector)
        layout.addWidget(self.canvas)
        layout.addWidget(self.calculate_metrics_btn)
        layout.addWidget(self.downoad_data_btn)

        # Adding metric labels to layout
        metrics_layout.addWidget(self.confidence_mean_mean_label, 0, 0)
        metrics_layout.addWidget(self.confidence_mean_std_label, 0, 1)
        metrics_layout.addWidget(self.confidence_std_mean_label, 0, 2)
        metrics_layout.addWidget(self.confidence_std_std_label, 0, 3)
        metrics_layout.addWidget(self.num_detections_mean_label, 0, 4)
        metrics_layout.addWidget(self.num_detections_std_label, 0, 5)
        metrics_layout.addWidget(self.area_mean_mean_label, 1, 0)
        metrics_layout.addWidget(self.area_mean_std_label, 1, 1)
        metrics_layout.addWidget(self.area_std_mean_label, 1, 2)
        metrics_layout.addWidget(self.area_std_std_label, 1, 3)
        metrics_layout.addWidget(self.overlap_ratio_mean_label, 1, 4)
        metrics_layout.addWidget(self.overlap_ratio_std_label, 1, 5)

        layout.addLayout(metrics_layout)
        self.setLayout(layout)
    
        # check if metrics already calculated for model
        # load the dataset images
        # inference trained model on dataset images
        # calculate metrics / distributions across inferences
        # display metrics and distributions in meaningful way
        # in analyze data return quality score of inferenced image

    def calculate_metrics(self):
        # TODO: abstract
        self.model_name = self.model_selector.currentText()
        self.model_path = self.db.model_table.get(Query().model_name == self.model_name)
        self.model_path = self.model_path["model_path"]
        # print(self.model_path, '<----------------')
        self.dataset_name = self.dataset_selector.currentText()
        if " (denoised)" in self.dataset_name:
            dn_dataset_name = self.dataset_name.replace(" (denoised)", "")
            self.dataset_path = self.db.dataset_table.get(Query().dataset_name == dn_dataset_name).get('denoise_dataset_path')
        else:
            self.dataset_path = self.db.dataset_table.get(Query().dataset_name == self.dataset_name).get('dataset_path')
        # print(self.dataset_path, '<----------------')

        self.metrics = DetectionQAMetrics(self.model_path, self.dataset_path)

        self.calculated_dataset_metrics.emit(self.metrics)

        # print(self.metrics.dataset_metrics_mean_std)
        self.plot_metrics(self.metrics.dataset_metrics, self.metrics.dataset_metrics_mean_std)

    def plot_metrics(self, metrics, metrics_mean_std):
        self.canvas.figure.clf()
        ax1, ax2 = self.canvas.figure.subplots(1, 2)

        # Sort by num_detections and apply the same order to confidence_mean
        sorted_indices = sorted(range(len(metrics["num_detections"])), key=lambda i: metrics["num_detections"][i])

        sorted_num_detections = [metrics["num_detections"][i] for i in sorted_indices]
        sorted_conf_mean = [metrics["confidence_mean"][i] for i in sorted_indices]
        
        # Plotting histograms
        ax1.bar(range(len(sorted_conf_mean)), sorted_conf_mean, color='skyblue', edgecolor='black')
        ax1.set_title("Mean Confidence of Predictions Per Image")
        ax1.set_xlabel("Image")
        ax1.set_ylabel("Mean Confidence")

        ax2.bar(range(len(sorted_num_detections)), sorted_num_detections, color='salmon', edgecolor='black')
        ax2.set_title("Number of Detections Per Image")
        ax2.set_xlabel("Image")
        ax2.set_ylabel("Number of Detections")

        # Adjust layout and render
        self.canvas.figure.tight_layout()
        self.canvas.draw()

        self.update_metrics_labels(metrics_mean_std)

    def update_metrics_labels(self, metrics_mean_std):
        self.confidence_mean_mean_label.setText(f"Average Confidence Score: {metrics_mean_std['confidence_mean_mean']:.2f}")
        self.confidence_mean_std_label.setText(f"Confidence Score Variability: {metrics_mean_std['confidence_mean_std']:.2f}")
        self.confidence_std_mean_label.setText(f"Average Confidence Spread: {metrics_mean_std['confidence_std_mean']:.2f}")
        self.confidence_std_std_label.setText(f"Confidence Spread Variability: {metrics_mean_std['confidence_std_std']:.2f}")
        self.num_detections_mean_label.setText(f"Average Number of Detections: {metrics_mean_std['num_detections_mean']:.2f}")
        self.num_detections_std_label.setText(f"Detection Count Variability: {metrics_mean_std['num_detections_std']:.2f}")
        self.area_mean_mean_label.setText(f"Average Detection Area: {metrics_mean_std['area_mean_mean']:.2f}")
        self.area_mean_std_label.setText(f"Detection Area Variability: {metrics_mean_std['area_mean_std']:.2f}")
        self.area_std_mean_label.setText(f"Average Area Spread: {metrics_mean_std['area_std_mean']:.2f}")
        self.area_std_std_label.setText(f"Area Spread Variability: {metrics_mean_std['area_std_std']:.2f}")
        self.overlap_ratio_mean_label.setText(f"Average Overlap Ratio: {metrics_mean_std['overlap_ratio_mean']:.2f}")
        self.overlap_ratio_std_label.setText(f"Overlap Ratio Variability: {metrics_mean_std['overlap_ratio_std']:.2f}")

    def download_data(self):
        if self.metrics is not None:
            self.metrics.export_image_metrics_to_csv(filename=f'{self.dataset_name}_image_metrics.csv')
        else:
            print("No metrics to download, please calculate metrics first.")

    def update(self):
        self.model_selector.clear()
        self.model_selector.addItems(map(lambda model: model['model_name'], self.db.load_models()))

        self.dataset_selector.clear()
        self.dataset_selector.addItems(
            chain(
                *map(
                    lambda dataset: [dataset['dataset_name']] + 
                                    ([f"{dataset['dataset_name']} (denoised)"] if 'denoise_dataset_path' in dataset and dataset['denoise_dataset_path'] else []),
                    self.db.load_datasets()
                )
            )
        )