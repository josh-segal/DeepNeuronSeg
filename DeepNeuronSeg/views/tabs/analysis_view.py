from DeepNeuronSeg.views.widgets.image_display import ImageDisplay
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QLabel, QComboBox, QPushButton, QFileDialog, QCheckBox, QListWidget
from PyQt5.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

class AnalysisView(QWidget):

    inference_images_signal = pyqtSignal(str, list)
    save_inferences_signal = pyqtSignal()
    download_data_signal = pyqtSignal()
    update_signal = pyqtSignal()
    display_graph_signal = pyqtSignal(bool)
    curr_image_signal = pyqtSignal()
    next_image_signal = pyqtSignal()
    load_image_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        metrics_layout = QGridLayout()
        self.image_display = ImageDisplay()
        # Model selection
        self.model_selector = QComboBox()

        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(lambda item: self.load_image(index=self.file_list.row(item)))
        
        # Image upload/selection
        self.select_btn = QPushButton("Select Images")
        self.inference_btn = QPushButton("Inference Images")
        self.save_btn = QPushButton("Save Inferences")
        self.download_btn = QPushButton("Download Data")  
        self.next_btn = QPushButton("Next Image")

        self.display_graph_checkbox = QCheckBox("Display Graph")
        self.display_graph_checkbox.toggled.connect(self.toggle_image_display_visibility)

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
        self.next_btn.clicked.connect(self.next_image)

        self.layout.addWidget(QLabel("Trained Model:"))
        self.layout.addWidget(self.model_selector)
        self.layout.addWidget(self.image_display)
        self.layout.addWidget(self.next_btn)
        self.image_display.hide()
        self.next_btn.hide()
        self.layout.addWidget(self.select_btn)
        self.layout.addWidget(self.inference_btn)
        self.layout.addWidget(self.display_graph_checkbox)
        self.layout.addWidget(self.save_btn)
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.file_list)
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

        self.layout.addLayout(metrics_layout)
        self.setLayout(self.layout)

    def next_image(self):
        self.next_image_signal.emit()

    def load_image(self, index):
        self.load_image_signal.emit(index)

    def toggle_image_display_visibility(self, checked):
        self.display_graph_signal.emit(checked)

    def handle_image_display(self):
        self.layout.removeWidget(self.canvas)
        self.canvas.hide()
        self.layout.insertWidget(4, self.image_display)
        self.image_display.show()
        self.layout.insertWidget(5, self.next_btn)
        self.next_btn.show()
        self.curr_image_signal.emit()

    def handle_graph_display(self, sorted_all_num_dets, sorted_all_conf_mean, colors):
        if sorted_all_num_dets is not None and sorted_all_conf_mean is not None and colors is not None:
            self.switch_to_graph_view(sorted_all_num_dets, sorted_all_conf_mean, colors)
        else:
            self.display_graph_checkbox.setChecked(False)
            self.clear_graph()
            print("No metrics to display, please calculate metrics first.")

    def switch_to_graph_view(self, sorted_all_num_dets, sorted_all_conf_mean, colors):
        self.image_display.clear()
        self.layout.removeWidget(self.image_display)
        self.image_display.hide()
        self.layout.removeWidget(self.next_btn)
        self.next_btn.hide()
        self.layout.insertWidget(4, self.canvas)
        self.canvas.show()
        self.update_graph(sorted_all_num_dets, sorted_all_conf_mean, colors)

    def inference_images(self):
        self.model_name = self.model_selector.currentText()
        self.inference_images_signal.emit(self.model_name, self.uploaded_files)
        # print("view inference_images_signal emitted")

    def save_inferences(self):
        self.save_inferences_signal.emit()

    def download_data(self):
        self.download_data_signal.emit()

    # TODO: select multiple models and compare results / ensemble ?
    def select_images(self):
        self.uploaded_files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png)")
        
    def update_graph(self, sorted_conf_mean, sorted_num_detections, colors):
        self.canvas.figure.clf()
        self.canvas.setMinimumSize(800, 400)
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

        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def clear_graph(self):
        self.canvas.figure.clf()
        self.canvas.draw()

    def update_dataset_metrics(self, metrics):
        #TODO: how to know what variance is acceptable, n-fold cross variation as baseline, how to calculate?
        self.confidence_mean_mean_label.setText(f"Average Confidence Score: {metrics['confidence_mean_mean']:.2f}")
        self.confidence_mean_std_label.setText(f"Confidence Score Variability: {metrics['confidence_mean_std']:.2f}")
        self.confidence_std_mean_label.setText(f"Average Confidence Spread: {metrics['confidence_std_mean']:.2f}")
        self.confidence_std_std_label.setText(f"Confidence Spread Variability: {metrics['confidence_std_std']:.2f}")
        self.num_detections_mean_label.setText(f"Average Number of Detections: {metrics['num_detections_mean']:.2f}")
        self.num_detections_std_label.setText(f"Detection Count Variability: {metrics['num_detections_std']:.2f}")
        self.area_mean_mean_label.setText(f"Average Detection Area: {metrics['area_mean_mean']:.2f}")
        self.area_mean_std_label.setText(f"Detection Area Variability: {metrics['area_mean_std']:.2f}")
        self.area_std_mean_label.setText(f"Average Area Spread: {metrics['area_std_mean']:.2f}")
        self.area_std_std_label.setText(f"Area Spread Variability: {metrics['area_std_std']:.2f}")
        self.overlap_ratio_mean_label.setText(f"Average Overlap Ratio: {metrics['overlap_ratio_mean']:.2f}")
        self.overlap_ratio_std_label.setText(f"Overlap Ratio Variability: {metrics['overlap_ratio_std']:.2f}")
    
    def update_analysis_metrics(self, metrics):
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
        self.update_signal.emit()

    def update_response(self, models, images):
        self.model_selector.clear()
        self.model_selector.addItems(models)

        self.file_list.clear()
        self.file_list.addItems(images)