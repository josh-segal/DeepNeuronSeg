from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QComboBox, QPushButton, QLabel, QCheckBox, QListWidget
from PyQt5.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from DeepNeuronSeg.views.widgets.image_display import ImageDisplay

class EvaluationView(QWidget):

    curr_image_signal = pyqtSignal()
    next_image_signal = pyqtSignal()
    calculate_metrics_signal = pyqtSignal(str, str)
    display_graph_signal = pyqtSignal(bool)
    download_data_signal = pyqtSignal(str)
    update_signal = pyqtSignal()
    dataset_changed_signal = pyqtSignal(str)
    load_image_signal = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        metrics_layout = QGridLayout()
        self.metrics = None
        self.image_display = ImageDisplay()

        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(lambda item: self.load_image(index=self.file_list.row(item)))
        # Model selection
        self.model_selector = QComboBox()

        self.dataset_selector = QComboBox()
        self.dataset_selector.activated.connect(self.on_dataset_changed)
        
        # Visualization area (placeholder for distribution plots)
        self.canvas = FigureCanvas(Figure(figsize=(12, 5)))

        self.calculate_metrics_btn = QPushButton("Calculate Metrics")
        self.calculate_metrics_btn.clicked.connect(self.calculate_metrics)

        #TODO: display individual images when graph hidden (?)
        self.display_graph_checkbox = QCheckBox("Display Graph")
        self.display_graph_checkbox.toggled.connect(self.toggle_image_display_visibility)

        self.next_btn = QPushButton("Next Image")
        self.next_btn.clicked.connect(self.next_image)

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

        self.layout.addWidget(QLabel("Trained Model:"))
        self.layout.addWidget(self.model_selector)
        self.layout.addWidget(QLabel("Dataset:"))
        self.layout.addWidget(self.dataset_selector)
        self.layout.addWidget(self.image_display)
        self.layout.addWidget(self.next_btn)
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.calculate_metrics_btn)
        self.layout.addWidget(self.display_graph_checkbox)
        self.layout.addWidget(self.downoad_data_btn)
        self.layout.addWidget(self.file_list)
        # Adding metric labels to self.layout
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

        self.layout.addLayout(metrics_layout)
        self.setLayout(self.layout)
    
        # check if metrics already calculated for model
        # load the dataset images
        # inference trained model on dataset images
        # calculate metrics / distributions across inferences
        # display metrics and distributions in meaningful way
        # in analyze data return quality score of inferenced image

    def on_dataset_changed(self, dataset_index):
        dataset_name = self.dataset_selector.itemText(dataset_index)
        self.dataset_changed_signal.emit(dataset_name)

    def next_image(self):
        self.next_image_signal.emit()

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

    def handle_graph_display(self, sorted_num_dets, sorted_conf_mean):
        if sorted_num_dets is not None and sorted_conf_mean is not None:
            self.switch_to_graph_view(sorted_num_dets, sorted_conf_mean)
        else:
            self.display_graph_checkbox.setChecked(False)
            self.clear_graph()
            print("No metrics to display, please calculate metrics first.")

    def switch_to_graph_view(self, sorted_num_dets, sorted_conf_mean):
        self.image_display.clear()
        self.layout.removeWidget(self.image_display)
        self.image_display.hide()
        self.layout.removeWidget(self.next_btn)
        self.next_btn.hide()
        self.layout.insertWidget(4, self.canvas)
        self.canvas.show()
        self.update_graph(sorted_num_dets, sorted_conf_mean)

    def update_graph(self, sorted_num_dets, sorted_conf_mean):
        self.canvas.figure.clf()
        self.canvas.setMinimumSize(800, 400)
        ax1, ax2 = self.canvas.figure.subplots(1, 2)

        ax1.bar(range(len(sorted_conf_mean)), sorted_conf_mean, color='skyblue', edgecolor='black')
        ax1.set_title("Mean Confidence of Predictions Per Image")
        ax1.set_xlabel("Image")
        ax1.set_ylabel("Mean Confidence")

        ax2.bar(range(len(sorted_num_dets)), sorted_num_dets, color='salmon', edgecolor='black')
        ax2.set_title("Number of Detections Per Image")
        ax2.set_xlabel("Image")
        ax2.set_ylabel("Number of Detections")

        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def clear_graph(self):
        self.canvas.figure.clf()
        self.canvas.draw()

    def calculate_metrics(self):
        model_name = self.model_selector.currentText()
        dataset_name = self.dataset_selector.currentText()

        self.calculate_metrics_signal.emit(model_name, dataset_name)

    def update_metrics_labels(self, metrics):
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

    def download_data(self):
        dataset_name = self.dataset_selector.currentText()
        self.download_data_signal.emit(dataset_name)

    def update(self):
        self.update_signal.emit()

    def load_image(self, index):
        self.load_image_signal.emit(index)

    def update_response(self, models, datasets, images):
        self.model_selector.clear()
        self.dataset_selector.clear()
        
        self.model_selector.addItems(models)
        self.dataset_selector.addItems(datasets)

        self.file_list.clear()
        self.file_list.addItems(images)