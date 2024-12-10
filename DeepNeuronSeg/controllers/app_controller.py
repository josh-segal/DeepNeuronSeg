import os
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget

from DeepNeuronSeg.views.widgets.image_display import ImageDisplay

from DeepNeuronSeg.models.data_manager import DataManager

from DeepNeuronSeg.models.upload_model import UploadModel
from DeepNeuronSeg.models.labeling_model import LabelingModel
from DeepNeuronSeg.models.generate_labels_model import GenerateLabelsModel
from DeepNeuronSeg.models.dataset_model import DatasetModel
from DeepNeuronSeg.models.analysis_model import AnalysisModel

from DeepNeuronSeg.views.tabs.upload_view import UploadView
from DeepNeuronSeg.views.tabs.labeling_view import LabelingView
from DeepNeuronSeg.views.tabs.generate_labels_view import GenerateLabelsView
from DeepNeuronSeg.views.tabs.dataset_view import DatasetView
from DeepNeuronSeg.views.tabs.evaluation_tab import EvaluationTab
from DeepNeuronSeg.views.tabs.analysis_view import AnalysisView
from DeepNeuronSeg.views.tabs.outlier_tab import OutlierTab
from DeepNeuronSeg.views.tabs.model_zoo_tab import ModelZooTab

from DeepNeuronSeg.controllers.upload_controller import UploadController
from DeepNeuronSeg.controllers.labeling_controller import LabelingController
from DeepNeuronSeg.controllers.generate_labels_controller import GenerateLabelsController
from DeepNeuronSeg.controllers.dataset_controller import DatasetController
from DeepNeuronSeg.controllers.analysis_controller import AnalysisController

from DeepNeuronSeg.views.tabs.training_tab import TrainingTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setup_data_dir()

        self.data_manager = DataManager()

        self.setWindowTitle("DeepNeuronSeg")
        self.setMinimumSize(1024, 768)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setMovable(False)

        self.upload_model = UploadModel(self.data_manager)
        self.upload_image_displayer = ImageDisplay(self.upload_model)
        self.upload_view = UploadView(self.upload_image_displayer)
        self.upload_controller = UploadController(self.upload_model, self.upload_view)
        
        self.labeling_model = LabelingModel(self.data_manager)
        self.labeling_image_displayer = ImageDisplay(self.labeling_model)
        self.labeling_view = LabelingView(self.labeling_image_displayer)
        self.labeling_controller = LabelingController(self.labeling_model, self.labeling_view)

        self.generate_labels_model = GenerateLabelsModel(self.data_manager)
        self.generate_labels_image_displayer_left = ImageDisplay(self.generate_labels_model)
        self.generate_labels_image_displayer_right = ImageDisplay(self.generate_labels_model)
        self.generate_labels_view = GenerateLabelsView(self.generate_labels_image_displayer_left, self.generate_labels_image_displayer_right)
        self.generate_labels_controller = GenerateLabelsController(self.generate_labels_model, self.generate_labels_view)

        self.dataset_model = DatasetModel(self.data_manager)
        self.dataset_view = DatasetView()
        self.dataset_controller = DatasetController(self.dataset_model, self.dataset_view)
        
        
        
        self.analysis_view = AnalysisView()
        self.analysis_model = AnalysisModel(self.data_manager)
        self.analysis_controller = AnalysisController(self.analysis_model, self.analysis_view)

        self.outlier_tab = OutlierTab(self.data_manager)
        self.evaluation_tab = EvaluationTab(self.data_manager)

        # self.analysis_view.calculated_outlier_data.connect(self.outlier_tab.receive_outlier_data)
        self.evaluation_tab.calculated_dataset_metrics.connect(self.analysis_controller.receive_dataset_metrics)
        # print("Connected signal to receive_dataset_metrics")
        
        # Create and add all self.tabs
        self.tabs.addTab(self.upload_view, "Upload Data")
        self.tabs.addTab(self.labeling_view, "Label Data")
        self.tabs.addTab(self.generate_labels_view, "Generate Labels")
        self.tabs.addTab(self.dataset_view, "Create Dataset")
        self.tabs.addTab(TrainingTab(self.data_manager), "Train Network")
        self.tabs.addTab(EvaluationTab(self.data_manager), "Evaluate Network")
        self.tabs.addTab(self.analysis_view, "Analyze Data")
        self.tabs.addTab(self.outlier_tab, "Extract Outliers")
        self.tabs.addTab(ModelZooTab(self.data_manager), "Model Zoo")
        
        layout.addWidget(self.tabs)
        self.tabs.currentChanged.connect(self.update_current_tab)

    def update_current_tab(self, index):
        current_tab = self.tabs.widget(index)
        if hasattr(current_tab, 'update') and callable(getattr(current_tab, 'update')):
            current_tab.update()

    def setup_data_dir(self):
        os.makedirs(os.path.join("data", "data_images"), exist_ok=True)