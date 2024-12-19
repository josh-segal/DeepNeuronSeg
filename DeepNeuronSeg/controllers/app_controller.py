import os
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget

from DeepNeuronSeg.models.data_manager import DataManager

from DeepNeuronSeg.models.upload_model import UploadModel
from DeepNeuronSeg.models.labeling_model import LabelingModel
from DeepNeuronSeg.models.generate_labels_model import GenerateLabelsModel
from DeepNeuronSeg.models.dataset_model import DatasetModel
from DeepNeuronSeg.models.training_model import TrainingModel
from DeepNeuronSeg.models.evaluation_model import EvaluationModel
from DeepNeuronSeg.models.analysis_model import AnalysisModel
from DeepNeuronSeg.models.outlier_model import OutlierModel
from DeepNeuronSeg.models.model_zoo_model import ModelZooModel

from DeepNeuronSeg.views.tabs.upload_view import UploadView
from DeepNeuronSeg.views.tabs.labeling_view import LabelingView
from DeepNeuronSeg.views.tabs.generate_labels_view import GenerateLabelsView
from DeepNeuronSeg.views.tabs.dataset_view import DatasetView
from DeepNeuronSeg.views.tabs.training_view import TrainingView
from DeepNeuronSeg.views.tabs.evaluation_view import EvaluationView
from DeepNeuronSeg.views.tabs.analysis_view import AnalysisView
from DeepNeuronSeg.views.tabs.outlier_view import OutlierView
from DeepNeuronSeg.views.tabs.model_zoo_view import ModelZooView

from DeepNeuronSeg.controllers.upload_controller import UploadController
from DeepNeuronSeg.controllers.labeling_controller import LabelingController
from DeepNeuronSeg.controllers.generate_labels_controller import GenerateLabelsController
from DeepNeuronSeg.controllers.dataset_controller import DatasetController
from DeepNeuronSeg.controllers.training_controller import TrainingController
from DeepNeuronSeg.controllers.evaluation_controller import EvaluationController
from DeepNeuronSeg.controllers.analysis_controller import AnalysisController
from DeepNeuronSeg.controllers.outlier_controller import OutlierController
from DeepNeuronSeg.controllers.model_zoo_controller import ModelZooController


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setup_data_dir()

        self.data_manager = DataManager()

        self.setWindowTitle("DeepNeuronSeg")
        self.setMinimumSize(1024, 768)
        self.resize(1600, 900)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setMovable(False)

        self.upload_model = UploadModel(self.data_manager)
        self.upload_view = UploadView()
        self.upload_controller = UploadController(self.upload_model, self.upload_view)
        
        self.labeling_model = LabelingModel(self.data_manager)
        self.labeling_view = LabelingView()
        self.labeling_controller = LabelingController(self.labeling_model, self.labeling_view)

        self.generate_labels_model = GenerateLabelsModel(self.data_manager)
        self.generate_labels_view = GenerateLabelsView()
        self.generate_labels_controller = GenerateLabelsController(self.generate_labels_model, self.generate_labels_view)

        self.dataset_model = DatasetModel(self.data_manager)
        self.dataset_view = DatasetView()
        self.dataset_controller = DatasetController(self.dataset_model, self.dataset_view)

        self.training_model = TrainingModel(self.data_manager)
        self.augmentations = self.training_model.augmentations.copy()        
        self.training_view = TrainingView(self.augmentations)
        self.training_controller = TrainingController(self.training_model, self.training_view)

        self.evaluation_model = EvaluationModel(self.data_manager)
        self.evaluation_view = EvaluationView()
        self.evaluation_controller = EvaluationController(self.evaluation_model, self.evaluation_view)
                
        self.analysis_model = AnalysisModel(self.data_manager)
        self.analysis_view = AnalysisView()
        self.analysis_controller = AnalysisController(self.analysis_model, self.analysis_view)

        self.outlier_view = OutlierView()
        self.outlier_model = OutlierModel(self.data_manager)
        self.outlier_controller = OutlierController(self.outlier_model, self.outlier_view)

        self.model_zoo_view = ModelZooView()
        self.model_zoo_model = ModelZooModel(self.data_manager)
        self.model_zoo_controller = ModelZooController(self.model_zoo_model, self.model_zoo_view)


        self.evaluation_model.update_metrics_labels_signal.connect(self.analysis_controller.receive_dataset_metrics)
        self.analysis_model.calculated_outlier_data.connect(self.outlier_controller.receive_outlier_data)
        self.upload_view.blinded_signal.connect(self.data_manager.set_blinded)
        
        # Create and add all self.tabs
        self.tabs.addTab(self.upload_view, "Upload Data")
        self.tabs.addTab(self.labeling_view, "Label Data")
        self.tabs.addTab(self.generate_labels_view, "Generate Labels")
        self.tabs.addTab(self.dataset_view, "Create Dataset")
        self.tabs.addTab(self.training_view, "Train Network")
        self.tabs.addTab(self.evaluation_view, "Evaluate Network")
        self.tabs.addTab(self.analysis_view, "Analyze Data")
        self.tabs.addTab(self.outlier_view, "Extract Outliers")
        self.tabs.addTab(self.model_zoo_view, "Model Zoo")
        
        layout.addWidget(self.tabs)
        self.tabs.currentChanged.connect(self.update_current_tab)

    def update_current_tab(self, index):
        current_tab = self.tabs.widget(index)
        if hasattr(current_tab, 'update') and callable(getattr(current_tab, 'update')):
            current_tab.update()

    def setup_data_dir(self):
        os.makedirs(os.path.join("data", "data_images"), exist_ok=True)