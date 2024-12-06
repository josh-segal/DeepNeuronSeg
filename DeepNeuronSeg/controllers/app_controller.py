import os
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget

from DeepNeuronSeg.models.data_manager import DataManager

from DeepNeuronSeg.views.tabs.upload_tab import UploadTab
from DeepNeuronSeg.views.tabs.labeling_tab import LabelingTab
from DeepNeuronSeg.views.tabs.generate_labels_tab import GenerateLabelsTab
from DeepNeuronSeg.views.tabs.dataset_tab import DatasetTab
from DeepNeuronSeg.views.tabs.training_tab import TrainingTab
from DeepNeuronSeg.views.tabs.evaluation_tab import EvaluationTab
from DeepNeuronSeg.views.tabs.analysis_tab import AnalysisTab
from DeepNeuronSeg.views.tabs.outlier_tab import OutlierTab
from DeepNeuronSeg.views.tabs.model_zoo_tab import ModelZooTab

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
        
        self.analysis_tab = AnalysisTab(self.data_manager)
        self.outlier_tab = OutlierTab(self.data_manager)
        self.evaluation_tab = EvaluationTab(self.data_manager)

        self.analysis_tab.calculated_outlier_data.connect(self.outlier_tab.receive_outlier_data)
        self.evaluation_tab.calculated_dataset_metrics.connect(self.analysis_tab.receive_dataset_metrics)
        print("Connected signal to receive_dataset_metrics")
        
        # Create and add all self.tabs
        self.tabs.addTab(UploadTab(self.data_manager), "Upload Data")
        self.tabs.addTab(LabelingTab(self.data_manager), "Label Data")
        self.tabs.addTab(GenerateLabelsTab(self.data_manager), "Generate Labels")
        self.tabs.addTab(DatasetTab(self.data_manager), "Create Dataset")
        self.tabs.addTab(TrainingTab(self.data_manager), "Train Network")
        self.tabs.addTab(EvaluationTab(self.data_manager), "Evaluate Network")
        self.tabs.addTab(self.analysis_tab, "Analyze Data")
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