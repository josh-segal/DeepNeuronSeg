from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QComboBox, QGridLayout, QSpinBox, QLineEdit, QCheckBox, QLabel, QPushButton, QSizePolicy
)
from PyQt5.QtCore import pyqtSignal  
from DeepNeuronSeg.views.widgets.hideable_input_panel import HideableInputPanel

class TrainingView(QWidget):

    update_signal = pyqtSignal()
    set_augmentations_signal = pyqtSignal(bool)
    train_signal = pyqtSignal(str, str, str, bool, bool, int, int)

    def __init__(self, augmentations):
        super().__init__()
        layout = QVBoxLayout()
        
        # Model selection
        self.model = None
        self.model_selector = QComboBox()
        self.model_selector.addItems(["YOLOv8n-seg", "YOLOv8l-seg"])
        
        # Training parameters
        params_layout = QGridLayout()
        self.dataset = QComboBox()

        self.augmentation_panel = HideableInputPanel(augmentations)

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(70)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 128)
        self.batch_size.setValue(4)
        self.model_name = QLineEdit()
        self.denoise = QCheckBox("Train and Use Custom Denoising Network")
        self.denoise_base = QCheckBox("Use DeepNeuronSeg pretrained denoise model")
        self.use_augmentations = QCheckBox("Use Default Augmentations")
        self.use_augmentations.setChecked(True)
        self.use_augmentations.toggled.connect(lambda checked: self.set_augmentations(checked))

        self.use_augmentations.setChecked(True)
        self.use_augmentations.setToolTip("""
        Augmentations:
        --------------
        Augmentations are techniques applied to training data to increase its diversity and improve model generalization.

        Default Value:
        --------------
        Default to using DeepNeuronSeg default augmentations, leaving unchecked will use NO augmentations.

        Notes:
        -------
        Augmentations can help prevent overfitting and improve model robustness by exposing it to more varied data. 

        Excessive or unrealistic augmentations can harm performance, experiment with different parameters for your specific project.
        """)

        self.denoise.toggled.connect(lambda checked: self.denoise_base.setChecked(not checked) if checked else None)
        self.denoise_base.toggled.connect(lambda checked: self.denoise.setChecked(not checked) if checked else None)
        
        self.epochs_label = QLabel("Epochs:")
        self.dataset_label = QLabel("Dataset:")
        self.batch_size_label = QLabel("Batch Size:")
        self.model_name_label = QLabel("Trained Model Name:")
        self.epochs_label.setToolTip("""
        Epochs:
        -------
        Epochs refer to the number of complete passes through the entire training dataset 
        during the training process.

        Default Value:
        --------------
        The default number of epochs is set to 70.

        Notes:
        -------
        Too few epochs can lead to underfitting, while too many may result in 
        overfitting. 

        In general largers datasets require more epochs and smaller datasets require fewer epochs. 
        Watch validation metrics to determine if a model is underfitting or overfitting. 
        
        validation loss; if training loss is decreasing but validation loss is increasing or plateuing, the model is likely overfitting
        validation accurary; if training accuracy is increasing but validation accuracy is decreasing or plateuing, the model is likely overfitting

        if validation loss/accuracy and/or training loss/accuracy are not plateuing of overfitting, the model is likely underfitting.
        
        """)
        self.dataset_label.setToolTip("""
        Dataset:
        --------
        Dataset refers to the ID of the dataset you wish to train your model on.

        Default Value:
        --------------
        The default dataset ID is set to 1.

        Notes:
        -------
        Training on different datasets or shuffles of the same dataset can produce different model results.
        
        """)
        self.batch_size_label.setToolTip("""
        Batch Size:
        -----------
        Batch size refers to the number of images the model sees before updating the weights.

        Default Value:
        --------------
        The default batch size is set to 4.

        Notes:
        -------
        Larger batch sizes may speed up and stabilize training but require more memory. Smaller batch sizes update weights more frequently and may lead to better generalization.
        
        """)
        self.model_name_label.setToolTip("""
        Trained Model Name:
        --------------------
        The name of the trained model that will be saved after training.
        
        Default Value:
        --------------
        The default model name is set to 'model'.
        
        Notes:
        -------
        The model name is used to save the trained model after training. The model will be saved in the 'models' directory.
        
        """)
        self.denoise.setToolTip("""

        Denoising Network:
        -------------------
        Denoising network refers to the use of a UNet model to denoise the dataset before training the main model.

        Default Value:
        --------------
        The default value is set to False.

        Notes:
        -------
        Denoising the dataset can improve the quality of the training data and the performance of the model, but may increase training time and introduces additional preprocessing steps during training and inference.

        """)
        self.denoise_base.setToolTip("""
        Denoising Network:
        -------------------
        Denoising network refers to the use of a UNet model to denoise the dataset before training the main model.

        Default Value:
        --------------
        The default value is set to False.

        Notes:
        -------
        This setting uses a pretrained denoising model from the DeepNeuronSeg library. No training time required, results may be more or less effective than a custom trained denoise model.

        """)

        params_layout.addWidget(self.dataset_label, 0, 0)
        params_layout.addWidget(self.dataset, 0, 1)
        params_layout.addWidget(self.epochs_label, 1, 0)
        params_layout.addWidget(self.epochs, 1, 1)
        params_layout.addWidget(self.batch_size_label, 2, 0)
        params_layout.addWidget(self.batch_size, 2, 1)
        params_layout.addWidget(self.model_name_label, 3, 0)
        params_layout.addWidget(self.model_name, 3, 1)
        params_layout.addWidget(self.denoise, 4, 1)
        params_layout.addWidget(self.denoise_base, 5, 1)
        params_layout.addWidget(self.use_augmentations, 6, 1)
        
        # Control buttons
        self.train_btn = QPushButton("Start Training")

        self.train_btn.clicked.connect(self.trainer)
        
        label = QLabel("Base Model:")
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        label.setFixedHeight(10)
        layout.addWidget(label)
        layout.addWidget(self.model_selector)
        layout.addLayout(params_layout)
        layout.addWidget(self.augmentation_panel)
        layout.addWidget(self.train_btn)
        layout.addStretch()
        self.setLayout(layout)

    def trainer(self):
        base_model = self.model_selector.currentText()
        dataset_name = self.dataset.currentText()
        epochs = self.epochs.value()
        batch_size = self.batch_size.value()
        model_name = self.model_name.text()
        denoise = self.denoise.isChecked()
        denoise_base = self.denoise_base.isChecked()
        self.train_signal.emit(model_name, base_model, dataset_name, denoise, denoise_base, epochs, batch_size)

    def set_augmentations(self, checked):
        self.set_augmentations_signal.emit(checked)

    def update_augmentations(self, augmentations):
        self.augmentation_panel.update(augmentations)

    def update_dataset_selector(self, datasets):
        self.dataset.clear()
        self.dataset.addItems(datasets)

    def update(self):
        self.update_signal.emit()

    def update_response(self, datasets):
        self.dataset.clear()
        self.dataset.addItems(datasets)
