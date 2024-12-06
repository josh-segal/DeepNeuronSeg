from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QComboBox, QGridLayout, QSpinBox, QLineEdit, QCheckBox, QLabel, QPushButton, QSizePolicy
)
from PyQt5.QtCore import Qt
from itertools import chain
from tinydb import Query
import os
from DeepNeuronSeg.models.denoise_model import DenoiseModel
from DeepNeuronSeg.views.widgets.hideable_input_panel import HideableInputPanel
from pathlib import Path


class TrainingTab(QWidget):
    def __init__(self, db):
        super().__init__()
        self.db = db
        layout = QVBoxLayout()
        
        # Model selection
        self.model = None
        self.model_selector = QComboBox()
        self.model_selector.addItems(["YOLOv8n-seg"])
        
        # Training parameters
        params_layout = QGridLayout()
        self.dataset = QComboBox()
        self.dataset.addItems(
            chain(
                *map(
                    lambda dataset: [dataset['dataset_name']] + 
                                    ([f"{dataset['dataset_name']} (denoised)"] if 'denoise_dataset_path' in dataset and dataset['denoise_dataset_path'] else []),
                    self.db.load_datasets()
                )
            )
        )

        self.default_augmentations = {
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'auto_augment': 'randaugment',
            'erasing': 0.4,
            'crop_fraction': 1.0
        }

        self.no_augmentations = {
            'hsv_h': 0.0,
            'hsv_s': 0.0,
            'hsv_v': 0.0,
            'degrees': 0.0,
            'translate': 0.0,
            'scale': 0.0,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.0,
            'mosaic': 0.0,
            'mixup': 0.0,
            'auto_augment': 'randaugment',
            'erasing': 0.0,
            'crop_fraction': 1.0
        }

        self.augmentations = self.default_augmentations.copy()
        self.augmentation_panel = HideableInputPanel(self.augmentations)

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 128)
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
        params_layout.addWidget(self.augmentation_panel, 7, 1)
        
        # Control buttons
        self.train_btn = QPushButton("Start Training")
        # self.stop_btn = QPushButton("Stop")

        self.train_btn.clicked.connect(self.trainer)
        
        label = QLabel("Base Model:")
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        label.setFixedHeight(10)
        layout.addWidget(label)
        layout.addWidget(self.model_selector)
        layout.addLayout(params_layout)
        layout.addWidget(self.train_btn)
        # layout.addWidget(self.stop_btn)
        self.setLayout(layout)

    def set_augmentations(self, checked):
        if checked:
            self.augmentations = self.default_augmentations.copy()
            self.augmentation_panel.update(self.augmentations)
            print(self.augmentations)
            # print('-'*50)
            # print(self.default_augmentations)
        else:
            self.augmentations = self.no_augmentations.copy()
            self.augmentation_panel.update(self.augmentations)
            print(self.augmentations)

    def trainer(self):
        if not self.model_name.text().strip():
            print("Model name required")
            return
        
        dataset = self.db.dataset_table.get(Query().dataset_name == self.dataset.currentText())
        dataset_path = dataset["dataset_path"]

        model_name_exists = self.db.model_table.contains(Query()["model_name"] == self.model_name.text().strip())
        if model_name_exists:
            print("Model name already exists, please choose a different name")
            return

        if self.denoise.isChecked():
            print("Training denoising network")

            #TODO: ALLOW USER TO TRAIN OWN MODEL OR USE PRETRAINED
            denoise_path = os.path.join(dataset_path, "denoise_model.pth")
            os.makedirs(denoise_path, exist_ok=True)

            dn_model = DenoiseModel(dataset_path=dataset_path, model_path=denoise_path)
            dn_model.unet_trainer(num_epochs=self.epochs.value(), batch_size=self.batch_size.value())
            dn_dataset_path = dn_model.create_dn_shuffle()

            dataset_data = Query()
            self.db.dataset_table.update(
                {"denoise_dataset_path": os.path.abspath(dn_dataset_path)}, 
                dataset_data.dataset_path == os.path.abspath(dataset_path)
            )

            print(f"Denoising images in {os.path.abspath(dataset_path)} and saving to {os.path.abspath(dn_dataset_path)}")

            dataset_path = os.path.abspath(dataset_path)

        elif self.denoise_base.isChecked():
            print("Using pretrained denoising network")
            denoise_path = (Path(__file__).resolve().parents[2] / "ml" / "denoise_model.pth").resolve()
            print('denoise path: ', denoise_path)
            # denoise_path = os.path.abspath("ml/denoise_model.pth")
            dn_model = DenoiseModel(dataset_path=dataset_path)
            dn_dataset_path = dn_model.create_dn_shuffle()

            dataset_data = Query()
            self.db.dataset_table.update(
                {"denoise_dataset_path": os.path.abspath(dn_dataset_path)}, 
                dataset_data.dataset_path == os.path.abspath(dataset_path)
            )

            print(f"Denoising images in {os.path.abspath(dataset_path)} and saving to {os.path.abspath(dn_dataset_path)}")

            dataset_path = os.path.abspath(dataset_path)
        else:
            denoise_path = None

        if self.model_selector.currentText() == "YOLOv8n-seg":
            # offset program load times by loading model here
            from ultralytics import YOLO
            print("Training YOLOv8n-seg")

            

            self.model = YOLO((Path(__file__).resolve().parents[2] / "ml" / "yolov8n-seg.pt").resolve())
            # self.model = YOLO("ml/yolov8n-seg.pt")
            self.model.train(
                #TODO: if denoised use denoised data dir, recreate yaml (?)
                data = os.path.abspath(f'{dataset_path}/data.yaml'),
                project = f'{dataset_path}/results',
                name = self.model_name.text().strip(),
                epochs = self.epochs.value(),
                patience = 0,
                batch = self.batch_size.value(),
                imgsz = 1024,
                hsv_h=augmentations['hsv_h'], 
                hsv_s=augmentations['hsv_s'], 
                hsv_v=augmentations['hsv_v'], 
                degrees=augmentations['degrees'], 
                translate=augmentations['translate'], 
                scale=augmentations['scale'], 
                shear=augmentations['shear'], 
                perspective=augmentations['perspective'], 
                flipud=augmentations['flipud'], 
                fliplr=augmentations['fliplr'], 
                mosaic=augmentations['mosaic'], 
                mixup=augmentations['mixup'],
                auto_augment=augmentations['auto_augment'],
                erasing=augmentations['erasing'],
                crop_fraction=augmentations['crop_fraction']
            )

        self.db.model_table.insert({
            "model_name": self.model_name.text().strip(),
            "model_path": f'{dataset_path}/results/{self.model_name.text().strip()}/weights/best.pt',
            "denoise_path": str(denoise_path)
        })

    def update(self):
        self.dataset.clear()
        self.dataset.addItems(
            chain(
                *map(
                    lambda dataset: [dataset['dataset_name']] + 
                                    ([f"{dataset['dataset_name']} (denoised)"] if 'denoise_dataset_path' in dataset and dataset['denoise_dataset_path'] else []),
                    self.db.load_datasets()
                )
            )
        )
