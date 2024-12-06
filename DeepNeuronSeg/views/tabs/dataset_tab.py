import os
import shutil
import random
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QDoubleSpinBox, QLineEdit, QCheckBox, QLabel, QListWidget, QPushButton
from tinydb import Query
from DeepNeuronSeg.utils.utils import get_image_mask_label_tuples, create_yaml

class DatasetTab(QWidget):
    def __init__(self, db):
        super().__init__()
        self.db = db
        layout = QVBoxLayout()
        
        # Dataset configuration
        config_layout = QGridLayout()
        self.train_split = QDoubleSpinBox()
        self.train_split.setRange(0.0, 1.0)
        self.train_split.setSingleStep(0.05)
        self.train_split.setValue(0.8)
        self.dataset_name = QLineEdit()
        
        # Augmentation options
        # self.flip_horizontal = QCheckBox("Horizontal Flip")
        # self.flip_vertical = QCheckBox("Vertical Flip")
        # self.enable_rotation = QCheckBox("Enable Rotation")
        # self.enable_crop = QCheckBox("Random Crop")
        
        config_layout.addWidget(QLabel("Train Split:"), 0, 0)
        config_layout.addWidget(self.train_split, 0, 1)
        # config_layout.addWidget(self.flip_horizontal, 1, 0)
        # config_layout.addWidget(self.flip_vertical, 1, 1)
        # config_layout.addWidget(self.enable_rotation, 2, 0)
        # config_layout.addWidget(self.enable_crop, 2, 1)
        config_layout.addWidget(QLabel("Dataset Name:"), 3, 0)
        config_layout.addWidget(self.dataset_name, 3, 1)
        
        # Image selection
        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QListWidget.MultiSelection)
        items = self.db.load_images()
        self.image_list.addItems([os.path.basename(file) for file in items])
        
        # Creation button
        self.create_btn = QPushButton("Create Dataset")
        self.create_btn.clicked.connect(self.create_dataset)
        
        layout.addLayout(config_layout)
        layout.addWidget(self.image_list)
        layout.addWidget(self.create_btn)
        self.setLayout(layout)
    
    def create_dataset(self):
        """
        INTEGRATION POINT:
        1. Get selected images
        2. Apply augmentation based on settings
        3. Create train/test split
        4. Save dataset configuration
        """
        selected_images = [os.path.join("data", "data_images", item.text()) for item in  self.image_list.selectedItems()]
        # print(selected_images)
        selected_masks = [
            record["mask_data"]["mask_path"]
            for record in self.db.image_table.search(Query().file_path.one_of(selected_images))
            if "mask_data" in record
        ]

        selected_labels = [
            record["mask_data"]["instances_list"]
            for record in self.db.image_table.search(Query().file_path.one_of(selected_images))
            if "mask_data" in record
        ]   

        dataset_parent_dir = os.path.join('data', 'datasets')
        os.makedirs(dataset_parent_dir, exist_ok=True)

        # if not os.path.exists(os.path.join(dataset_parent_dir, 'dataset_metadata.json')):
        #     with open(os.path.join(dataset_parent_dir, 'dataset_metadata.json'), 'w') as f:
        #         json.dump({}, f)

        dataset_dir = 'dataset'
        counter = 0
        #TODO: possibly broken for module logic ?
        self.dataset_path = os.path.abspath(os.path.join(dataset_parent_dir, f"{dataset_dir}_{counter}"))
        while os.path.exists(self.dataset_path):
            counter += 1
            self.dataset_path = os.path.abspath(os.path.join(dataset_parent_dir, f"{dataset_dir}_{counter}"))

        # dataset_metadata = get_data(file_path=os.path.join(dataset_parent_dir, 'dataset_metadata.json'))

        if not self.dataset_name.text().strip():
            self.dataset_name.setText(f"{self.dataset_path}")
            print("Dataset name not provided, using default")
        if self.db.dataset_table.get(Query().dataset_name == self.dataset_name.text().strip()):
            print("Dataset name already exists, please choose a different name")
            return

        self.db.dataset_table.insert({
            "dataset_name": self.dataset_name.text().strip(),
            "dataset_path": self.dataset_path
        })

        # set_data(file_path=os.path.join(dataset_parent_dir, 'dataset_metadata.json'), metadata=dataset_metadata)

        os.makedirs(self.dataset_path, exist_ok=False)
        os.makedirs(os.path.join(self.dataset_path, "images"), exist_ok=False)
        os.makedirs(os.path.join(self.dataset_path, "masks"), exist_ok=False)
        os.makedirs(os.path.join(self.dataset_path, "labels"), exist_ok=False)

        for image, mask, labels in zip(selected_images, selected_masks, selected_labels):
            # print(labels,"\n", "\n")
            image_name = os.path.basename(image)
            mask_name = os.path.basename(mask)
            
            image_path = os.path.join(self.dataset_path, "images", image_name)
            mask_path = os.path.join(self.dataset_path, "masks", mask_name)
            label_path = os.path.join(self.dataset_path, "labels", f"{os.path.splitext(image_name)[0]}.txt")

            shutil.copy(image, image_path)
            shutil.copy(mask, mask_path)

            with open(label_path, "w") as f:
                for label in labels:
                    # print(label, "\n")
                    label_seg = label["segmentation"]
                    normalized_label = [format(float(coord) / 512 if i % 2 == 0 else float(coord) / 512, ".6f") for i, coord in enumerate(label_seg)]
                    # print("label", label_seg)
                    # print("normalized label", normalized_label)
                    f.write(f"0 " + " ".join(normalized_label) + "\n")

        self.create_shuffle()

    def create_shuffle(self):
        image_paths, mask_paths, label_paths = get_image_mask_label_tuples(self.dataset_path)

        #TODO: would train test split be more appropriate here?
        combined = list(zip(image_paths, mask_paths, label_paths))
        random.shuffle(combined)
        shuffled_image_paths, shuffled_mask_paths, shuffled_label_paths = zip(*combined)

        split_index = int(len(shuffled_image_paths) * self.train_split.value())

        train_images = shuffled_image_paths[:split_index]
        val_images = shuffled_image_paths[split_index:]
        train_masks = shuffled_mask_paths[:split_index]
        val_masks = shuffled_mask_paths[split_index:]
        train_labels = shuffled_label_paths[:split_index]
        val_labels = shuffled_label_paths[split_index:]

        # counter = 0
        # shuffle_path = os.path.join(self.dataset_path, f"shuffle_{counter}")
        # while os.path.exists(shuffle_path):
        #     counter += 1
        #     shuffle_path = os.path.join(self.dataset_path, f"shuffle_{counter}")

        # os.makedirs(shuffle_path, exist_ok=False)

        train_images_dir = os.path.join( "train", "images")
        val_images_dir = os.path.join( "val", "images")

        create_yaml(os.path.join(self.dataset_path, "data.yaml"), train_images_dir, val_images_dir)

        os.makedirs(os.path.join(self.dataset_path, train_images_dir), exist_ok=False)
        os.makedirs(os.path.join(self.dataset_path, "train", "masks"), exist_ok=False)
        os.makedirs(os.path.join(self.dataset_path, "train", "labels"), exist_ok=False)

        os.makedirs(os.path.join(self.dataset_path, val_images_dir), exist_ok=False)
        os.makedirs(os.path.join(self.dataset_path, "val", "masks"), exist_ok=False)
        os.makedirs(os.path.join(self.dataset_path, "val", "labels"), exist_ok=False)

        for image, mask, label in zip(train_images, train_masks, train_labels):
            shutil.copy(image, os.path.join(self.dataset_path, "train", "images", os.path.basename(image)))
            shutil.copy(mask, os.path.join(self.dataset_path, "train", "masks", os.path.basename(mask)))
            shutil.copy(label, os.path.join(self.dataset_path, "train", "labels", os.path.basename(label)))

        for image, mask, label in zip(val_images, val_masks, val_labels):
            shutil.copy(image, os.path.join(self.dataset_path, "val", "images", os.path.basename(image)))
            shutil.copy(mask, os.path.join(self.dataset_path, "val", "masks", os.path.basename(mask)))
            shutil.copy(label, os.path.join(self.dataset_path, "val", "labels", os.path.basename(label)))

    def update(self):
        self.image_list.clear()
        items = self.db.load_images()
        self.image_list.addItems([os.path.basename(file) for file in items])