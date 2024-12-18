import os
import shutil
import random
from tinydb import Query
from DeepNeuronSeg.utils.utils import get_image_mask_label_tuples, create_yaml
from PyQt5.QtCore import QObject

class DatasetModel:
    def __init__(self, db):
        super().__init__()
        self.db = db

    def load_images(self):
        return self.db.load_images()
    
    def create_dataset(self, images, dataset_name, train_split):
        selected_images = [os.path.join("data", "data_images", item.text()) for item in images]

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

        dataset_dir = 'dataset'
        counter = 0
        dataset_path = os.path.abspath(os.path.join(dataset_parent_dir, f"{dataset_dir}_{counter}"))
        while os.path.exists(dataset_path):
            counter += 1
            dataset_path = os.path.abspath(os.path.join(dataset_parent_dir, f"{dataset_dir}_{counter}"))

        if not dataset_name:
            print("Please enter a dataset name")
            return
        if self.db.dataset_table.get(Query().dataset_name == dataset_name):
            print("Dataset name already exists, please choose a different name")
            return

        self.db.dataset_table.insert({
            "dataset_name": dataset_name,
            "dataset_path": dataset_path
        })

        # set_data(file_path=os.path.join(dataset_parent_dir, 'dataset_metadata.json'), metadata=dataset_metadata)

        os.makedirs(dataset_path, exist_ok=False)
        os.makedirs(os.path.join(dataset_path, "images"), exist_ok=False)
        os.makedirs(os.path.join(dataset_path, "masks"), exist_ok=False)
        os.makedirs(os.path.join(dataset_path, "labels"), exist_ok=False)

        for image, mask, labels in zip(selected_images, selected_masks, selected_labels):
            # print(labels,"\n", "\n")
            image_name = os.path.basename(image)
            mask_name = os.path.basename(mask)
            
            image_path = os.path.join(dataset_path, "images", image_name)
            mask_path = os.path.join(dataset_path, "masks", mask_name)
            label_path = os.path.join(dataset_path, "labels", f"{os.path.splitext(image_name)[0]}.txt")

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

        self.create_shuffle(dataset_path, train_split)

    def create_shuffle(self, dataset_path, train_split):
        image_paths, mask_paths, label_paths = get_image_mask_label_tuples(dataset_path)

        combined = list(zip(image_paths, mask_paths, label_paths))
        random.shuffle(combined)
        shuffled_image_paths, shuffled_mask_paths, shuffled_label_paths = zip(*combined)

        split_index = int(len(shuffled_image_paths) * train_split)

        train_images = shuffled_image_paths[:split_index]
        val_images = shuffled_image_paths[split_index:]
        train_masks = shuffled_mask_paths[:split_index]
        val_masks = shuffled_mask_paths[split_index:]
        train_labels = shuffled_label_paths[:split_index]
        val_labels = shuffled_label_paths[split_index:]

        train_images_dir = os.path.join( "train", "images")
        val_images_dir = os.path.join( "val", "images")

        create_yaml(os.path.join(dataset_path, "data.yaml"), train_images_dir, val_images_dir)

        os.makedirs(os.path.join(dataset_path, train_images_dir), exist_ok=False)
        os.makedirs(os.path.join(dataset_path, "train", "masks"), exist_ok=False)
        os.makedirs(os.path.join(dataset_path, "train", "labels"), exist_ok=False)

        os.makedirs(os.path.join(dataset_path, val_images_dir), exist_ok=False)
        os.makedirs(os.path.join(dataset_path, "val", "masks"), exist_ok=False)
        os.makedirs(os.path.join(dataset_path, "val", "labels"), exist_ok=False)

        for image, mask, label in zip(train_images, train_masks, train_labels):
            shutil.copy(image, os.path.join(dataset_path, "train", "images", os.path.basename(image)))
            shutil.copy(mask, os.path.join(dataset_path, "train", "masks", os.path.basename(mask)))
            shutil.copy(label, os.path.join(dataset_path, "train", "labels", os.path.basename(label)))

        for image, mask, label in zip(val_images, val_masks, val_labels):
            shutil.copy(image, os.path.join(dataset_path, "val", "images", os.path.basename(image)))
            shutil.copy(mask, os.path.join(dataset_path, "val", "masks", os.path.basename(mask)))
            shutil.copy(label, os.path.join(dataset_path, "val", "labels", os.path.basename(label)))