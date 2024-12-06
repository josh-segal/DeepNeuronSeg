from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from tqdm import tqdm
from tinydb import Query
from DeepNeuronSeg.views.widgets.image_display import ImageDisplay
from DeepNeuronSeg.utils.utils import save_label
from DeepNeuronSeg.models.segmentation_model import segment, composite_mask


class GenerateLabelsTab(QWidget):
    def __init__(self, db):
        super().__init__()
        self.db = db
        layout = QVBoxLayout()
        image_layout = QHBoxLayout()
        config_layout = QHBoxLayout()

        self.current_index = 0
        self.uploaded_files = []
        self.metadata_labels = []
        self.data = []

        self.left_image = ImageDisplay(self)
        self.right_image = ImageDisplay(self)

        self.generate_btn = QPushButton("Generate Labels")
        self.next_btn = QPushButton("Next Image")
        self.display_btn = QPushButton("Display Labels")
        self.generate_btn.clicked.connect(self.generate_labels)
        self.next_btn.clicked.connect(lambda: self.left_image.show_item(next_item=True))
        self.next_btn.clicked.connect(lambda: self.right_image.show_item(mask=True))
        self.display_btn.clicked.connect(self.display_labels)

        
        config_layout.addWidget(self.generate_btn)
        config_layout.addWidget(self.next_btn)
        config_layout.addWidget(self.display_btn)

        
        image_layout.addWidget(self.left_image)
        image_layout.addWidget(self.right_image)

        layout.addLayout(image_layout)
        layout.addLayout(config_layout)

        self.setLayout(layout)

    def generate_labels(self):
        # print(result)
        # print(self.uploaded_files)
        # print(self.labels)
        # print(len(result))

        query = Query()

        data_to_mask = self.db.image_table.search(~query["mask_data"].exists())

        # self.uploaded_files = self.db.load_images()
        # self.labels = self.db.load_labels()

        for item in tqdm(data_to_mask, desc="Generating Labels", unit="image"):
            label = item.get("labels", [])
            file_path = item.get("file_path", "")

            if not label:
                print("No labels provided for image", file_path)
                continue

            mask_data = self.generate_label(file_path, label)
            self.db.image_table.update({"mask_data": mask_data}, query.file_path == file_path)


        self.display_labels()
        # for i, (uploaded_file, label) in enumerate(tqdm(zip(self.uploaded_files, self.labels), total=len(self.uploaded_files), desc="Generating Labels", unit="image")):
        #     # print(i)
        #     if label is None:
        #         print("No labels provided for image", uploaded_file)
        #         continue

        #     # add TQDM progress bar before images are shown
        #     label_data = self.generate_label(uploaded_file, label)
        #     for image in self.data:
        #         if image["file_path"] == uploaded_file:
        #             image["mask_data"] = label_data
        #             break

        #     # save somewhere somehow in relation to uploaded_file
        #     set_data(metadata=self.data)
        # display image label pairs with button to see next pair
        
        # allow user editing of generated labels

    def generate_label(self, image_path, labels):
        """Generate labels for the given image."""
        masks, scores = segment(image_path, labels)
        final_image, num_cells, instances_list = composite_mask(masks)

        final_image_path = save_label(final_image=final_image, image_path=image_path)
        # print(scores)
        # print(scores[0])
        # save final_image to labeled_data folder
        # print("final_image_path", final_image_path)
        # print("scores", scores_numpy)
        # print("num_cells", num_cells)
        # print("instances_list", instances_list)
        return {
            "mask_path": final_image_path,
            "scores": scores[0],
            "num_cells": num_cells,
            "instances_list": instances_list
        }

    def display_labels(self):
        self.left_image.show_item()
        self.right_image.show_item(mask=True)

    def update(self):
        pass
