from tqdm import tqdm
from tinydb import Query
from DeepNeuronSeg.utils.utils import save_label
from DeepNeuronSeg.models.segmentation_model import segment, composite_mask
from DeepNeuronSeg.models.image_manager import ImageManager
from PyQt5.QtWidgets import QMessageBox

class GenerateLabelsModel:
    def __init__(self, db):
        super().__init__()
        self.db = db
        self.image_manager = ImageManager(self.db)

    def generate_labels(self):
        
        query = Query()

        data_to_mask = self.db.image_table.search(~query["mask_data"].exists())

        for item in tqdm(data_to_mask, desc="Generating Labels", unit="image"):
            label = item.get("labels", [])
            file_path = item.get("file_path", "")

            if not label:
                # QMessageBox.warning(self, "No Labels", "No labels provided for image", file_path)
                print('No labels provided for image', file_path)
                continue

            mask_data = self.generate_label(file_path, label)
            self.db.image_table.update({"mask_data": mask_data}, query.file_path == file_path)


    def generate_label(self, image_path, labels):
        """Generate labels for the given image."""
        masks, scores = segment(image_path, labels)
        final_image, num_cells, instances_list = composite_mask(masks)

        final_image_path = save_label(final_image=final_image, image_path=image_path)

        return {
            "mask_path": final_image_path,
            "scores": scores[0],
            "num_cells": num_cells,
            "instances_list": instances_list
        }
