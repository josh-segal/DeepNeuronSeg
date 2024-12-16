from tinydb import Query
from DeepNeuronSeg.views.widgets.image_display import ImageDisplay
from DeepNeuronSeg.utils.label_parsers import parse_png_label, parse_txt_label, parse_csv_label, parse_xml_label
from DeepNeuronSeg.utils.utils import trim_underscores
from DeepNeuronSeg.models.image_manager import ImageManager
import os
from PyQt5.QtCore import QObject


class LabelingModel(QObject):
    def __init__(self, db):
        super().__init__()
        self.db = db
        self.image_manager = ImageManager(self.db)

    def parse_labels(self, labels):

        for label_file in labels:
            label_name = os.path.splitext(os.path.basename(label_file))[0]
            label_name = trim_underscores(label_name)
            label_name = label_name + ".png"
            label_name = os.path.join('data', 'data_images', label_name)

            image_data = Query()
            matched_image = self.db.image_table.get(image_data.file_path == label_name)
            if matched_image:
                if label_file.endswith(".png"):
                    label_data = parse_png_label(label_file)
                elif label_file.endswith(".txt"):
                    label_data = parse_txt_label(label_file)
                elif label_file.endswith(".csv"):
                    label_data = parse_csv_label(label_file)
                elif label_file.endswith(".xml"):
                    label_data = parse_xml_label(label_file)
                else:
                    print("Invalid label format")
                    label_data = []

                if label_data:
                    self.db.image_table.update({"labels": label_data}, image_data.file_path == label_name)
            else:
                print(f"Image not found in database {label_name}")
                continue
        
    def add_cell_marker(self, pos):
        # print("adding cell")
        if not (0 <= pos.x() <= 512 and 0 <= pos.y() <= 512):
            return None, None, None, None

        # Get all records from the image_table
        images = self.image_manager.get_images()

        # Define file_path based on self.current_index
        file_path = images[self.image_manager.current_index] if 0 <= self.image_manager.current_index < len(images) else None

        image_query = Query()
        image_data = self.db.image_table.get(image_query.file_path == file_path)
        if image_data:
            self.db.image_table.update({"labels": image_data.get("labels", []) + [(pos.x(), pos.y())]}, image_query.file_path == file_path)
            item, index, total, points = self.image_manager.get_item(show_labels=True)
            return item, index, total, points
        else:
            print(f"Image not found in database {file_path}")
            return None, None, None, None

    def remove_cell_marker(self, pos, tolerance=5):
        if not (0 <= pos.x() <= 512 and 0 <= pos.y() <= 512):
            return None, None, None, None

        # Get all records from the image_table
        images = self.image_manager.get_images()

        # Define file_path based on self.current_index
        file_path = images[self.image_manager.current_index] if 0 <= self.image_manager.current_index < len(images) else None

        image_query = Query()
        image_data = self.db.image_table.get(image_query.file_path == file_path)
        if image_data:
            # Update labels: append the new position
            self.db.image_table.update({"labels": [label for label in image_data.get("labels", []) if not (abs(label[0] - pos.x()) < tolerance and abs(label[1] - pos.y()) < tolerance)]}, image_query.file_path == file_path)
            item, index, total, points = self.image_manager.get_item(show_labels=True)
            return item, index, total, points
        else:
            print(f"Image not found in database {file_path}")
            return None, None, None, None

    def load_image(self, index):
        self.image_manager.set_index(index)
        item, index, total, points = self.image_manager.get_item(show_labels=True)
        return item, index, total, points

    def next_image(self):
        self.image_manager.next_image()
        item, index, total, points = self.image_manager.get_item(show_labels=True)
        return item, index, total, points