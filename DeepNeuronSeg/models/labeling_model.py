from tinydb import Query
from DeepNeuronSeg.views.widgets.image_display import ImageDisplay
from DeepNeuronSeg.utils.label_parsers import parse_png_label, parse_txt_label, parse_csv_label, parse_xml_label
from DeepNeuronSeg.utils.utils import trim_underscores
import os
from PyQt5.QtCore import QObject


class LabelingModel(QObject):
    def __init__(self, db):
        super().__init__()
        self.db = db
        self.current_index = 0
        self.uploaded_files = []

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
        adjusted_pos = self.image_display.image_label.adjust_pos(pos)
        if not (0 <= adjusted_pos.x() <= 512 and 0 <= adjusted_pos.y() <= 512):
            return

        # Get all records from the image_table
        images = self.db.load_images()

        # Define file_path based on self.current_index
        file_path = images[self.current_index]['file_path'] if 0 <= self.current_index < len(images) else None

        image_query = Query()
        image_data = self.db.image_table.get(image_query.file_path == file_path)
        if image_data:
            self.db.image_table.update({"labels": image_data.get("labels", []) + [(adjusted_pos.x(), adjusted_pos.y())]}, image_query.file_path == file_path)
            self.image_display.show_item(points=True)
            # self.image_display.show_image_with_points()

    def remove_cell_marker(self, pos, tolerance=5):
        adjusted_pos = self.image_display.image_label.adjust_pos(pos)
        if not (0 <= adjusted_pos.x() <= 512 and 0 <= adjusted_pos.y() <= 512):
            return

        # Get all records from the image_table
        images = self.db.load_images()

        # Define file_path based on self.current_index
        file_path = images[self.current_index]['file_path'] if 0 <= self.current_index < len(images) else None

        image_query = Query()
        image_data = self.db.image_table.get(image_query.file_path == file_path)
        if image_data:
            # Update labels: append the new position
            self.db.image_table.update({"labels": [label for label in image_data.get("labels", []) if not (abs(label[0] - adjusted_pos.x()) < tolerance and abs(label[1] - adjusted_pos.y()) < tolerance)]}, image_query.file_path == file_path)
            self.image_display.show_item(points=True)
            # self.image_display.show_image_with_points()