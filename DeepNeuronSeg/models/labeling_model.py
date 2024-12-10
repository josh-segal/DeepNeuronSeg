from tinydb import Query
from DeepNeuronSeg.views.widgets.image_display import ImageDisplay



class LabelingModel:
    def __init__(self, db):
        super().__init__()
        self.db = db
        self.current_index = 0
        self.uploaded_files = []
        
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