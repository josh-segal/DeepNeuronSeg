from PyQt5.QtCore import QObject

class LabelingController(QObject):
    def __init__(self, model, view):
        super().__init__()
        self.model = model
        self.view = view

        self.view.add_cell_marker_signal.connect(self.add_cell_marker)
        self.view.upload_labels_signal.connect(self.upload_labels)
        self.view.remove_cell_marker_signal.connect(self.remove_cell_marker)
        self.view.load_image_signal.connect(self.load_image)
        self.view.next_image_signal.connect(self.next_image)
        self.view.update_signal.connect(self.update_view)

    def add_cell_marker(self, pos):
        adjusted_pos = self.view.image_display.image_label.adjust_pos(pos)
        item, index, total, points = self.model.add_cell_marker(adjusted_pos)
        if item:
            self.view.image_display.display_frame(item, index, total, points)

    def remove_cell_marker(self, pos, tolerance=5):
        adjusted_pos = self.view.image_display.image_label.adjust_pos(pos)
        item, index, total, points = self.model.remove_cell_marker(adjusted_pos, tolerance)
        if item:
            self.view.image_display.display_frame(item, index, total, points)

    def upload_labels(self, labels):
        self.model.parse_labels(labels)

    def load_image(self, index):
        self.model.image_manager.set_index(index)
        item, index, total, points = self.model.image_manager.get_item(show_labels=True)
        if item:
            self.view.image_display.display_frame(item, index, total, points)

    def next_image(self):
        self.model.image_manager.next_image()
        item, index, total, points = self.model.image_manager.get_item(show_labels=True)
        if item:
            self.view.image_display.display_frame(item, index, total, points)

    def update_view(self):
        images = self.model.image_manager.get_images()
        item, index, total, points = self.model.image_manager.get_item(show_labels=True)
        if item:
            self.view.update_response(images)
            self.view.image_display.display_frame(item, index, total, points)