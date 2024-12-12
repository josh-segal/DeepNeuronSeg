from PyQt5.QtCore import QObject

class LabelingController(QObject):
    def __init__(self, model, view):
        super().__init__()
        self.model = model
        self.view = view

        self.view.add_cell_marker_signal.connect(self.add_cell_marker)
        self.view.upload_labels_signal.connect(self.upload_labels)
        self.view.remove_cell_marker_signal.connect(self.remove_cell_marker)

    def add_cell_marker(self, pos):
        self.model.add_cell_marker(pos)

    def remove_cell_marker(self, pos, tolerance=5):
        self.model.remove_cell_marker(pos, tolerance)

    def upload_labels(self, labels):
        self.model.parse_labels(labels)