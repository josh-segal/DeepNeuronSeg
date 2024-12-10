

class LabelingController:
    def __init__(self, model, view):
        self.model = model
        self.view = view

        self.view.add_cell_marker_signal.connect(self.add_cell_marker)
        self.view.remove_cell_marker_signal.connect(self.remove_cell_marker)

    def add_cell_marker(self, pos):
        self.model.add_cell_marker(pos)

    def remove_cell_marker(self, pos, tolerance=5):
        self.model.remove_cell_marker(pos, tolerance)