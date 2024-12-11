from PyQt5.QtCore import QObject

class GenerateLabelsController(QObject):
    def __init__(self, model, view):
        super().__init__()
        self.model = model
        self.view = view

        self.view.generate_labels_signal.connect(self.generate_labels)

    def generate_labels(self):
        self.model.generate_labels()
        self.view.display_labels()