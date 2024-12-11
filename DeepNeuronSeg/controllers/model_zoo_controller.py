from PyQt5.QtCore import QObject

class ModelZooController(QObject):
    def __init__(self, model, view):
        super().__init__()
        self.model = model
        self.view = view