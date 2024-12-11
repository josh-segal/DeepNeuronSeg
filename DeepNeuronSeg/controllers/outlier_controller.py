from PyQt5.QtCore import QObject

class OutlierController(QObject):
    def __init__(self, model, view):
        super().__init__()
        self.model = model
        self.view = view