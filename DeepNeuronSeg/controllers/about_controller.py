from PyQt5.QtCore import QObject

class AboutController(QObject):

    def __init__(self, model, view):
        super().__init__()
        self.model = model
        self.view = view
        self._blinded = False

    def set_blinded(self, value):
        self._blinded = value
        self.update()

    def update(self):
        self.view.blind_checkbox.setChecked(self._blinded)
