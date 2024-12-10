from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt5.QtCore import QPoint, pyqtSignal

class LabelingView(QWidget):

    add_cell_marker_signal = pyqtSignal(QPoint)
    remove_cell_marker_signal = pyqtSignal(QPoint, int)

    def __init__(self, image_display):
        super().__init__()
        self.image_display = image_display
        layout = QVBoxLayout()
        self.load_btn = QPushButton("Display Data")
        self.next_btn = QPushButton("Next Image")
        self.next_btn.clicked.connect(lambda: self.image_display.show_item(next_item=True, points=True))
        self.load_btn.clicked.connect(self.load_data)
    
        

        self.image_display.image_label.left_click_registered.connect(self.add_cell_marker)
        self.image_display.image_label.right_click_registered.connect(self.remove_cell_marker)
        
        layout.addWidget(self.image_display)
        layout.addWidget(self.next_btn)
        layout.addWidget(self.load_btn)

        self.setLayout(layout)
       
    def load_data(self):
        self.image_display.show_item(points=True)
        
    def add_cell_marker(self, pos):
        self.add_cell_marker_signal.emit(pos)

    def remove_cell_marker(self, pos, tolerance=5):
        self.remove_cell_marker_signal.emit(pos, tolerance)