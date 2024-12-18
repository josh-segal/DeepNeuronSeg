from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFileDialog, QListWidget
from PyQt5.QtCore import QPointF, pyqtSignal
from DeepNeuronSeg.views.widgets.image_display import ImageDisplay
class LabelingView(QWidget):

    add_cell_marker_signal = pyqtSignal(QPointF)
    remove_cell_marker_signal = pyqtSignal(QPointF, int)
    upload_labels_signal = pyqtSignal(list)
    load_image_signal = pyqtSignal(int)
    next_image_signal = pyqtSignal()
    update_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.image_display = ImageDisplay()
        layout = QVBoxLayout()
        self.upload_label_btn = QPushButton("Upload Labels")
        self.next_btn = QPushButton("Next Image")

        self.file_list = QListWidget()

        self.upload_label_btn.clicked.connect(self.upload_labels)
        self.next_btn.clicked.connect(self.next_image)

        self.image_display.image_label.left_click_registered.connect(self.add_cell_marker)
        self.image_display.image_label.right_click_registered.connect(self.remove_cell_marker)
        self.file_list.itemClicked.connect(lambda item: self.load_image(index=self.file_list.row(item)))
        
        layout.addWidget(self.upload_label_btn)
        layout.addWidget(self.image_display)
        layout.addWidget(self.file_list)
        layout.addWidget(self.next_btn)
        layout.addStretch()
        self.setLayout(layout)

    def upload_labels(self):
        uploaded_labels, _ = QFileDialog.getOpenFileNames(self, "Select Labels", "", "Labels (*.png *.txt *.csv *.xml)")
        self.upload_labels_signal.emit(uploaded_labels)
        
    def add_cell_marker(self, pos):
        self.add_cell_marker_signal.emit(pos)

    def remove_cell_marker(self, pos, tolerance=5):
        self.remove_cell_marker_signal.emit(pos, tolerance)

    def next_image(self):
        self.next_image_signal.emit()

    def load_image(self, index):
        self.load_image_signal.emit(index)

    def update(self):
        self.update_signal.emit()

    def update_response(self, images):
        self.file_list.clear()
        self.file_list.addItems(images)