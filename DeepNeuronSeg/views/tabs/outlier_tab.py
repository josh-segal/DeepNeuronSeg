from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QDoubleSpinBox, QListWidget, QLabel
from DeepNeuronSeg.views.widgets.image_display import ImageDisplay

class OutlierTab(QWidget):
    def __init__(self, db):
        super().__init__()
        self.db = db
        layout = QVBoxLayout()
        
        
        # Image display
        self.image_display = ImageDisplay(self)
        
        # Outlier controls
        controls_layout = QHBoxLayout()
        self.confirm_btn = QPushButton("Confirm")
        self.relabel_btn = QPushButton("Relabel")
        self.skip_btn = QPushButton("Skip")
        controls_layout.addWidget(self.confirm_btn)
        controls_layout.addWidget(self.relabel_btn)
        controls_layout.addWidget(self.skip_btn)
        
        # Outlier list
        self.outlier_threshold = QDoubleSpinBox()
        self.outlier_threshold.setSingleStep(0.5)
        self.outlier_threshold.setValue(3) 

        self.outlier_list = QListWidget()
        self.outlier_list.itemClicked.connect(self.display_outlier_image)

        threshold_layout = QHBoxLayout()

        threshold_layout.addWidget(QLabel("Outlier Threshold:"))
        threshold_layout.addWidget(self.outlier_threshold)
        
        layout.addWidget(self.image_display)
        layout.addLayout(controls_layout)
        layout.addLayout(threshold_layout)
        self.setLayout(layout)
        
        """
        INTEGRATION POINT:
        1. Implement outlier detection
        2. Handle relabeling process
        3. Update dataset with confirmed/relabeled data
        """

    def display_outlier_image(self, item):
            image_path = item.text()
            self.image_display._display_image(image_path, self.outlier_list.row(item) + 1, self.outlier_list.count())
            # if relabel button clicked, add to db, calculate pseudo labels from masks and display labels for refining 
            # should remove prediction from pred table ? do I need pred table ?
            #TODO: should model data store db was trained on ?
    

    def receive_outlier_data(self, data):
        
        for file, score in data.items():
            if score > self.outlier_threshold.value():
                self.outlier_list.addItem(file)
                
        

    def update(self):
        pass