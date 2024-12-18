from PyQt5.QtCore import QObject

class OutlierModel(QObject):
    def __init__(self, db):
        super().__init__()
        self.db = db
        self.outlier_threshold = 3

    def display_outlier_image(self, item):
            image_path = item.text()
            self.image_display._display_image(image_path, self.outlier_list.row(item) + 1, self.outlier_list.count())
            # if relabel button clicked, add to db, calculate pseudo labels from masks and display labels for refining 
            # should remove prediction from pred table ? do I need pred table ?

    def update_outlier_threshold(self, value):
        self.outlier_threshold = value

    def receive_outlier_data(self, data):
        outlier_list = []
        for item in data:
            for file, score in item.items():
                if score > self.outlier_threshold:
                    outlier_list.append(file)
        return outlier_list

    def update(self):
        pass