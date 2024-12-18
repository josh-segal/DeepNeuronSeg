from PyQt5.QtCore import QObject
import os
class OutlierController(QObject):
    def __init__(self, model, view):
        super().__init__()
        self.model = model
        self.view = view

        self.view.update_outlier_threshold_signal.connect(self.model.update_outlier_threshold)

    def receive_outlier_data(self, data):
        print("# outliers: ", len(data))
        outlier_list = self.model.receive_outlier_data(data)
        outlier_list = [os.path.splitext(os.path.basename(path))[0] for path in outlier_list]
        self.view.update_outliers(outlier_list)