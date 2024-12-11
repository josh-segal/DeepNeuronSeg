from PyQt5.QtCore import QObject

class UploadController(QObject):
    def __init__(self, model, view):
        super().__init__()
        self.model = model
        self.view = view

        self.view.upload_images_signal.connect(self.upload_images)
        self.view.upload_labels_signal.connect(self.upload_labels)
        self.view.update_signal.connect(self.update_view)
        
        
        self.model.update_images_signal.connect(self.update_images)

    def upload_images(self, images, project, cohort, brain_region, image_id):
        print("upload images controller")
        self.model.upload_images(images, project, cohort, brain_region, image_id)
        

    def update_images(self, images):
        self.view.update_images(images)

    def upload_labels(self, labels):
        self.model.parse_labels(labels)

    def update_view(self):
        images = self.model.db.load_images()
        self.view.update_response(images)
    