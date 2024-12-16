from PyQt5.QtCore import QObject

class UploadController(QObject):
    def __init__(self, model, view):
        super().__init__()
        self.model = model
        self.view = view

        self.view.upload_images_signal.connect(self.upload_images)
        self.view.update_signal.connect(self.update_view)
        self.view.load_image_signal.connect(self.load_image)
        self.view.next_image_signal.connect(self.next_image)
        
        self.model.update_images_signal.connect(self.update_images)

        self.update_view()

    def upload_images(self, images, project, cohort, brain_region, image_id):
        print("upload images controller")
        self.model.upload_images(images, project, cohort, brain_region, image_id)
        

    def update_images(self, images):
        self.view.update_images(images)

    def load_image(self, index):
        self.model.image_manager.set_index(index)
        item, index, total, points = self.model.image_manager.get_item(show_labels=False)
        self.view.image_display.display_frame(item, index, total, points)

    def next_image(self):
        self.model.image_manager.next_image()
        item, index, total, points = self.model.image_manager.get_item(show_labels=False)
        self.view.image_display.display_frame(item, index, total, points)

    def update_view(self):
        images = self.model.image_manager.get_images()
        self.view.update_response(images)
        item, index, total, points = self.model.image_manager.get_item(show_labels=False)
        self.view.image_display.display_frame(item, index, total, points)
    