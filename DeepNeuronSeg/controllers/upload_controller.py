from PyQt5.QtCore import QObject

class UploadController(QObject):
    def __init__(self, model, view):
        super().__init__()
        self.model = model
        self.view = view
        self._blinded = False

        self.view.upload_images_signal.connect(self.upload_images)
        self.view.update_signal.connect(self.update_view)
        self.view.load_image_signal.connect(self.load_image)
        self.view.next_image_signal.connect(self.next_image)
        
        self.model.update_images_signal.connect(self.update_images)

        self.update_view()

    def set_blinded(self, value):
        self._blinded = value
        self.update_view()
    
    def upload_images(self, images):
        self.model.upload_images(images)
        self.update_view()

    def display_image(self):
        item, index, total, points = self.model.image_manager.get_item(show_labels=False)
        
        if item:
            self.view.image_display.display_frame(item, index, total, points)
        else:
            self.view.image_display.clear()
            self.view.image_display.text_label.setText("No image found")
        

    def update_images(self, images):
        if self._blinded:
            display_images = [str(itm[1]) for itm in images]
        else:
            display_images = [itm[0] for itm in images]
        self.view.update_images(display_images)

    def load_image(self, index):
        self.model.image_manager.set_index(index)
        self.display_image()

    def next_image(self):
        self.model.image_manager.next_image()
        self.display_image()

    def update_view(self):
        images = self.model.image_manager.get_images()
        if self._blinded:
            display_images = [str(img[1]) for img in images]
        else:
            display_images = [img[0] for img in images]
        self.view.update_response(display_images)
        self.display_image()
    