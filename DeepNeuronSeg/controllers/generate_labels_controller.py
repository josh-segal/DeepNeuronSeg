from PyQt5.QtCore import QObject

class GenerateLabelsController(QObject):
    def __init__(self, model, view):
        super().__init__()
        self.model = model
        self.view = view

        self.view.generate_labels_signal.connect(self.generate_labels)
        self.view.update_signal.connect(self.update_view)
        self.view.next_image_signal.connect(self.next_image)
        self.view.load_image_signal.connect(self.load_image)

    def generate_labels(self):
        self.model.generate_labels()
        self.update_view()

    def load_image(self, index):
        self.model.image_manager.set_index(index)
        left_item, left_index, left_total, left_points = self.model.image_manager.get_item(show_labels=True)
        right_item, _, _, right_points = self.model.image_manager.get_item(show_masks=True, no_wrap=True)
        if left_item:
            self.view.left_image.display_frame(left_item, left_index, left_total, left_points)
        else:
            self.view.left_image.clear()
            self.view.left_image.text_label.setText("No image found")
        if right_item:
            self.view.right_image.display_frame(right_item, left_index, left_total, right_points)
        else:
            self.view.right_image.clear()
            self.view.right_image.text_label.setText("No mask found")

    def next_image(self):
        self.model.image_manager.next_image()
        left_item, left_index, left_total, left_points = self.model.image_manager.get_item(show_labels=True)
        right_item, right_index, right_total, right_points = self.model.image_manager.get_item(show_masks=True, no_wrap=True)
        if left_item:
            self.view.left_image.display_frame(left_item, left_index, left_total, left_points)
        else:
            self.view.left_image.clear()
            self.view.left_image.text_label.setText("No image found")
        if right_item:
            self.view.right_image.display_frame(right_item, left_index, left_total, right_points)
        else:
            self.view.right_image.clear()
            self.view.right_image.text_label.setText("No mask found")

    def update_view(self):
        images = self.model.image_manager.get_images()
        left_item, left_index, left_total, left_points = self.model.image_manager.get_item(show_labels=True)
        right_item, right_index, right_total, right_points = self.model.image_manager.get_item(show_masks=True, no_wrap=True)
        if left_item:
            self.view.left_image.display_frame(left_item, left_index, left_total, left_points)
        else:
            self.view.left_image.clear()
            self.view.left_image.text_label.setText("No image found")
        if right_item:
            self.view.right_image.display_frame(right_item, left_index, left_total, right_points)
        else:
            self.view.right_image.clear()
            self.view.right_image.text_label.setText("No mask found")
        self.view.update_response(images)   


