import os
import shutil
from PIL import Image
from PyQt5.QtWidgets import QDialog
from PyQt5.QtCore import pyqtSignal, QObject
from tinydb import Query
from DeepNeuronSeg.utils.utils import trim_underscores
from DeepNeuronSeg.views.widgets.frame_selection_dialog import FrameSelectionDialog
from DeepNeuronSeg.models.image_manager import ImageManager



class UploadModel(QObject):

    upload_images_signal = pyqtSignal(list)
    update_images_signal = pyqtSignal(list)

    def __init__(self, db):
        super().__init__()
        self.db = db
        self.current_index = 0
        self.uploaded_files = []
        self.image_manager = ImageManager(self.db)
    
    def upload_images(self, uploaded_files, project, cohort, brain_region, image_id):
        print("upload images model")

        self.use_selected_frame_for_all = False
        self.selected_frame = 0

        image_data = Query()
        for file in uploaded_files[:]:

            image_name = os.path.basename(file)
            image_name = trim_underscores(image_name)
            image_name = image_name.replace(".tif", ".png")

            #TODO: store this path somewhere so not hardcoded ?
            image_path = os.path.join('data', 'data_images', image_name)

            if self.db.image_table.get(image_data.file_path == image_name):
                print(f"Image already exists in database {image_name}")
                uploaded_files.remove(file)
                continue
        
            # tif operations
            if file.lower().endswith('.tif'):
                with Image.open(file) as img:
                        num_frames = img.n_frames

                        if num_frames > 1 and not self.use_selected_frame_for_all:
                            dialog = FrameSelectionDialog(num_frames)
                            if dialog.exec_() == QDialog.Accepted:
                                self.selected_frame = dialog.selected_frame
                                self.use_selected_frame_for_all = dialog.use_for_all

                            img.seek(self.selected_frame)
                            frame_to_save = img.copy()
                            frame_to_save.save(image_path, format='PNG')
                        else:
                            # print("Converting tif to png", image_path)
                            img.save(image_path, format='PNG')

            # png operations
            else:
                shutil.copy(file, image_path)
            
            file = image_path

            # add to db
            #TODO: add an apply to all for some metadata get rid of or automate per image ones, don't need metadata not a database.
            self.db.image_table.insert({
                "file_path": image_path, 
                "project": project, 
                "cohort": cohort, 
                "brain_region": brain_region, 
                "image_id": image_id if image_id else len(self.db.image_table),
                "labels": []
                })

        items = self.db.load_images()
        self.update_images_signal.emit(items)
    
    def load_image(self, index):
        self.current_index = index
        self.image_manager.set_index(index)
        item, index, total, points = self.image_manager.get_item()
        return item, index, total, points

    def next_image(self):
        self.image_manager.next_image()
        item, index, total, points = self.image_manager.get_item()
        return item, index, total, points

    def update(self):
        self.file_list.clear()
        items = self.db.load_images()
        self.file_list.addItems([os.path.basename(file) for file in items])