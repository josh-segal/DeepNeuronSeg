
class OutlierModel:
    def __init__(self, db):
        super().__init__()
        self.db = db

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