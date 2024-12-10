


class AnalysisController:
    def __init__(self, model, view):
        self.model = model
        self.view = view

        self.view.inference_images_signal.connect(self.inference_images)
        self.view.save_inferences_signal.connect(self.save_inferences)
        self.view.download_data_signal.connect(self.download_data)

        self.update_model_selector()

    def inference_images(self, model_name, uploaded_files):
        self.model.inference_images(model_name, uploaded_files)
        print("controller recieved signal",)

    def save_inferences(self):
        self.model.save_inferences()

    def download_data(self):
        self.model.download_data()

    def update_model_selector(self):

        models = map(lambda model: model['model_name'], self.model.load_models())
        self.view.set_model_names(models)

    def receive_dataset_metrics(self, dataset_metrics_model):
            print("received dataset_metrics_model", dataset_metrics_model)
            self.dataset_metrics_model = dataset_metrics_model