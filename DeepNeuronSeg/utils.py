import json

def get_data(file_path='image_metadata.json'):
    with open(file_path, 'r') as f:
                data = json.load(f)
    return data

def set_data(file_path='image_metadata.json', metadata=None):
    with open(file_path, 'w') as f:
        json.dump(metadata, f, indent=4)