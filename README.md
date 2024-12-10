# DeepNeuronSeg
DeepNeuronSeg is a full-stack, end-to-end machine learning pipeline designed for neuroimaging data analysis. This robust framework streamlines the entire workflow, from data preprocessing and augmentation to advanced neural network-based denoising and segmentation. With a focus on performance and ease of use, DeepNeuronSeg empowers researchers to efficiently analyze complex neuroimaging datasets and derive meaningful insights with minimal overhead and lightning fast speeds.


# Installation Guide

- Installation requirements
    - Python
    - Git
- In terminal at desired location write commands:
    - mkdir test_folder
        - makes the desired directory for downloading the project
    - cd test_folder
        - navigates into the desired directory
    - Git clone https://github.com/josh-segal/DeepNeuronSeg.git
        - This downloads a copy of the project to your local computer
    - cd DeepNeuronSeg
        - This navigates into the DeepNeuronSeg project directory
    - python -m venv venv
        - This creates a python virtual environment to download all the dependencies for DeepNeuronSeg without conflict from your local system/downloads
    - venv/Scripts/activate (Windows) or source venv/bin/activate (MacOS)
        - This activates the virtual environment
    - pip install -r requirements.txt
        - This installs the dependencies required for DeepNeuronSeg
    - python -m DeepNeuronSeg
        - This launches the DeepNeuronSeg program, start exploring!
- To launch again navigate to DeepNeuronSeg directory and re-activate the virtual environment and use python -m DeepNeuronSeg

# Usage

## Upload Data

Upload images by selecting png files from file explorer

Upload labels in png (binary mask), csv, txt, XML (last 3 from imageJ cell counter download coordinates)

Option to input project ID, cohort, brain region, image ID

scroll through images to confirm or select through file selector

## Label Data

Display data to load uploaded data

Click on cells in image to set label

Right click to remove cells

Next Image to navigate over data

## Generate Labels

Generate Labels to pass images and labels to label generator

Next image to scroll through data

Display Labels to display on startup

## Create Dataset

Train Split to set amount of data to train on, remainder to validate on

Dataset Name to set name of dataset 

File selector to select which files you want to include in your dataset

## Train Network

Choose base model to train on
Choose dataset to train with

Set Epochs, batch size for training

Choose trained model name

Choose to train custom denoise model, use default denoise model, or no denoise model

Use default dataset augmentation, no dataset augmentation, or custom dataset augmentation

## Evaluate Network

Choose trained model to evaluate

Choose dataset to evaluate

Calculates average and variability metrics for chosen dataset with chosen model

Optionally display graph of number of detections and confidence of images in dataset

Download data to download a CSV of images to raw metrics

## Analyze Data

Pass through new data to the model and retrieve resultant average and variability metrics

Compares to base dataset and computes a overall variance score to determine if data is outlier

Option to display graph with new data inserted

Option to save inferences as images with predictions marked

## Extract Outliers

Displays data with outlier score above set outlier threshold, user can change threshold manually

User can validate data or relabel data

relabel data inserts image and labels into data, user can add to dataset and retrain

## Model Zoo

User can choose from any of trained models and inference images

Displays inferences for user to inspect

User can save inferences to computer
