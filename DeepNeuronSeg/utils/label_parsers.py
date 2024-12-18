import numpy as np
from PIL import Image
import cv2
import pandas as pd
from DeepNeuronSeg.utils.utils import norm_coords

import xml.etree.ElementTree as ET


def parse_png_label(label_file):
    label_array = np.array(Image.open(label_file))
    _, _, _, centroids = cv2.connectedComponentsWithStats(label_array)

    coordinates = [tuple(map(int, cent)) for cent in centroids[1:]]

    return coordinates

def parse_txt_label(label_file):
    with open(label_file, 'r') as file:
        content = file.read()
        coordinates = []

        largest_x = 0
        largest_y = 0

        for line in content.strip().splitlines():
            x, y = map(float, line.strip().split('\t'))

            if x > largest_x:
                largest_x = x
            if y > largest_y:
                largest_y = y

            coordinates.append((x, y))

    if largest_x > 512 and largest_y > 512:
        coordinates = norm_coords(coordinates, max_x=largest_x, max_y=largest_y)
    elif largest_x > 512:
        coordinates = norm_coords(coordinates, max_x=largest_x)
    elif largest_y > 512:
        coordinates = norm_coords(coordinates, max_y=largest_y)

    return coordinates

def parse_csv_label(label_file):
    df = pd.read_csv(label_file)
    
    # Extract the 'X' and 'Y' columns
    x_values = df['X'].tolist()
    y_values = df['Y'].tolist()

    largest_x = max(x_values)
    largest_y = max(y_values)

    coordinates = list(zip(x_values, y_values))

    if largest_x > 512 and largest_y > 512:
        coordinates = norm_coords(coordinates, max_x=largest_x, max_y=largest_y)
    elif largest_x > 512:
        coordinates = norm_coords(coordinates, max_x=largest_x)
    elif largest_y > 512:
        coordinates = norm_coords(coordinates, max_y=largest_y)
    
    return coordinates

def parse_xml_label(label_file):
    tree = ET.parse(label_file)
    root = tree.getroot()
    largest_x = 0
    largest_y = 0
    # Extract all MarkerX and MarkerY values
    coordinates = []
    for marker in root.findall('.//Marker'):
        x = marker.find('MarkerX').text
        y = marker.find('MarkerY').text

        if x > largest_x:
            largest_x = x
        if y > largest_y:
            largest_y = y

        coordinates.append((float(x), float(y)))

    if largest_x > 512 and largest_y > 512:
        coordinates = norm_coords(coordinates, max_x=largest_x, max_y=largest_y)
    elif largest_x > 512:
        coordinates = norm_coords(coordinates, max_x=largest_x)
    elif largest_y > 512:
        coordinates = norm_coords(coordinates, max_y=largest_y)

    return coordinates