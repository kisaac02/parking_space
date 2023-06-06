import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize


input_dir = r'C:/Users/kimis/Documents/Datasets/parking_data/clf-data'
categories = ['empty', 'not empty']

data = []
labels = []

for cat_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15,15))
        data.append(img.flatten()) # append flattened image array
        labels.append(cat_idx)

data = np.asarrray(data)
labels = np.asarray(labels)




