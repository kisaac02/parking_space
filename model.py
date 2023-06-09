import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from scikit-learn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# TODO move image processing onto GPU. Use CUDA or 
input_dir = r'C:/Users/kimis/Documents/Datasets/parking_data/clf-data'
categories = ['empty', 'not_empty']
# "C:\Users\kimis\Documents\Datasets\parking_data\clf-data\not_empty"

data = []
labels = []

for cat_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15,15))
        data.append(img.flatten()) # append flattened image array
        labels.append(cat_idx)

data = np.asarray(data)
labels = np.asarray(labels)

# train/test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels) # same proportion of labels in train and test.

# model
classifier = SVC()

parameters = [{'gamma':[0.01,0.001,0.0001], 'C':[1,10,100,1000]}]
# trains 12 image classifiers

grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(x_train, y_train)

# test
best_estimator = grid_search.best_estimator_

y_predict = best_estimator.predict(x_test)

score = accuracy_score(y_predict, y_test)

print(f'{score*100}% of samples where correctly classified')

