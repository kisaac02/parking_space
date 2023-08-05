# Parking space tutorial

I followed 2 videos by @ComputerVisionEngineer on YouTube to build a model that identifies empty and occupied parking spaces in a video from a static camera giving a birds eye perspective of a parking lot. The method requires a mask of the parking lot which means that the code cannot be applied to other parking lots without having masks for them as well.

Video link:
https://www.youtube.com/watch?v=F-884J2mnOY


# model.py
## Training data
The training data is small cropped images of parking spaces. Images are in folders corresponding to the class: Empty or Not Empty
The training data consists of 3045 images of each class. A Train:Test split of 80:20 is used.
The classes are visually very distinct which means we don't need complex classifier and can use a SVM.

## Pre-processing
Images are resized and flattened.

## Model
Linear Support Vector Machine classifier with best parameters of gamma and C discovered through grid search cross validation.
This classifier is robust but not state of the art.

Accuracy is used as model performance metric.

# main.py
The video of the parking lot is concatenated with the same video playing backwards as a method to get more video data to test on.
A mask of the parking lot indicating all spaces is used to segment/divide the video.
The mask is used with connectedcommponents to identify the components.
These are passed to the get_parking_spots_bboxes function to get the location and size of each spot.

Firstly I built the application to classify spots in every frame but this is unneccesary as it takes approximately 10 seconds to park a car. So classing every spot every N frames instead of each frame is better for performance.

Then to improve on this method I only wanted to classify the spots on the initial frame and then when there are changes to the spots. I used a round a simple way of calculating the change between the cropped images of each spot. If the difference of the mean of the image pixels is larger than 0.4 then the spot is passed to the empty_or_not function.
The empty_or_not function is performed for each spot when the previous is None.

For each spot on each frame a rectangle is drawn with the colour indicating the status of the parking spot.
Text is added to the video showing the number of available spots out of the total spots.