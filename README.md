# Parking space tutorial

I followed 2 videos by @ComputerVisionEngineer on YouTube to build a model that identifies empty and occupied parking spaces in a video from a static camera giving a birds eye perspective of a parking lot. The method requires a mask of the parking lot which means that the code cannot be applied to other parking lots without having masks for them as well.

Video link:
https://www.youtube.com/watch?v=F-884J2mnOY

This is what a screenshot of what the final output video looks like. With each parking spot outlined in a colour indicating the status and the number of available spots out of the total spots displayed.

![Screenshot of parking lot video](https://github.com/kisaac02/parking_space/blob/main/Parking%20lot%20screenshot.png?raw=true)

# model.py
## Training data
The training data comprises of small cropped images of parking spaces. Images are in folders corresponding to the class: Empty or Not Empty.
The training data consists of 3,045 images of each class. A Train:Test split of 80:20 is used.
The classes are visually very distinct so a simple classfier should provide good enough performance.

## Pre-processing
Images are resized and flattened.

## Model
A Linear Support Vector Machine (SVM) classifier was used with best parameters of gamma and C discovered through grid search cross validation.
This classifier is robust but not state of the art.
Accuracy is used as model performance metric.
The trained SVM classifier model provided a high level of accuracy.

# main.py
The video of the parking lot is concatenated with the same video playing backwards as a method to get more video data to test on.
A mask of the parking lot indicating all spaces is used to segment the video frames.
The mask is used with connectedcommponents to identify the component rectangles.
These are passed to the get_parking_spots_bboxes function to get the location and size of each spot.

Firstly I built the application to classify spots in every frame but this is unneccesary as it takes approximately 10 seconds to park a car. I then changed the code to classify every spot every N frames instead of each frame is better for performance.

Then to improve on this method I only wanted to classify the spots on the initial frame and then when there are changes to the spots. I used a simple way of calculating the change between the cropped images of each spot. If the difference of the mean of the image pixels is larger than 0.4 then the spot is passed to the empty_or_not function.
The empty_or_not function is performed for each spot when the previous is None.

For each spot on each frame a rectangle is drawn with the colour indicating the status of the parking spot.
Text is added to the video showing the number of available spots out of the total spots.
