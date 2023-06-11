# parking_space
Parking space tutorial

Video link:
https://www.youtube.com/watch?v=F-884J2mnOY

Mask used to segment/divide image

Using cropped area of large parking lot video

To get more video for training use video concatenated with same video playing backwards

Model is pretrained
Training data available
Training data is small cropped images of parking spaces. Empty or Not Empty

Categories are visually very distinct.
So don't need complex classifier
Classifier very robust. But not state of the art.
SVM grid search to find best paramters of gamma and C.

Don't need to clasasify spots in every frame. Takes approx 10 seconds to park a car. So classing every spot every N frames instead of frame will be better for performance.

Calculates the difference between the means of the previous spot crop and the current spot crop. If the change > 0.4 then it passes the spot to the empty or not function.
The is an rough and simple way of calculating the change between images.
Does empty or not for each spot when previous is None.