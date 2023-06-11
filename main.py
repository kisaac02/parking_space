import cv2
import os
import numpy as np
import pickle
from skimage.transform import resize

def get_parking_spots_bboxes(connected_components):
    # grabs each component and exctracts bounding box
    (totallabels, label_ids, values, centroid) = connected_components
    slots = []
    coef = 1
    for i in range(1, totallabels):
        # extract coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])
    return slots

EMPTY = True
NOT_EMPTY = False
MODEL = pickle.load(open("model.p", "rb"))

def empty_or_not(spot_bgr):
    flat_data = []
    # reshapes image
    img_resized = resize(spot_bgr, (15,15,3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    # model pretrained classifier
    y_output = MODEL.predict(flat_data)

    if y_output == 0:
        return EMPTY
    else:
        return NOT_EMPTY

# rough way of calculating the difference between to images. To see if anything hs changed.
def calc_diff(img1, img2):
    return np.abs(np.mean(img1) - np.mean(img2))

folder_path = r'C:/Users/kimis/Documents/Datasets/parking_data'
video_path =  os.path.join(folder_path, 'parking_1920_1080_loop.mp4')
mask_path =  os.path.join(folder_path, 'mask_1920_1080.png')

# using the mask to create bounding boxes
mask = cv2.imread(mask_path, 0) # open as greyscale image
cap = cv2.VideoCapture(video_path)

# connected commonents. Graph theory. Each box is a connected component. Connected to other points in one components but components aren't connected to each other.
comps = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

# gives bounding box for each parking spot as list of coords
spots = get_parking_spots_bboxes(comps)

# iterate through frames in video
spots_status = [None for j in spots]
diffs = [None for j in spots]

previous_frame = None

ret = True
step = 30
frame_number = 0

while ret:
    ret, frame = cap.read()
    
    # quick calc to see if frame has changed
    if frame_number % step == 0 and previous_frame is not None:
        for spot_idx, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1+h, x1:x1+w, :]
            diffs[spot_idx] = calc_diff(spot_crop, previous_frame[y1:y1+h, x1:x1+w, :])          


    if frame_number % step == 0:
        for spot_idx, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1+h, x1:x1+w, :]

            spot_status = empty_or_not(spot_crop)

            spots_status[spot_idx] = spot_status
    
    # copy frame into previous_frame variable
    if frame_number % step == 0:
        previous_frame = frame.copy()

# decoupled spot status prediction from drawing rectangle
    for spot_idx, spot in enumerate(spots):
        spot_status = spots_status[spot_idx]
        x1, y1, w, h = spots[spot_idx]
        if spot_status:
        # draws a rectangle for each spot
            frame = cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0,255,0), 2) # (255,0,0) blue colour. width = 2
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0,0,255), 2)

    # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'): # press q to close window
        break

    frame_number += 1
# issue - popup window doesn't close

cap.release()
cv2.destroyAllWindows()