import cv2

def get_parking_spots_bboxes(connected_components):
    (totallabels, label_ids, values, centroid) = connected_components
    slots = []
    coef = 1
    for i in range(1, totallabels):
        # extract coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] = coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] = coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] = coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] = coef)

        slots.append([x1, y1, w, h])
    return slots


video_path = r'C:/Users/kimis/Documents/Datasets/parking_data/parking_crop_loop.mp4'
mask = r'C:/Users/kimis/Documents/Datasets/parking_data/mask_crop.png'

cv2.imread(mask, 0) # open as greyscale image
cap = cv2.VideoCapture(video_path)

comps = cv2.connectedComponents(mask, 4, cv2.CV_325)
# connected commonents. Graph theory. Each box is a connected component. Connected to other points in one components but components aren't connected to each other.



# iterate through frames in video
ret = True
while ret:
    ret, frame = cap.read()

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
# issue - popup window doesn't close

cap.release()
cv2.destroyAllWindows()