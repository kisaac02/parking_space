import cv2

video_path = r'C:/Users/kimis/Documents/Datasets/parking_data/parking_crop_loop.mp4'

cap = cv2.VideoCapture(video_path)

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