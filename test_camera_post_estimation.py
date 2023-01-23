import numpy as np
import cv2

import pose_estimation
from pose_estimation import PoseEstimation



cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow("frame", 720, 720)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
    
nFrame = 0
while True:

    ret, frame = cap.read()
    h, w, c = frame.shape
    frame = frame[0:h, int(w/2-h/2):int(w/2+h/2)]

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break


    if nFrame % 1 == 0:
        labels = PoseEstimation.labelImage(frame)
        labelledFrame = PoseEstimation.applyLabels(frame, labels)
        cv2.imshow('frame', frame)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame)
    nFrame += 1
        
    if cv2.waitKey(1) == ord('q'):
        break

        
    

cap.release()
cv2.destroyAllWindows()