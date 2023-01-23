import numpy as np
import cv2
import torch
import gc

import pose_estimation
from pose_estimation import PoseEstimation




cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow("frame", 720, 720)


cap = cv2.VideoCapture('./files/togen_renka.mp4')
capDim = (int(cap.get(3)), int(cap.get(4)))
capFPS = cap.get(5)
out = cv2.VideoWriter('./out/labelled_video.mp4', 
                         cv2.VideoWriter_fourcc(*'mp4v'), capFPS, capDim)


while cap.isOpened():
    ret, frame = cap.read()

    if not ret: 
        print("Can't receive frame (stream end?). Exiting ...")
        break

    labels = PoseEstimation.labelImage(frame)
    labelledFrame = PoseEstimation.applyLabels(frame, labels)
    out.write(labelledFrame)
    cv2.imshow('frame', labelledFrame)
    #print(torch.cuda.memory_summary())
    
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()




    
    