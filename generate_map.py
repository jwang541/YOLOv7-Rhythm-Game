import numpy as np
import cv2
import pickle
import os

import pose_estimation
from pose_estimation import PoseEstimation




cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow("frame", 720, 720)


cap = cv2.VideoCapture('./files/lilac_720.mp4')
capDim = (int(cap.get(3)), int(cap.get(4)))
capFPS = cap.get(5)


newPath = './out/lilac/'
if not os.path.exists(newPath):
    os.makedirs(newPath)
    
out = cv2.VideoWriter(newPath + 'video.mp4', 
                         cv2.VideoWriter_fourcc(*'mp4v'), capFPS, capDim)

poses = []

nFrame = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    out.write(frame)

    labels = PoseEstimation.labelImage(frame)
    labelledFrame = PoseEstimation.applyLabels(frame, labels)
    cv2.imshow('frame', labelledFrame)
        
    poses.append({
        'frame': nFrame,
        'pose': labels
    })
        
    nFrame += 1
    
    if cv2.waitKey(1) == ord('q'):
        break

with open(newPath + 'poses.pkl', 'wb') as f:
    pickle.dump(poses, f)

cap.release()
out.release()
cv2.destroyAllWindows()





    
    