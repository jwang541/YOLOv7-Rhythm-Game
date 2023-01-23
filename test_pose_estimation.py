import numpy as np
import cv2
import torch

import pose_estimation
from pose_estimation import PoseEstimation




img = cv2.imread('./files/test_pic.png')

labels = PoseEstimation.labelImage(img)
labelledImg = PoseEstimation.applyLabels(img, labels)

        
cv2.namedWindow('Labelled', cv2.WINDOW_NORMAL)
cv2.resizeWindow("Labelled", 640, 640)
cv2.imshow("Labelled", labelledImg)


print(torch.cuda.memory_summary())


cv2.imwrite('./out/labelled_pose.png', labelledImg)


cv2.waitKey(0)
cv2.destroyAllWindows()
