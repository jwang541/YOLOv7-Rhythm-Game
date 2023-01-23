import numpy as np
import cv2

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
import hourglass


# =============================================================================
# data = torch.load('./models/hg_s8_b1.pth.tar')
# model = hourglass.hg(num_stacks=8, num_blocks=1, num_classes=16)
# model = nn.DataParallel(model)
# model.load_state_dict(data['state_dict'])
# model = model.cuda()
# model.eval()
# =============================================================================


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data = torch.load('./models/yolov7-w6-pose.pt', map_location=device)
model = data['model']
model.float().eval()

if torch.cuda.is_available():
    model.half().to(device)


inWidth = 640
inHeight = 640
nParts = 16
POSE_PAIRS = [
    [15, 13], 
    [13, 11], 
    [16, 14], 
    [14, 12], 
    [11, 12], 
    [5, 11],            
    [6, 12], 
    [5, 6], 
    [5, 7], 
    [6, 8], 
    [7, 9], 
    [8, 10], 
    [1, 2],
    [0, 1], 
    [0, 2], 
    [1, 3], 
    [2, 4], 
    [3, 5], 
    [4, 6],
]


class PoseEstimation:
    def labelImage(image):
        with torch.no_grad():
            modelIn = letterbox(image, 640, stride=64, auto=True)[0]
            modelIn = transforms.ToTensor()(modelIn)
            modelIn = torch.tensor(np.array([modelIn.numpy()]))
            
            if torch.cuda.is_available():
                modelIn = modelIn.half().to(device)
    
            modelOut, _ = model(modelIn)
            
            modelOut = non_max_suppression_kpt(modelOut, 0.25, 0.65, 
                                               nc=model.yaml['nc'], 
                                               nkpt=model.yaml['nkpt'], 
                                               kpt_label=True)
    
            #with torch.no_grad():
            modelOut = output_to_keypoint(modelOut)
            #print(modelOut.shape)
            
            if modelOut.shape[0] > 0:
                keypoints = modelOut[0, 7:]
                reshaped = np.asarray(keypoints).reshape(17, 3)                
                return np.append(reshaped[:, 0:2] / 640.0, reshaped[:, 2:3], axis=1)
            else:
                return np.zeros([17, 3])
    
    
    def applyLabels(img, labels):
        labelledImg = img
        for i in range(len(labels)):
            x, y, prob = labels[i]
            if prob < 0.1:
                continue
            
            cv2.circle(labelledImg, 
                       (int(x * labelledImg.shape[0]), int(y * labelledImg.shape[1])), 
                       3, (0,225,255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(labelledImg, "{}".format(i), 
                        (int(x * labelledImg.shape[0]), int(y * labelledImg.shape[1])), 
                        cv2.FONT_HERSHEY_PLAIN, 1, 
                        (0,0,255), 1, lineType=cv2.LINE_AA)
            
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            x1, y1, prob1 = labels[partA]
            x2, y2, prob2 = labels[partB]
            if prob1 < 0.1 or prob2 < 0.1:
                continue
            
            cv2.line(labelledImg, 
                     (int(x1 * labelledImg.shape[0]), int(y1 * labelledImg.shape[1])), 
                     (int(x2 * labelledImg.shape[0]), int(y2 * labelledImg.shape[1])), 
                     (0, 255, 0), 2)
        return labelledImg
    