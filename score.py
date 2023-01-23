import numpy as np
from enum import Enum




def rmsd(M1, M2):
    if M1.shape != M2.shape:
       raise Exception('rmsd: shapes are not equal')
    return np.sqrt(((M1 - M2) ** 2).mean())
    



    
    
class Accuracy(Enum):
    RED = 0
    YELLOW = 1
    GREEN = 2
    BLUE = 3
    
class Score:
    joints = [
        [5, 7, 9],      # left elbow
        [6, 8, 10],     # right elbow
        [11, 13, 15],   # left knee
        [12, 14, 16],   # right knee
        [3, 5, 7],      # left shoulder
        [4, 6, 8],      # right shoulder
        [5, 11, 13],    # left hip
        [6, 12, 14],    # right hip
    ]  
    
    @classmethod
    def evaluate(cls, target, targetProbs, pose, poseProbs):
        error = 0.0
        for joint in cls.joints:
            if targetProbs[joint[0]] < 0.1 or \
              targetProbs[joint[1]] < 0.1 or \
              targetProbs[joint[2]] < 0.1:
                error += 0.0
                continue
            
            if poseProbs[joint[0]] < 0.1 or \
              poseProbs[joint[1]] < 0.1 or \
              poseProbs[joint[2]] < 0.1:
                error += 1.0
                continue
            
            U1 = target[joint[0]] - target[joint[1]]
            U2 = target[joint[2]] - target[joint[1]]
            
            V1 = pose[joint[0]] - pose[joint[1]]
            V2 = pose[joint[2]] - pose[joint[1]]
            
            cosU = np.dot(U1, U2) / (np.linalg.norm(U1) * np.linalg.norm(U2))
            cosV = np.dot(V1, V2) / (np.linalg.norm(V1) * np.linalg.norm(V2))
            
            mU = np.arccos(cosU)
            mV = np.arccos(cosV)
            
            t1 = np.arctan2(U1[0] * U2[1] - U1[1] * U2[0], np.dot(U1, U2))
            t2 = np.arctan2(V1[0] * V2[1] - V1[1] * V2[0], np.dot(V1, V2))
            
            t1 = np.fmod(2.0 * np.pi + t1, 2.0 * np.pi)
            t2 = np.fmod(2.0 * np.pi + t2, 2.0 * np.pi)
            
            error += min(np.abs(t1 - t2), 2.0 * np.pi - np.abs(t1 - t2)) / np.pi
            print(mU, mV, t1, t2)
        print('----------')

        if error < 0.5:
            return Accuracy.BLUE
        elif error < 1.0:
            return Accuracy.GREEN
        elif error < 2.0:
            return Accuracy.YELLOW
        else:
            return Accuracy.RED
            
        