import cv2
import numpy as np

import PyQt5 as qt
import PyQt5.QtGui as gui
import PyQt5.QtWidgets as widget
import PyQt5.QtCore as core
import sys
import time
import pickle
import gc

import pose_estimation
from pose_estimation import PoseEstimation
from score import Score
from score import Accuracy


# TODO: implement exponential smoothing
# TODO: make 3D pose comparison
# TODO: polish UI

class CameraThread(core.QThread):
    imageUpdateSignal = core.pyqtSignal(np.ndarray)
    
    def __init__(self, camCap):
        super().__init__()
        self.active = True
        self.capture = camCap
    
    def run(self):
        while self.active:
            ret, frame = self.capture.read()
            if ret:
                self.imageUpdateSignal.emit(frame)
        self.capture.release()
        
    def stop(self):
        self.active = False
        self.wait()
        
        
class PoseEstimationThread(core.QThread):
    poseUpdateSignal = core.pyqtSignal()
    scoreUpdateSignal = core.pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.active = True
    
    def run(self):
        while self.active:
            time.sleep(0.1)
            self.poseUpdateSignal.emit()
            self.scoreUpdateSignal.emit()
        
    def stop(self):
        self.active = False
        self.wait()
        
        
class MapThread(core.QThread):
    mapUpdateSignal = core.pyqtSignal(np.ndarray)
    
    def __init__(self, mapCap):
        super().__init__()
        self.active = True
        self.capture = mapCap
    
    def run(self):
        while self.active:
            #time.sleep(0.033333)
            time.sleep(1)
            ret, frame = self.capture.read()
            if ret:
                self.mapUpdateSignal.emit(frame)
        self.capture.release()
        
    def stop(self):
        self.active = False
        self.wait()
        
  
        
class Window(widget.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Poise: Pose Estimation Rhythm Game")
        
        self.leftImage = widget.QLabel(self)
        self.leftImage.setGeometry(0, 0, 810, 810)
        self.leftImage.move(100, 0)
        
        self.rightImage = widget.QLabel(self)
        self.rightImage.setGeometry(0, 0, 810, 810)
        self.rightImage.move(1010, 0)
        
        testPixmap = gui.QPixmap('./files/test_pic.png')
        self.leftImage.setPixmap(testPixmap)
        self.rightImage.setPixmap(testPixmap)
    
        self.poseEstimationLabels = None
        self.poseEstimationThread = PoseEstimationThread()
        self.poseEstimationThread.poseUpdateSignal.connect(self.updatePoseEstimation)
        self.poseEstimationThread.scoreUpdateSignal.connect(self.updateScore)
        self.poseEstimationThread.start()
        
        self.cameraCapture = cv2.VideoCapture(0)
        self.lastCameraImage = None
        self.cameraThread = CameraThread(self.cameraCapture)
        self.cameraThread.imageUpdateSignal.connect(self.updateCameraImage)
        self.cameraThread.start()
        
        self.mapCapture = cv2.VideoCapture('./maps/' + sys.argv[1] + '/video.mp4')
        self.mapThread = MapThread(self.mapCapture)
        self.mapThread.mapUpdateSignal.connect(self.updateMapImage)
        self.mapThread.start()
        
        self.scoreLabel = widget.QLabel(self)
        self.scoreLabel.setGeometry(0, 0, 300, 200)
        self.scoreLabel.move(810, 810)
        self.scoreLabel.setText('Fun')
        self.scoreLabel.setStyleSheet('''
            QLabel { 
                background-color : white; 
                color : black; 
                font-size : 72px
            }''')
        
        with open('./maps/' + sys.argv[1] + '/poses.pkl', 'rb') as f:
            self.poseByFrame = pickle.load(f)

        

    def updateCameraImage(self, image):
        img = image
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        img = img[0:h, int(w/2-h/2):int(w/2+h/2)]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        self.lastCameraImage = img
        
        if self.poseEstimationLabels is not None:
            img = PoseEstimation.applyLabels(img, self.poseEstimationLabels)
        
        img = gui.QImage(img.data, 
                         img.shape[1], img.shape[0], 
                         gui.QImage.Format_RGB888)
        img = img.scaled(810, 810, core.Qt.KeepAspectRatio)

        self.leftImage.setPixmap(gui.QPixmap.fromImage(img))
        
        
    def updatePoseEstimation(self):
        if self.lastCameraImage is not None:
            self.poseEstimationLabels = PoseEstimation.labelImage(self.lastCameraImage)
        
        
    def updateMapImage(self, image):
        img = image
        h, w, c = img.shape
        img = img[0:h, int(w/2-h/2):int(w/2+h/2)]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        nFrame = int(self.mapCapture.get(cv2.CAP_PROP_POS_FRAMES))
        if 0 <= nFrame < len(self.poseByFrame):
            labels = self.poseByFrame[nFrame]['pose']
            img = PoseEstimation.applyLabels(img, labels)

        img = gui.QImage(img.data, 
                         img.shape[1], img.shape[0], 
                         gui.QImage.Format_RGB888)
        img = img.scaled(810, 810, core.Qt.KeepAspectRatio)

        self.rightImage.setPixmap(gui.QPixmap.fromImage(img))
        
        
    def updateScore(self):
        if self.poseEstimationLabels is not None:
            nFrame = int(self.mapCapture.get(cv2.CAP_PROP_POS_FRAMES))
            if 0 <= nFrame < len(self.poseByFrame):
                mapLabels = self.poseByFrame[nFrame]['pose']
                
                M1 = np.empty([0, 3])
                M2 = np.empty([0, 3])
                
                for i in range(len(mapLabels)):
                    if mapLabels[i] is not None:
                        if self.poseEstimationLabels[i] is not None:
                            M1 = np.append(M1, [np.asarray(mapLabels[i])], axis=0)
                            M2 = np.append(M2, [np.asarray(self.poseEstimationLabels[i])], axis=0)
                                  
                score = Score.evaluate(M1[:, 0:2], M1[:, 2:3], M2[:, 0:2], M2[:, 2:3])
                if score == Accuracy.RED:
                    self.scoreLabel.setText('Red')
                elif score == Accuracy.YELLOW:
                    self.scoreLabel.setText('Yellow')
                elif score == Accuracy.GREEN:
                    self.scoreLabel.setText('Green')
                else:
                    self.scoreLabel.setText('Blue')
                    
        
    
    def closeEvent(self, event):
        self.cameraThread.stop()
        self.poseEstimationThread.stop()
        self.mapThread.stop()
        event.accept()
    


if __name__=="__main__":
    if (len(sys.argv) < 2):
        sys.exit()
    
    app = widget.QApplication(sys.argv)
    window = Window()
    window.showMaximized()
    sys.exit(app.exec_())
    
    
    
    
    
    