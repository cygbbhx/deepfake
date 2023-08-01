import cv2
import numpy as np
from .capture import VideoCaptureThreading

def getFrames(filePath):
    cap = cv2.VideoCapture(filePath)
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    #frame_num = 300

    cap = VideoCaptureThreading(src=filePath)
    cap.start()
    
    count = 0

    images = []
    success, image = cap.read()
    while count < frame_num:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        success, image = cap.read()
        count += 1
    cap.stop()

    frames = np.stack(images)
    return frames