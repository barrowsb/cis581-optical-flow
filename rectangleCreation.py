
# (INPUT) frame: HxWx3 matrix representing the RGB image 
# (OUTPUT) bbox: 4x2xF matrix representing four corners of rectangle, where F is each rectangle
# (OUTPUT) bboxImg: HxWx3 matrix representing RGB with overlayed rectangle

import cv2
import numpy as np

def rectangleCreation(frame):
    x,y,w,h = 145,160,55,80  # Rectangle position and dimension
    bbox = np.array([[x,y],[x+w,y],[x,y+h],[x+w,y+h]])
    bboxImg = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)  # Draws the Rectangle on the Frame
    
    return bbox,bboxImg
