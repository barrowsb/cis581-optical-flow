# (INPUT) bbox: 4x2xF matrix of the bounding box coordinates
# (INPUT) bboxImg: HxWx3 matrix of the current image with the bounding box
# (INPUT) pathHistory: 2xN list of center coordinates of the bounding boxes
# (OUTPUT) bboxImg: Updated with all centers from past and current frames
# (OUTPUT) pathHistory: Appended with newest center coordinates

import cv2
import math
import numpy as np

def drawTrajectory(bbox,bboxImg,pathHistory):
    
    # Loop for each bounding box and append the new box center to its history
    point,dimension,n_box = bbox.shape
    center = []
    for i in range(n_box):
        x_coord = math.floor(bbox[0,0,i] + (bbox[1,0,i] - bbox[0,0,i])/2)
        y_coord = math.floor(bbox[0,1,i] + (bbox[2,1,i] - bbox[0,1,i])/2)
        center = np.array([x_coord, y_coord])
        pathHistory.append(center)
        
    # Loop for each bounding box and draw a circle in the center of all boxes within history
    for i in range(n_box):
        for centroid in range(len(pathHistory)):
            bboxImg = cv2.circle(bboxImg, tuple(pathHistory[centroid]), 1, (0,255,255), 2)
                
    return bboxImg,pathHistory
