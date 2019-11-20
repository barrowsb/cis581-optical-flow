
# (INPUT) startXs: NxF matrix representing the starting X coordinates of all the features in the first frame for all the bounding boxes
# (INPUT) startYs: NxF matrix representing the starting Y coordinates of all the features in the first frame for all the bounding boxes
# (INPUT) img1: HxWx3 matrix representing the first image frame
# (INPUT) img2: HxWx3 matrix representing the second image frame
# (OUTPUT) newXs: NxF matrix representing the new X coordinates of all the features in all the bounding boxes
# (OUTPUT) newYs: NxF matrix representing the new Y coordinates of all the features in all the bounding boxes

import numpy as np
from scipy.interpolate import interp2d
import math
import cv2
from estimateFeatureTranslation import *


def estimateAllTranslation(startXs,startYs,img1,img2):
    
    # Convert Images to Grayscale
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
    # Sobel Gradient Filters
    Ix = cv2.Sobel(img1,cv2.CV_64F,1,0,ksize=5)
    Iy = cv2.Sobel(img1,cv2.CV_64F,0,1,ksize=5)
    
    # Feature Translation
    newX,newY = estimateFeatureTranslation(startXs,startYs,Ix,Iy,img1,img2)
    
    # Translate Feature Coordinates in Bounding Boxes by Flow
    newXs = startXs + int(flow[0])
    newYs = startYs + int(flow[1])
    
    
    return newXs,newYs