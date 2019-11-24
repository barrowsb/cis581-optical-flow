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



def estimateAllTranslation(startXs,startYs,img1,img2,sigma,window_size): 
    
    # Convert Images to Grayscale
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
    # Blur
    img1 = cv2.GaussianBlur(img1,(4*sigma+1,4*sigma+1),sigma)
    img2 = cv2.GaussianBlur(img2,(4*sigma+1,4*sigma+1),sigma)
    
    # Gradient Filters
    Iy, Ix = np.gradient(img1.astype(float))
    
    # Dimensional Parameters
    n_rows,n_cols = img1.shape
    n_features = startXs.shape[0] 

    # Feature Translation
    newXs = np.zeros((n_features,1))
    newYs = np.zeros((n_features,1))
    for i in range(n_features):
        startX = startXs[i]
        startY = startYs[i]
        newX,newY = estimateFeatureTranslation(startX,startY,Ix,Iy,img1,img2,window_size)
        newXs[i] = newX
        newYs[i] = newY    
        
    
    return newXs,newYs