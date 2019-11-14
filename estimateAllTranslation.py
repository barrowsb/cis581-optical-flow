
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


def estimateAllTranslation(startXs,startYs,img1,img2):
    
    # Convert Images to Grayscale
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Dimensional Parameters
    n_rows,n_cols = img1.shape
    n_features = startXs.shape  
    size_window = 9
        
    # Sobel Gradient Filters
    Ix = cv2.Sobel(img1,cv2.CV_64F,1,0,ksize=5)
    Iy = cv2.Sobel(img1,cv2.CV_64F,0,1,ksize=5)
    
    # Temporal Gradient
    It = img2 - img1
    
    #x_grid, y_grid = np.meshgrid(n_cols, n_rows)
    #f = interp2d(x_grid, y_grid)
    
    # only need to run once on one window anywhere?
    # can iterate in different spots to improve accuracy?
    # the below center is random
    
    x_center = 100
    y_center = 100
    
    sum_Ix_Ix = np.zeros((1,1))
    sum_Ix_Iy = np.zeros((1,1))
    sum_Iy_Iy = np.zeros((1,1))
    sum_Ix_It = np.zeros((1,1))
    sum_Iy_It = np.zeros((1,1))
    TL_x = x_center - math.floor(size_window/2)
    TL_y = y_center - math.floor(size_window/2)
    
    for r in range(size_window):
        for c in range(size_window):
            sum_Ix_Ix = sum_Ix_Ix + Ix[TL_y+r,TL_x+c] * Ix[TL_y+r,TL_x+c]
            sum_Ix_Iy = sum_Ix_Iy + Ix[TL_y+r,TL_x+c] * Iy[TL_y+r,TL_x+c]
            sum_Iy_Iy = sum_Iy_Iy + Iy[TL_y+r,TL_x+c] * Iy[TL_y+r,TL_x+c]
            sum_Ix_It = sum_Ix_It + Ix[TL_y+r,TL_x+c] * It[TL_y+r,TL_x+c]
            sum_Iy_It = sum_Iy_It + Iy[TL_y+r,TL_x+c] * It[TL_y+r,TL_x+c]

    # A Matrix in KLT Linear System Equation
    gradientMatrix = np.array([[sum_Ix_Ix, sum_Ix_Iy],[sum_Ix_Iy, sum_Iy_Iy]])
    
    # b Matrix in KLT Linear System Equation
    temporalMatrix = -np.array([[sum_Ix_It],[sum_Iy_It]])
    
    # Solving Linear System Equation    
    A = np.squeeze(np.transpose(gradientMatrix))
    b = np.squeeze(temporalMatrix)
    flow = np.dot(A,b)
    
    # Translate Feature Coordinates in Bounding Boxes by Flow
    newXs = startXs + int(flow[0])
    newYs = startYs + int(flow[1])
    
    
    return newXs,newYs