
# (INPUT) startX: Represents the starting X coordinate for a single feature in the first frame
# (INPUT) startY: Represents the starting Y coordinate for a single feature in the first frame
# (INPUT) Ix: HxW matrix representing the gradient along the X-direction
# (INPUT) Iy: HxW matrix representing the gradient along the Y-direction
# (INPUT) img1: HxWx3 matrix representing the first image frame
# (INPUT) img2: HxWx3 matrix representing the second image frame
# (OUTPUT) newX: Represents the new X coordinate for a single feature in the second frame
# (OUTPUT) newY: Represents the new Y coordinate for a single feature in the second frame

import numpy as np
import math
from numpy.linalg import pinv
import cv2
from interp2 import *

def estimateFeatureTranslation(startX,startY,Ix,Iy,img1,img2,window_size):
    
    # Sampling Window Side Length
    size_window = window_size
    
    # Image Dimensions
    yMax, xMax = img1.shape

    # Temporal Gradient
    It = img2 - img1
    
    # Use these???
    #x_grid, y_grid = np.meshgrid(n_cols, n_rows)
    #f = interp2d(x_grid, y_grid)
    
    # Center of Window is an individual feature coordinate
    x_center = startX
    y_center = startY
    
    sum_Ix_Ix = 0 
    sum_Ix_Iy = 0
    sum_Iy_Iy = 0
    sum_Ix_It = 0
    sum_Iy_It = 0
    TL_x = int(x_center - math.floor(size_window/2))
    TL_y = int(y_center - math.floor(size_window/2))
    
    # Loop through window around feature
    for r in range(size_window):
        for c in range(size_window):
            if ((TL_x+c < xMax) & (TL_y+r < yMax) & (TL_x-c >= 0) & (TL_y-r >= 0)):
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
    A = np.squeeze(np.linalg.inv(gradientMatrix))
    b = np.squeeze(temporalMatrix)
    flow = np.dot(A,b)
    
    newX = startX + math.ceil(flow[0])
    newY = startY + math.ceil(flow[1])
    
    #print(flow)
    
    
    return newX,newY