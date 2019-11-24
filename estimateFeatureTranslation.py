# (INPUT) startX: Represents the starting X coordinate for a single feature in the first frame
# (INPUT) startY: Represents the starting Y coordinate for a single feature in the first frame
# (INPUT) Ix: HxW matrix representing the gradient along the X-direction
# (INPUT) Iy: HxW matrix representing the gradient along the Y-direction
# (INPUT) img1: HxWx3 matrix representing the first image frame
# (INPUT) img2: HxWx3 matrix representing the second image frame
# (INPUT) window_size: An integer representing the side length of the feature window
# (OUTPUT) newX: Represents the new X coordinate for a single feature in the second frame
# (OUTPUT) newY: Represents the new Y coordinate for a single feature in the second frame

import numpy as np
import math
from numpy.linalg import pinv
import cv2
from interp2 import *

def estimateFeatureTranslation(startX,startY,Ix,Iy,img1,img2,window_size):
    
    # Image Dimensions
    yMax, xMax = img1.shape

    # Temporal Gradient
    It = img2 - img1
        
    # Center of Window is an individual feature coordinate
    x_center = startX
    y_center = startY
    
    # Initialize Sums
    sum_Ix_Ix = 0 
    sum_Ix_Iy = 0
    sum_Iy_Iy = 0
    sum_Ix_It = 0
    sum_Iy_It = 0
    
    # Meshgrid
    x_grid, y_grid = np.meshgrid(np.arange(window_size), np.arange(window_size))
    TL_x = x_center - math.floor(window_size/2)  # Top Left X Coordinate of Window
    TL_y = y_center - math.floor(window_size/2)  # Top Left Y Coordinate of Window
    x_flat_W = x_grid.ravel() + TL_x
    y_flat_W = y_grid.ravel() + TL_y
    
    # Interpolation
    img1_interp = interp2(img1, x_flat_W, y_flat_W)
    
    Ix_interp = interp2(Ix, x_flat_W, y_flat_W)
    Ix_interp.reshape(-1,1)
    Iy_interp = interp2(Iy, x_flat_W, y_flat_W)
    Iy_interp.reshape(-1,1) 
        
    Ix_Iy = np.zeros((window_size*window_size,2))
    Ix_Iy[:,0] = Ix_interp
    Ix_Iy[:,1] = Iy_interp
    gradientMatrix = np.dot(np.transpose(Ix_Iy), Ix_Iy)    
    

    for i in range(5):
        # Center of Window is an individual feature coordinate

        TL_x = x_center - math.floor(window_size/2)  # Top Left X Coordinate of Window
        TL_y = y_center - math.floor(window_size/2)  # Top Left Y Coordinate of Window
        x_flat_W = x_grid.ravel() + TL_x
        y_flat_W = y_grid.ravel() + TL_y

        img2_interp = interp2(img2, x_flat_W, y_flat_W)
        It = img2_interp - img1_interp 
        temporalMatrix = -np.dot(np.transpose(Ix_Iy), It)
        
        # Solving Linear System Equation    
        A = np.squeeze(np.linalg.pinv(gradientMatrix))
        b = np.squeeze(temporalMatrix)
        flow = np.dot(A,b)
        
        x_center = x_center + (flow[0])
        y_center = y_center + (flow[1])        
    
    newX = x_center
    newY = y_center
    
    return newX,newY