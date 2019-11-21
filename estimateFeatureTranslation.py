
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

def estimateFeatureTranslation(startX,startY,Ix,Iy,img1,img2):
    
    # Sampling Window Side Length
    size_window = 9
    
    # Image Dimensions
    xMax, yMax = img1.shape

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
            if ((TL_x+r < xMax) & (TL_y+c < yMax) & (TL_x-r >= 0) & (TL_y-c >= 0)):
                sum_Ix_Ix = sum_Ix_Ix + Ix[TL_x+r,TL_y+c] * Ix[TL_x+r,TL_y+c]
                sum_Ix_Iy = sum_Ix_Iy + Ix[TL_x+r,TL_y+c] * Iy[TL_x+r,TL_y+c]
                sum_Iy_Iy = sum_Iy_Iy + Iy[TL_x+r,TL_y+c] * Iy[TL_x+r,TL_y+c]
                sum_Ix_It = sum_Ix_It + Ix[TL_x+r,TL_y+c] * It[TL_x+r,TL_y+c]
                sum_Iy_It = sum_Iy_It + Iy[TL_x+r,TL_y+c] * It[TL_x+r,TL_y+c]

    # A Matrix in KLT Linear System Equation
    gradientMatrix = np.array([[sum_Ix_Ix, sum_Ix_Iy],[sum_Ix_Iy, sum_Iy_Iy]])
            
    # b Matrix in KLT Linear System Equation
    temporalMatrix = -np.array([[sum_Ix_It],[sum_Iy_It]])
        
    # Solving Linear System Equation    
    A = np.squeeze(np.transpose(gradientMatrix))
    b = np.squeeze(temporalMatrix)
    flow = np.dot(A,b)
    
    newX = startX + int(flow[0])
    newY = startY + int(flow[1])
    
    #print(flow)
    
    return newX,newY