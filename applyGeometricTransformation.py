
# (INPUT) startXs: NxF matrix representing the starting X coordinates of all the features in the first frame for all the bounding boxes
# (INPUT) startYs: NxF matrix representing the starting Y coordinates of all the features in the first frame for all the bounding boxes
# (INPUT) newXs: NxF matrix representing the second X coordinates of all the features in the first frame for all the bounding boxes
# (INPUT) newYs: NxF matrix representing the second Y coordinates of all the features in the first frame for all the bounding boxes
# (INPUT) bbox: 4x2xF matrix representing the four new corners of the bounding box where F is the number of detected objects
# (INPUT) xMax: Width of frame in pixels
# (INPUT) yMax: Height of frame in pixels
# (OUTPUT) newXs: N1xF matrix representing the X coordinates of the remaining features in all the bounding boxes after eliminating outliers
# (OUTPUT) newYs: N1xF matrix representing the Y coordinates of the remaining features in all the bounding boxes after eliminating outliers
# (OUTPUT) newbbox: Fx4x2 the bounding box position after geometric transformation

from skimage import transform as tf
import numpy as np
#from rejectOutliers import *

def applyGeometricTransformation(startXs,startYs,newXs,newYs,bbox,xMax,yMax):
    
    # Max allowed pixel distance between start and new location
    threshold = 10
    
    # Eliminate features that behave abnormally
#    startXs,startYs,newXs,newYs = rejectOutliers(startXs,startYs,newXs,newYs)
    
    # Loop Through Feature Points to Remove unideal points
    i = 0
    sum_shift_x = 0
    sum_shift_y = 0
    while(i < len(startXs)):
        # If the change in feature position exceeds 4 pixels in x or y
        if ((newXs[i] - startXs[i] > threshold) | (newYs[i] - startYs[i] > threshold)):
            # Elimate the Feature Point across all lists
            newXs = np.delete(newXs,i)
            newYs = np.delete(newYs,i)
            startXs = np.delete(startXs,i)
            startYs = np.delete(startYs,i)
        # If the change in position results in leaving the image dimensions
        elif ((newXs[i] > xMax) | (newYs[i] > yMax) | (newXs[i] < 0) | (newYs[i] < 0)):
            # Elimate the Feature Point across all lists
            newXs = np.delete(newXs,i)
            newYs = np.delete(newYs,i)
            startXs = np.delete(startXs,i)
            startYs = np.delete(startYs,i)
        # If the feature position change is acceptable
        else:
            # Sum the shifts in x and y
            sum_shift_x = sum_shift_x + newXs[i] - startXs[i]
            sum_shift_y = sum_shift_y + newYs[i] - startYs[i]
            i += 1
    
    # Find average of the x and y feature shifts
    if (i != 0):
        avg_shift_x = sum_shift_x / i
        avg_shift_y = sum_shift_y / i
    else:
        avg_shift_x = 0
        avg_shift_y = 0

    # Mannually translate bounding box relative to feature movement
    shiftMatrix = np.zeros((4,2))
    shiftMatrix[:,0] = avg_shift_x
    shiftMatrix[:,1] = avg_shift_y
    newbbox = bbox + shiftMatrix
    newbbox = np.int16(newbbox)

    smallestX = 1000
    smallestY = 1000
    biggestX = 0
    biggestY = 0
    
    # Grow size of box to ensure all good points are inside
    for i in range(len(startXs)):
        # Check Mins and Maxes
        if (newXs[i] < smallestX):
            smallestX = newXs[i]
        if (newXs[i] > biggestX):
            biggestX = newXs[i]
        if (newYs[i] < smallestY):
            smallestY = newYs[i]
        if (newYs[i] > biggestY):
            biggestY = newYs[i]
        
        # Enlarge Bounding Box
        # If point is to the left of left box boundary increase box left
        if ((newXs[i] < newbbox[0,0]) or (newXs[i] < newbbox[2,0])) :
            newbbox[0,0]=newXs[i]
            newbbox[2,0]=newXs[i]
        # If point is to the right of right box boundary increase box right  
        if ((newXs[i] > newbbox[1,0]) or (newXs[i] > newbbox[3,0])) :
            newbbox[1,0]=newXs[i]
            newbbox[3,0]=newXs[i]
        # If point is above top box boundary increase box up
        if ((newYs[i] < newbbox[0,1]) or (newYs[i] < newbbox[1,1])) :
            newbbox[0,1]=newYs[i]
            newbbox[1,1]=newYs[i]
        # If point is below bottom box boundary increase box down  
        if ((newYs[i] > newbbox[2,1]) or (newYs[i] > newbbox[3,1])) :
            newbbox[2,1]=newYs[i]
            newbbox[3,1]=newYs[i]
            
    # Shrink The Bounding Box
    # If the smallestX is larger than the left boundary
    if ((smallestX > newbbox[0,0]) and (smallestX > newbbox[2,0])):
        newbbox[0,0]=smallestX
        newbbox[2,0]=smallestX
    # If the biggestX is smaller than the right boundary
    if ((biggestX < newbbox[1,0]) and (biggestX < newbbox[3,0])):
        newbbox[1,0]=biggestX
        newbbox[3,0]=biggestX
    # If the smallestY is larger than the upper boundary
    if ((smallestY > newbbox[0,1]) and (smallestY > newbbox[1,1])):
        newbbox[0,1]=smallestY
        newbbox[1,1]=smallestY
    # If the biggestY is smaller than the lower boundary
    if ((biggestY > newbbox[2,1]) and (biggestY > newbbox[3,1])):
        newbbox[2,1]=biggestY
        newbbox[3,1]=biggestY
            
    newbbox = np.int16(newbbox)
    
    return newXs,newYs,newbbox
