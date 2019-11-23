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

def applyGeometricTransformation(startXs,startYs,newXs,newYs,bbox,xMax,yMax):
    
    # Max allowed pixel distance between start and new location
    threshold = 4
    
    # Loop Through Feature Points
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
#            # Sum the shifts in x and y
#            sum_shift_x = sum_shift_x + newXs[i] - startXs[i]
#            sum_shift_y = sum_shift_y + newYs[i] - startYs[i]
            i += 1
    
#    # Find average of the x and y feature shifts
#    if (i != 0):
#        avg_shift_x = sum_shift_x / i
#        avg_shift_y = sum_shift_y / i
#    else:
#        avg_shift_x = 0
#        avg_shift_y = 0
#    
#    # Translate bounding box relative to feature movement
#
#    shiftMatrix = np.zeros((4,2))
#    shiftMatrix[:,0] = avg_shift_x
#    shiftMatrix[:,1] = avg_shift_y
#    newbbox = bbox + shiftMatrix
#    newbbox = np.int16(newbbox)
#    tform=np.zeros((4,2))
    newbbox = np.zeros((4,2))
    startXs = startXs.reshape(-1,1)
    startYs = startYs.reshape(-1,1)
#    tform[:,0]=tf.estimate_transform('similarity',startXs,newXs)
#    tform[:,1]=tf.estimate_transform('similarity',startYs,newYs)
    start = np.hstack((startXs,startYs))
    new = np.hstack((newXs,newYs))
    tform=tf.estimate_transform('similarity',start,new)
#    tform_x=tf.estimate_transform('similarity',startXs,newXs)
#    tform_y=tf.estimate_transform('similarity',startYs,newYs)   
#    newbbox = tf.warp(bbox, inverse_map = tform.inverse)
#    tform = np.vstack((tform_x,tform_y))
#    newbbox[:,0] = tf.warp(bbox, inverse_map = tform_x.inverse)
#    newbbox[:,1] = tf.warp(bbox, inverse_map = tform_y.inverse)
    newbbox = tf.warp(bbox, inverse_map = tform.inverse)
    newbbox = np.int16(newbbox)

    return newXs,newYs,newbbox

