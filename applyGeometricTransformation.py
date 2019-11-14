
# (INPUT) startXs: NxF matrix representing the starting X coordinates of all the features in the first frame for all the bounding boxes
# (INPUT) startYs: NxF matrix representing the starting Y coordinates of all the features in the first frame for all the bounding boxes
# (INPUT) newXs: NxF matrix representing the second X coordinates of all the features in the first frame for all the bounding boxes
# (INPUT) newYs: NxF matrix representing the second Y coordinates of all the features in the first frame for all the bounding boxes
# (INPUT) bbox: Fx4x2 matrix representing the four new corners of the bounding box where F is the number of detected objects
# (OUTPUT) Xs: N1xF matrix representing the X coordinates of the remaining features in all the bounding boxes after eliminating outliers
# (OUTPUT) Ys: N1xF matrix representing the Y coordinates of the remaining features in all the bounding boxes after eliminating outliers
# (OUTPUT) newbbox: Fx4x2 the bounding box position after geometric transformation



def applyGeometricTransformation(startXs,startYs,newXs,newYs,bbox):
    
    
    return Xs,Ys,newbbox

