# (INPUT) startXs: NxF matrix representing the starting X coordinates of all the features in the first frame for all the bounding boxes
# (INPUT) starYs: NxF matrix representing the starting Y coordinates of all the features in the first frame for all the bounding boxes
# (INPUT) newXs: NxF matrix representing the new X coordinates of all the features in all the bounding boxes
# (INPUT) newYs: NxF matrix representing the new Y coordinates of all the features in all the bounding boxes
# (INPUT) m: Number of standard deviations
# (OUTPUT) startXs: NxF matrix representing the starting X coordinates of all the features in the first frame for all the bounding boxes without outliers
# (OUTPUT) starYs: NxF matrix representing the starting Y coordinates of all the features in the first frame for all the bounding boxes without outliers
# (OUTPUT) newXs: NxF matrix representing the new X coordinates of all the features in all the bounding boxes without outliers
# (OUTPUT) newYs: NxF matrix representing the new Y coordinates of all the features in all the bounding boxes without outliers

import numpy as np

def rejectOutliers(startXs,startYs,newXs,newYs,m=2):
    
    if startXs.shape[0]>6:
        dist = np.zeros(shape=startXs.shape,dtype=np.float16)
        for i in range(startXs.shape[0]):
            dist[i] = np.sqrt((startXs[i]-newXs[i])**2+(startYs[i]-newYs[i])**2)
        
        inliers = abs(dist - np.mean(dist)) <= m * np.std(dist)+0.005
    
        startXs = startXs[inliers]
        startYs = startYs[inliers]
        newXs = newXs[inliers]
        newYs = newYs[inliers]
        
        startXs = np.expand_dims(startXs,axis=1)
        startYs = np.expand_dims(startYs,axis=1)
        newXs = np.expand_dims(newXs,axis=1)
        newYs = np.expand_dims(newYs,axis=1)
        

    return startXs,startYs,newXs,newYs