import numpy as np

def rejectOutliers(startXs,startYs,newXs,newYs,m=2):
    
    print(startXs.shape)
    
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
        
#    print(dist)
#    print(np.mean(dist))
#    print(np.std(dist))
#    print(inliers)
    print(startXs.shape)
    print()
    
    return startXs,startYs,newXs,newYs