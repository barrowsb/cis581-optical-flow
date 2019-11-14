
# (INPUT) rawVideo: The input video containing one or more objects
# (OUTPUT) trackedVideo: The generated output video showing all the tracked features (please do try to show the trajectories for all the features) on the object as well as the bounding boxes

from rectangleCreation import *
from getFeatures import *
from estimateAllTranslation import *
from estimateFeatureTranslation import *
from applyGeometricTransformation import *
import cv2
import matplotlib.pyplot as plt


def objectTracking(rawVideo):
    
    # Output Video Formatting
    width = int(rawVideo.get(3))
    height = int(rawVideo.get(4))
    fps = int(rawVideo.get(5))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    trackedVideo = cv2.VideoWriter('trackedVideo.avi', fourcc, fps, (width, height)) # RGB output video
    #trackedVideo = cv2.VideoWriter('trackedVideo.avi', fourcc, fps, (width, height), 0) # Gray output video
    
    # Loop Through Video Frames
    countFrame = 0
    while(True): 
        frameFound, frame = rawVideo.read()  # Extract Current Frame
        if (frameFound):                     # If frames remain in the video
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
            
            # Create a Bounding Rectangle on the First Frame
            if (countFrame == 0):
                bbox,bboxImg = rectangleCreation(frame)
                cv2.imshow('Bounding Box Image', bboxImg)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
                # Feature Detection
                x,y = getFeatures(grayFrame,bbox)

                # Plot Feature Detection
                plt.imshow(frame[:,:,[2,1,0]])
                featureImg = plt.scatter(x, y, c='b', s=5)
                

            # Feature Tracking
            #newXs,newYs = estimateAllTranslation(startXs,startYs,img1,img2)

            #newX,newY = estimateFeatureTranslation(startX,startY,Ix,Iy,img1,img2)

            #Xs,Ys,newbbox = applyGeometricTransformation(startXs,startYs,newXs,newYs,bbox)
            
            
            
            trackedVideo.write(frame)
            countFrame += 1
        else:
            break
    
    
    
    return trackedVideo
