
# (INPUT) rawVideo: The input video containing one or more objects
# (OUTPUT) trackedVideo: The generated output video showing all the tracked features (please do try to show the trajectories for all the features) on the object as well as the bounding boxes

from rectangleCreation import *
from getFeatures import *
from estimateAllTranslation import *
from estimateFeatureTranslation import *
from applyGeometricTransformation import *
from drawTrajectory import *
import cv2
import matplotlib.pyplot as plt


def objectTracking(rawVideo):
    
    # Output Video Formatting
    width = int(rawVideo.get(3))
    height = int(rawVideo.get(4))
    fps = int(rawVideo.get(5))
    n_frames = int(rawVideo.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    trackedVideo = cv2.VideoWriter('trackedVideo.avi', fourcc, fps, (width, height)) # RGB output video
    #trackedVideo = cv2.VideoWriter('trackedVideo.avi', fourcc, fps, (width, height), 0) # Gray output video
    
    # Loop Through Video Frames
    countFrame = 0
    frameCollection = np.zeros((height,width,3,n_frames+1))
    pathHistory = []
    while(True): 
        frameFound, frame = rawVideo.read()  # Extract Current Frame
        frameCollection[:,:,:,countFrame] = frame
        if (frameFound):                     # If frames remain in the video
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
            
            # If the video is on the first frame
            if (countFrame == 0):
                # Create a Bounding Rectangle on the First Frame
                bbox = rectangleCreation(frame)
    
                # Feature Detection
                startXs,startYs = getFeatures(grayFrame,bbox)

                # Plot Feature Detection
                plt.imshow(frame[:,:,[2,1,0]])
                featureImg = plt.scatter(startXs, startYs, c='b', s=5)
            
            # If this frame is after the first frame in the video
            else:
                # Overall Tracking
                #newXs,newYs = estimateAllTranslation(startXs,startYs,frameCollection[:,:,:,countFrame-1],frameCollection[:,:,:,countFrame])
                
                # Feature Tracking
                #Loop for every feature?
                    #newX,newY = estimateFeatureTranslation(startXs,startYs,Ix,Iy,frameCollection[:,:,:,countFrame-1],frameCollection[:,:,:,countFrame])

                #Xs,Ys,newbbox = applyGeometricTransformation(startXs,startYs,newXs,newYs,bbox)
                #bbox = newbbox
                
                # PLACEHOLDER (DELETE WHEN NEWBOX EXISTS)
                bbox = rectangleCreation(frame)
            
            # Draws the Rectangle on the RGB Frame
            bboxImg = cv2.rectangle(frame, (bbox[0,0],bbox[0,1]), (bbox[3,0], bbox[3,1]), (0,0,255), 2)
            
            # Draw persistent centroids of the bounding box for the trajectory
            bboxImg,pathHistory = drawTrajectory(bbox,bboxImg,pathHistory)
            
            # Add a new frame with bounded box
            trackedVideo.write(bboxImg)
            
            # Iterate for frameCollection history
            countFrame += 1
        else:
            break
    
    
    return trackedVideo
