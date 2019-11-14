
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
    pathHistory = []
    while(True): 
        frameFound, frame = rawVideo.read()  # Extract Current Frame
        
        if (frameFound):                     # If frames remain in the video
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
            
            # If the video is on the first frame
            if (countFrame == 0):
                # Create a Bounding Rectangle on the First Frame
                bbox = rectangleCreation(frame)
    
                # Feature Detection
                startXs,startYs = getFeatures(grayFrame,bbox)

                # Plot Feature Detection on first frame bounded region
                plt.imshow(frame[:,:,[2,1,0]])
                featureImg = plt.scatter(startXs, startYs, c='b', s=2)
            
            # If this frame is after the first frame in the video
            else:                
                # Overall Tracking
                newXs,newYs = estimateAllTranslation(startXs,startYs,prevFrame,frame)
                
                # Feature Tracking
                # Sobel Gradient Filters
                Ix = cv2.Sobel(grayFrame,cv2.CV_64F,1,0,ksize=5)
                Iy = cv2.Sobel(grayFrame,cv2.CV_64F,0,1,ksize=5)
                #Loop for every feature?
                    #newX,newY = estimateFeatureTranslation(startXs,startYs,Ix,Iy,prevFrame,frame)

                # Final Transformation of Feature Positions and Box
                Xs,Ys,newbbox = applyGeometricTransformation(startXs,startYs,newXs,newYs,bbox)
                
                # Update Feature Positions and Bounding Box for Next Frame
                startXs,startYs = Xs,Ys
                bbox = newbbox
            
            # Draws the Rectangle on the RGB Frame
            bboxImg = cv2.rectangle(frame, (bbox[0,0],bbox[0,1]), (bbox[3,0], bbox[3,1]), (0,0,255), 2)
            
            # Draw persistent centroids of the bounding box for the trajectory
            bboxImg,pathHistory = drawTrajectory(bbox,bboxImg,pathHistory)
            
            # Add a new frame with bounded box
            trackedVideo.write(bboxImg)
            
            # Iterate for frame history
            prevFrame = frame
            prevGrayFrame = grayFrame
            countFrame += 1
        else:
            break
    
    
    return trackedVideo
