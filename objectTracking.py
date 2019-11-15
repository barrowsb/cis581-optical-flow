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
                r,c,n_box = bbox.shape
                if (n_box == 2):
                    startXs1 = startXs[:,0]
                    startYs1 = startYs[:,0]
                    startXs2 = startXs[:,1]
                    startYs2 = startYs[:,1]

                # Plot Feature Detection on first frame bounded region
                plt.imshow(frame[:,:,[2,1,0]])
                featureImg = plt.scatter(startXs, startYs, c='b', s=2)
            
            # If this frame is after the first frame in the video
            else:               
                # Sobel Gradient Filters
                Ix = cv2.Sobel(grayFrame,cv2.CV_64F,1,0,ksize=5)
                Iy = cv2.Sobel(grayFrame,cv2.CV_64F,0,1,ksize=5)
                
                r,c,n_box = bbox.shape
                newbbox = np.zeros(shape=(4,2,n_box))

                if (n_box == 1):  # For 1 Box
                    
                    # Overall Translation
                    newXs,newYs = estimateAllTranslation(startXs,startYs,prevFrame,frame)
                    
                    # Feature Translation
                    #newX,newY = estimateFeatureTranslation(startXs,startYs,Ix,Iy,prevFrame,frame)
                    
                    # Final Transformation of Feature Positions and Box
                    Xs,Ys,newbbox = applyGeometricTransformation(startXs,startYs,newXs,newYs,bbox)
                    
                    startXs,startYs = Xs,Ys
                    
                if (n_box == 2):  # For 2 Boxes
                    
                    
                    # Overall Translation
                    newXs1,newYs1 = estimateAllTranslation(startXs1,startYs1,prevFrame,frame)
                    newXs2,newYs2 = estimateAllTranslation(startXs2,startYs2,prevFrame,frame)
                    
                    # Feature Translation
                    #newX,newY = estimateFeatureTranslation(startXs,startYs,Ix,Iy,prevFrame,frame)
                    #newX,newY = estimateFeatureTranslation(startXs,startYs,Ix,Iy,prevFrame,frame)
        
                    # Final Transformation of Feature Positions and Box
                    Xs1,Ys1,newbbox[:,:,0] = applyGeometricTransformation(startXs1,startYs1,newXs1,newYs1,bbox[:,:,0])
                    Xs2,Ys2,newbbox[:,:,1] = applyGeometricTransformation(startXs2,startYs2,newXs2,newYs2,bbox[:,:,1])
                
                    startXs1,startYs1 = Xs1,Ys1
                    startXs2,startYs2 = Xs2,Ys2
                
                # Update Feature Positions and Bounding Box for Next Frame
                bbox = newbbox
            
            # Draws the Rectangle(s) on the RGB Frame
            point,dimension,n_box = bbox.shape
            indivBoxFrame = frame
            for i in range(n_box):
                bboxImg = cv2.rectangle(indivBoxFrame, (int(bbox[0,0,i]),int(bbox[0,1,i])), (int(bbox[3,0,i]), int(bbox[3,1,i])), (0,0,255), 2)
                indivBoxFrame = bboxImg
            
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
