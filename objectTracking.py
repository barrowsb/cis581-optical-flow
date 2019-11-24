# (INPUT) rawVideo: The input video containing one or more objects
# (INPUT) n_box: The number of bounding boxes
# (INPUT) max_pts: Maximum number of points to be tracked
# (INPUT) sigma: Standard deviation for Gaussian blur
# (INPUT) window_size: Size of window around each feature point to be analyzed for feature tracking
# (OUTPUT) trackedVideo: The generated output video showing all the tracked features,bounding boxes, and a feature's trajectory

from rectangleCreation import rectangleCreation
from getFeatures import getFeatures
from estimateAllTranslation import estimateAllTranslation
from applyGeometricTransformation import applyGeometricTransformation
from drawTrajectory import drawTrajectory
import cv2
import matplotlib.pyplot as plt
import numpy as np

def objectTracking(rawVideo,n_box,max_pts=20,sigma=1,window_size=25,choice):
    
    # Output Video Formatting
    width = int(rawVideo.get(3))
    height = int(rawVideo.get(4))
    fps = int(rawVideo.get(5))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    trackedVideo = cv2.VideoWriter('trackedVideo.avi', fourcc, fps, (width, height)) # RGB output video
    
    # Loop Through Video Frames
    countFrame = 0
    pathHistory = []
    while(True):
        
        # Extract Current Frame
        frameFound, frame = rawVideo.read()
        blankFrame = np.zeros((height,width,3), dtype=np.uint8)
        featureImg = blankFrame

        # If frames remain in the video
        if (frameFound):
            
            # Convert frame to grayscale
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # If the video is on the first frame
            if (countFrame == 0):
                
                # Create Bounding Rectangle(s) on the First Frame
                bbox = rectangleCreation(frame,n_box)
    
                # Feature Detection
                startXs,startYs = getFeatures(grayFrame,bbox,shi=True,max_pts=max_pts)
                pt,dim,n_box = bbox.shape
                # Split if multiple bboxes
                if (n_box > 1):
                    startXs1 = startXs[:,0]
                    startYs1 = startYs[:,0]
                    startXs2 = startXs[:,1]
                    startYs2 = startYs[:,1]
                    if (n_box > 2):
                        startXs3 = startXs[:,2]
                        startYs3 = startYs[:,2]

                # Visualize features
                for box in range(startXs.shape[1]):
                    for point in range(startXs.shape[0]):
                        featureImg = cv2.circle(featureImg, (int(startXs[point,box]),int(startYs[point,box])),2,(255,0,0),4)

                # Plot Feature Detection on first frame bounded region
                plt.imshow(frame[:,:,[2,1,0]])
                featureImg = plt.scatter(startXs, startYs, c='b', s=2)
            
            # If this frame is after the first frame in the video
            else:
                
                # Initialize output
                newbbox = np.zeros(shape=(pt,dim,n_box))
                
                if (n_box == 1):  # For 1 Box
                    
                    # Overall Translation
                    newXs,newYs = estimateAllTranslation(startXs,startYs,prevFrame,frame,sigma=sigma,window_size=window_size)
                    
                    # Final Transformation of Feature Positions and Box
                    Xs,Ys,newbbox = applyGeometricTransformation(startXs,startYs,newXs,newYs,np.squeeze(bbox),width,height,n_box,choice)
                    startXs,startYs = Xs,Ys
                    
                    # Visualize features
                    for point in range(startXs.shape[0]):
                        featureImg = cv2.circle(featureImg, (int(startXs[point]),int(startYs[point])),2,(255,0,0),4)
                    
                if (n_box == 2):  # For 2 Boxes
                    
                    # Overall Translation
                    newXs1,newYs1 = estimateAllTranslation(startXs1,startYs1,prevFrame,frame,sigma=sigma,window_size=window_size)
                    newXs2,newYs2 = estimateAllTranslation(startXs2,startYs2,prevFrame,frame,sigma=sigma,window_size=window_size)
        
                    # Final Transformation of Feature Positions and Box
                    Xs1,Ys1,newbbox[:,:,0] = applyGeometricTransformation(startXs1,startYs1,newXs1,newYs1,np.squeeze(bbox[:,:,0]),width,height,n_box,choice)
                    Xs2,Ys2,newbbox[:,:,1] = applyGeometricTransformation(startXs2,startYs2,newXs2,newYs2,np.squeeze(bbox[:,:,1]),width,height,n_box,choice)
                    startXs1,startYs1 = Xs1,Ys1
                    startXs2,startYs2 = Xs2,Ys2
                    
                    for point in range(startXs1.shape[0]):
                        featureImg = cv2.circle(featureImg, (int(startXs1[point]),int(startYs1[point])),2,(255,0,0),4)
                    for point in range(startXs2.shape[0]):
                        featureImg = cv2.circle(featureImg, (int(startXs2[point]),int(startYs2[point])),2,(255,0,0),4)

                    
                if (n_box == 3):  # For 3 Boxes
                    
                    # Overall Translation
                    newXs1,newYs1 = estimateAllTranslation(startXs1,startYs1,prevFrame,frame,sigma=sigma,window_size=window_size)
                    newXs2,newYs2 = estimateAllTranslation(startXs2,startYs2,prevFrame,frame,sigma=sigma,window_size=window_size)
                    newXs3,newYs3 = estimateAllTranslation(startXs3,startYs3,prevFrame,frame,sigma=sigma,window_size=window_size)
        
                    # Final Transformation of Feature Positions and Box
                    Xs1,Ys1,newbbox[:,:,0] = applyGeometricTransformation(startXs1,startYs1,newXs1,newYs1,np.squeeze(bbox[:,:,0]),width,height,n_box,choice)
                    Xs2,Ys2,newbbox[:,:,1] = applyGeometricTransformation(startXs2,startYs2,newXs2,newYs2,np.squeeze(bbox[:,:,1]),width,height,n_box,choice)
                    Xs3,Ys3,newbbox[:,:,2] = applyGeometricTransformation(startXs3,startYs3,newXs3,newYs3,np.squeeze(bbox[:,:,2]),width,height,n_box,choice)
                    startXs1,startYs1 = Xs1,Ys1
                    startXs2,startYs2 = Xs2,Ys2
                    startXs3,startYs3 = Xs3,Ys3
                    
                    for point in range(startXs1.shape[0]):
                        featureImg = cv2.circle(featureImg, (int(startXs1[point]),int(startYs1[point])),2,(255,0,0),4)
                    for point in range(startXs2.shape[0]):
                        featureImg = cv2.circle(featureImg, (int(startXs2[point]),int(startYs2[point])),2,(255,0,0),4)
                    for point in range(startXs3.shape[0]):
                        featureImg = cv2.circle(featureImg, (int(startXs3[point]),int(startYs3[point])),2,(255,0,0),4)
                
                # Update Feature Positions and Bounding Box for Next Frame
                bbox = newbbox.reshape((pt,dim,n_box))
            
            # Draw the Rectangle(s) on the RGB Frame
            for i in range(n_box):
                overlay = cv2.rectangle(blankFrame, (int(bbox[0,0,i]),int(bbox[0,1,i])), (int(bbox[3,0,i]), int(bbox[3,1,i])), (0,0,255), 2)            
                
            # Overlay the Boxes onto the Frame
            bboxImg = cv2.add(frame,overlay)
            
            # Draw persistent centroids of the bounding box for the trajectory
            bboxImg,pathHistory = drawTrajectory(bbox,bboxImg,pathHistory)
            
            # Add a new frame with bounded box
            trackedVideo.write(bboxImg)
            
            # Iterate for frame history
            prevFrame = frame
            countFrame += 1
        
        # If no frames remain
        else:
            break

    return trackedVideo