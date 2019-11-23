# (INPUT) rawVideo: The input video containing one or more objects
# (OUTPUT) trackedVideo: The generated output video showing all the tracked features (please do try to show the trajectories for all the features) on the object as well as the bounding boxes

from rectangleCreation import *
from getFeatures import *
from estimateAllTranslation import *
from applyGeometricTransformation import *
from drawTrajectory import *
import cv2
import matplotlib.pyplot as plt


def objectTracking(rawVideo,n_boxes,max_pts,sigma,window_size):
    
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
        blankFrame = np.zeros((height,width,3), dtype=np.uint8)

        if (frameFound):                     # If frames remain in the video
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
            
            # If the video is on the first frame
            if (countFrame == 0):
                # Create a Bounding Rectangle on the First Frame
                bbox = rectangleCreation(frame,n_boxes)
    
                # Feature Detection
                startXs,startYs = getFeatures(grayFrame,bbox,shi=True,max_pts=max_pts)
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
                
                r,c,n_box = bbox.shape
                newbbox = np.zeros(shape=(4,2,n_box))

                if (n_box == 1):  # For 1 Box
                    
                    # Overall Translation
                    newXs,newYs = estimateAllTranslation(startXs,startYs,prevFrame,frame,sigma=sigma,window_size=window_size)
                    
                    # Final Transformation of Feature Positions and Box
                    Xs,Ys,newbbox = applyGeometricTransformation(startXs,startYs,newXs,newYs,np.squeeze(bbox),width,height)
                    
                    startXs,startYs = Xs,Ys
                    
                if (n_box == 2):  # For 2 Boxes
                    
                    # Overall Translation
                    newXs1,newYs1 = estimateAllTranslation(startXs1,startYs1,prevFrame,frame,sigma=sigma,window_size=window_size)
                    newXs2,newYs2 = estimateAllTranslation(startXs2,startYs2,prevFrame,frame,sigma=sigma,window_size=window_size)
        
                    # Final Transformation of Feature Positions and Box
                    Xs1,Ys1,newbbox[:,:,0] = applyGeometricTransformation(startXs1,startYs1,newXs1,newYs1,np.squeeze(bbox[:,:,0]),width,height)
                    Xs2,Ys2,newbbox[:,:,1] = applyGeometricTransformation(startXs2,startYs2,newXs2,newYs2,np.squeeze(bbox[:,:,1]),width,height)
                
                    startXs1,startYs1 = Xs1,Ys1
                    startXs2,startYs2 = Xs2,Ys2
                
                # Update Feature Positions and Bounding Box for Next Frame
                bbox = newbbox.reshape((4,2,n_box))
            
            # Draws the Rectangle(s) on the RGB Frame
            point,dimension,n_box = bbox.shape
            for i in range(n_box):
                pureBoxes = cv2.rectangle(blankFrame, (int(bbox[0,0,i]),int(bbox[0,1,i])), (int(bbox[3,0,i]), int(bbox[3,1,i])), (0,0,255), 2)
            
            # Overlay the Boxes onto the Frame
            bboxImg = cv2.add(frame,pureBoxes)
            
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
