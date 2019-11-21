
# (INPUT) frame: HxWx3 matrix representing the RGB image 
# (OUTPUT) bbox: 4x2xF matrix representing four corners of rectangle, where F is each rectangle

import cv2
import numpy as np

def rectangleCreation(frame,n_boxes):
    #n_boxes = 2  # Number of bounding boxes manually creating
    bbox = np.zeros((4,2,n_boxes))
    for i in range(n_boxes):
        name = "Draw bounding box #" + str(i+1) + " and press 'Enter'."
        (x, y, bw, bh) = cv2.selectROI(name,frame)
        cv2.destroyAllWindows()
        bbox[:,:,i] = np.array([[x,y],[x+bw,y],[x,y+bh],[x+bw,y+bh]])

#    print(bbox)        
#    x1,y1,w1,h1 = 145,160,55,80  # First Rectangle position and dimension
#    x2,y2,w2,h2 = 250,100,100,100  # Second Rectangle
#    
#    #Construct Bounding Box
#    bbox = np.zeros((4,2,n_boxes))
#    bbox[:,:,0] = np.array([[x1,y1],[x1+w1,y1],[x1,y1+h1],[x1+w1,y1+h1]])
#    bbox[:,:,1] = np.array([[x2,y2],[x2+w2,y2],[x2,y2+h2],[x2+w2,y2+h2]])
    
    return bbox