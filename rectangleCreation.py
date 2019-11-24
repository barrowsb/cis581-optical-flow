
# (INPUT) frame: HxWx3 matrix representing the RGB image 
# (OUTPUT) bbox: 4x2xF matrix representing four corners of rectangle, where F is each rectangle

import cv2
import numpy as np

def rectangleCreation(frame,n_boxes):

    bbox = np.zeros((4,2,n_boxes))
    
    for i in range(n_boxes):
        name = "Draw bounding box #" + str(i+1) + " and press 'Enter'."
        (x, y, bw, bh) = cv2.selectROI(name,frame)
        cv2.destroyAllWindows()
        bbox[:,:,i] = np.array([[x,y],[x+bw,y],[x,y+bh],[x+bw,y+bh]])
   
    return bbox
