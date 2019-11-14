
# (INPUT) bbox: 4x2xF matrix of the bounding box coordinates
# (INPUT) bboxImg: HxWx3 matrix of the current image with the bounding box
# (INPUT) pathHistory: 2xN list of center coordinates of the bounding boxes
# (OUTPUT) bboxImg: Updated with all centers from past and current frames
# (OUTPUT) pathHistory: Appended with newest center coordinates

import cv2

def drawTrajectory(bbox,bboxImg,pathHistory):
    center = (int(bbox[0,0] + (bbox[1,0] - bbox[0,0])/2), int(bbox[0,1] + (bbox[2,1] - bbox[0,1])/2))
    pathHistory.append(center)
    for centroid in range(len(pathHistory)):
        bboxImg = cv2.circle(bboxImg, pathHistory[centroid], 1, (0,255,255), 2)
                
    return bboxImg,pathHistory
