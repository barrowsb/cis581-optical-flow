
# (INPUT)
# (INPUT)
# (INPUT)
# (OUTPUT)
# (OUTPUT)

import cv2

def drawTrajectory(bbox,bboxImg,pathHistory):
    center = (int(bbox[0,0] + (bbox[1,0] - bbox[0,0])/2), int(bbox[0,1] + (bbox[2,1] - bbox[0,1])/2))
    pathHistory.append(center)
    for centroid in range(len(pathHistory)):
        bboxImg = cv2.circle(bboxImg, pathHistory[centroid], 1, (0,255,255), 2)
                
    return bboxImg,pathHistory
