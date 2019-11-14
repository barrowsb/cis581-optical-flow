
# (INPUT) startX: Represents the starting X coordinate for a single feature in the first frame
# (INPUT) startY: Represents the starting Y coordinate for a single feature in the first frame
# (INPUT) Ix: HxW matrix representing the gradient along the X-direction
# (INPUT) Iy: HxW matrix representing the gradient along the Y-direction
# (INPUT) img1: HxWx3 matrix representing the first image frame
# (INPUT) img2: HxWx3 matrix representing the second image frame
# (OUTPUT) newX: Represents the new X coordinate for a single feature in the second frame
# (OUTPUT) newY: Represents the new Y coordinate for a single feature in the second frame

def estimateFeatureTranslation(startX,startY,Ix,Iy,img1,img2):
    
    
    
    return newX,newY