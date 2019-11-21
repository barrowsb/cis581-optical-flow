
# (INPUT) img: HxW matrix representing the grayscale input image
# (INPUT) bbox: 4x2xF matrix representing the four corners of the bounding box where F is the number of objects you would like to track
# (OUTPUT) feat_X: NxF matrix representing the N row coordinates of the features across F objects
# (OUTPUT) feat_Y: NxF matrix representing the N column coordinates of the features across F objects

import cv2
import numpy as np

def getFeatures(img,bbox,shi=False,max_pts=150):
    
    # Dimensional Parameters for Looping
    point,dimension,n_box = bbox.shape
    feat_X = np.zeros((max_pts,n_box))
    feat_Y = np.zeros((max_pts,n_box))

    # Loop for each bounding box
    for i in range(n_box):
        # Examine only image inside bounding box(es)
        rowStart = int(bbox[0,1,i])
        rowEnd = int(bbox[2,1,i])
        colStart = int(bbox[0,0,i])
        colEnd = int(bbox[1,0,i])
        window = img[rowStart:rowEnd, colStart:colEnd]
        
        # Blur image
        window = cv2.blur(window,(5,5))
        
        if shi == True:
            # Shi-Tomasi corner detector
            corners = cv2.goodFeaturesToTrack(window,max_pts,0.001,2)

            # Count number of corners found
            found = corners.shape[0]
            
            # Convert x and y back to global image coordinates
            feat_X[0:found,i] = corners[:,:,0].astype(np.float64).ravel() + bbox[0,0,i]
            feat_Y[0:found,i] = corners[:,:,1].astype(np.float64).ravel() + bbox[0,1,i]
            feat_X[found:max_pts,i] = None
            feat_Y[found:max_pts,i] = None
            
        if shi == False:
            # Harris Corner Detector
            cimg = cv2.cornerHarris(window,2,3,0.04)
            
            # Start of ANMS        
            # Tune for thresholding
            thresh1 = 0.005 # pre-threshold
            thresh2 = 1.1   # anms-threshold
            
            # %% Pre-thresholding
            
            # Create meshgrids and flatten
            nr,nc = cimg.shape
            cols,rows = np.meshgrid(range(nc),range(nr))
            colsf = cols.flatten()
            rowsf = rows.flatten()
            cimgf = cimg.flatten()
            
            # Threshold
            prethresh = thresh1*np.amax(cimgf)
            cimgf = np.where(cimgf>prethresh, np.asarray(cimgf), np.zeros((nr*nc,)))
            indices = np.nonzero(cimgf)
            cimgf = list(cimgf[indices])
            rowsf = list(rowsf[indices])
            colsf = list(colsf[indices])
            length = len(cimgf)
            
            # %% ANMS
            
            # Initialize variables
            # (for matrices: row index is reference corner, column index is corner being compared to)
            inf = (nr**2)+(nc**2) # longest possible distance^2
            greater = np.zeros((length,length),dtype=bool) # Boolean matrix (1 if above anms-thresh)
            dist2 = np.zeros((length,length)) # distance^2 matrix (from ref to compare corner)
            
            # Find all corners above anms-threshold
            compare,ref = np.meshgrid(cimgf,cimgf) # row-wise and column-wise matrices of cimgf vectors
            anmsthresh = ref*thresh2
            greater = compare > anmsthresh
            
            # Compute distance from all corners (ref) to all other corners (comp)
            rows_comp,cols_comp = np.meshgrid(rowsf,colsf) # row-wise and column-wise matrices of indices
            rows_ref = np.transpose(rows_comp) # (rows_ref does not change row-wise, rows_comp does)
            cols_ref = np.transpose(cols_comp) # (cols_ref does not change row-wise, cols_comp does)
            dist2 = (rows_ref - rows_comp)**2 + (cols_ref - cols_comp)**2
        
            # Find minimum distance to other corner above thresh
            logical_dist2 = np.where(greater,dist2,np.ones((length,length))*inf) # inf where not(greater)
            j_min = np.argmin(logical_dist2,axis=1) # col (j) indices of minimum distance in each row
            i_min =  np.asarray(range(length)) # row (i) indices
            min_dist2 = dist2[i_min,j_min] # minimum distance to sufficiently larger corner
            
            # Find row and col indices of minimum distance from above
            min_rows = np.asarray(rowsf)[j_min]
            min_cols = np.asarray(colsf)[j_min]
            
            # Sort by decending radius
            z2 = zip(min_dist2,min_rows,min_cols)
            z2 = sorted(z2, key=lambda x: x[0],reverse=True)
            min_dist2_sorted,min_rows_sorted,min_cols_sorted = zip(*list(z2))
            min_dist2_sorted = list(min_dist2_sorted)
            min_rows_sorted = list(min_rows_sorted)
            min_cols_sorted = list(min_cols_sorted)
            
            # Set outputs, trimmed to N = max_pts corners
            found = len(min_cols_sorted)
            if found < max_pts:
                max_pts = found
                print("Warning: found fewer than N corners above pre-threshold... N reduced to " + str(found) + ".")
            x = np.reshape(np.asarray(min_cols_sorted[0:max_pts]),(max_pts,1))
            y = np.reshape(np.asarray(min_rows_sorted[0:max_pts]),(max_pts,1))
            
            # Convert x and y back to global image coordinates
            feat_X[:,i] = x.ravel() + bbox[0,0,i]
            feat_Y[:,i] = y.ravel() + bbox[0,1,i]
    
    return feat_X,feat_Y