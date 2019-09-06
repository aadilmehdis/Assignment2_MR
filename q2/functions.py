import numpy as np
import matplotlib.pyplot as plt
import cv2
import time


def correspondingFeatureDetection(img1, img2):

    orb = cv2.ORB_create()
    Keypoints1, Descriptors1 = orb.detectAndCompute(img1, None)
    # Keypoints1, Descriptors1 = orb.compute(img1,Keypoints1)
    # print(Keypoints1)
    Keypoints2, Descriptors2 = orb.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(Descriptors1,Descriptors2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    kp1_list = np.mat([])
    kp2_list = np.mat([])
    k = 0

    for m in matches:

        img1Idx = m.queryIdx
        img2Idx = m.trainIdx

        (img1x, img1y) = Keypoints1[img1Idx].pt
        (img2x, img2y) = Keypoints2[img2Idx].pt
        # print([img1x,img1y,1])
        if k == 0:
            kp1_list = [[img1x,img1y,1]]
            kp2_list = [[img2x,img2y,1]]
            k = 1
        else:
            kp1_list = np.append(kp1_list,[[img1x,img1y,1]],axis = 0)
            kp2_list = np.append(kp2_list,[[img2x,img2y,1]],axis = 0)
        
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,Keypoints1,img2,Keypoints2,matches[:30],None, flags=2)

    plt.imshow(img3),plt.show()
    return kp1_list,kp2_list#Keypoints1, Descriptors1, Keypoints2, Descriptors2, matches

def F_matrix(image_coords_1, image_coords_2):
    '''
    image_coords_1  : N*3 homogeneous coordinates of image pixels    
    image_coords_2  : N*3 homogeneous coordinates of image pixels
    return          : Fundamental Matrix of dimension 3*3
    '''
    
    A = np.zeros((len(image_coords_1),9))

    # Constructing A matrix of size N*9
    for i in range(len(image_coords_1)):
        A[i,:] = np.kron(image_coords_1[i,:], image_coords_2[i,:])
        
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    F = np.reshape(vh[8,:], (3,3))
    uf, sf, vhf = np.linalg.svd(F, full_matrices=True)
    
    F = uf @ np.diag(np.array([sf[0], sf[1], 0])) @ vhf

    return F

def F_RANSAC(image_points_1, image_points_2, threshold, n_iters):
    '''
        image_points_1  : N*3 matrix of a normalized 2D homogeneous of image 1 
        image_points_2  : N*3 matrix of a normalized 2D homogeneous of image 2
        threshold       : Inlier threshold
        n_iters         : Number of Iterations 
        return          : Fundamental Matrix of dimension 3*3
    '''

    n = 8 
    F = np.zeros((3,3))
    max_inliers = -9999999

    for i in range(n_iters):

        # Randomly sample 8 matching points from image 1 and image 2
        indices = np.random.choice(image_points_1.shape[0], n, replace=False)  
        matched_points_1 = image_points_1[indices] 
        matched_points_2 = image_points_2[indices]

        # Get the F matrix for the current iteration
        F_current = F_matrix(matched_points_1, matched_points_2)

        # Counting the number of inliers 
        number_current_inliers = 0
        for i in range(len(image_points_1)):
            error = np.abs(image_points_2[i,:] @ F_current @ image_points_1[i,:].T)
            if error < threshold:
                number_current_inliers += 1

        # Updating F matrix
        if number_current_inliers > max_inliers:
            max_inliers = number_current_inliers
            F = F_current
    return F


def NormalizationMat(image_coords):
    '''
        image_coords : N*3 matrix of image coordinates
        return       : 3*3 Scale Transformation matrix 
    '''

    mu = np.mean(image_coords,axis = 0)
    d = 0
    for i in range(len(image_coords)):
        d = d + np.sqrt((image_coords[i,0] - mu[0])**2 + (image_coords[i,1] - mu[1])**2)

    d = d/i
    T = np.mat([
        [1.44/d, 0, -1.44 * mu[0]/d], 
        [0, 1.44/d, -1.44 * mu[1]/d],
        [0,      0,               1]
        ])
    
    return T
    


if __name__ == "__main__":

    img1 = cv2.imread('../mr19-assignment2-data/images/000000.png')
    img2 = cv2.imread('../mr19-assignment2-data/images/000001.png')

    kp1, kp2 = correspondingFeatureDetection(img1, img2)
    T1 = NormalizationMat(kp1[0:9,:])
    T2 = NormalizationMat(kp2[0:9,:])

    points1 = T1 @ kp1.T
    points2 = T2 @ kp2.T


    F = F_RANSAC(points1.T, points2.T, 0.005, 500)

    F = T2.T @ F @ T1

    for i in range(8):
        print(kp2[i,:]@T2.T@F@T1@kp1[i,:].T)