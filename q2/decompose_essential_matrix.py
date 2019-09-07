import numpy as np
import cv2



def algebraic_triangulation(x1, x2, P1, P2):

    X = np.zeros((len(x1),4))

    for i in range(len(x1)):
        J = np.zeros((4,4))
        J[:,0] = x1[i,0] * P1[2,:] - P1[0,:]
        J[:,1] = x1[i,1] * P1[2,:] - P1[1,:]
        J[:,2] = x2[i,0] * P2[2,:] - P2[0,:]
        J[:,3] = x2[i,1] * P2[2,:] - P2[1,:]

        u, s, vh = np.linalg.svd(J, full_matrices=False)
        X[i,:] = vh[3,:]
        X[i,:] = X[i,:] / X[i,3]
    
    return X



def calculatePointsInfrontOfCam(P,P2, points3D):
    
    
    PointsInfrontOfCam = 0

    for i in range(len(points3D)):
        if points3D[i,2] > 0:
            PointsInfrontOfCam = PointsInfrontOfCam + 1

    return PointsInfrontOfCam

    

def decompose_essential_matrix(E, K, img_points1, img_points2, K_inverse):

    # w = np.array([
    #     [ 0, -1, 0],
    #     [ 1,  0, 0],
    #     [ 0,  0, 1],
    # ])
    # z = np.array([
    #     [ 0, 1, 0],
    #     [-1, 0, 0],
    #     [ 0, 0, 0],
    # ])
    # u, s, vh = np.linalg.svd(E, full_matrices=True)

    # t = u[:,2]
    # r1 = u @ w @ vh.T
    # r2 = u @ w.T @ vh.T

    # P =  K @ np.concatenate((np.eye(3),np.zeros((3,1))),axis = 1)
    # P1 = K @ np.concatenate((r1,t),axis=1)
    # P2 = K @ np.concatenate((r1,-t),axis=1)
    # P3 = K @ np.concatenate((r2,t),axis=1)
    # P4 = K @ np.concatenate((r2,-t),axis=1)

    R1, R2, T = cv2.decomposeEssentialMat(E)

    P =  K @ np.concatenate((np.eye(3),np.zeros((3,1))),axis = 1)
    P1 = K @ np.concatenate((R1,T),axis=1)
    P2 = K @ np.concatenate((R1,-T),axis=1)
    P3 = K @ np.concatenate((R2,T),axis=1)
    P4 = K @ np.concatenate((R2,-T),axis=1)

    X_P1 = algebraic_triangulation(img_points1, img_points2, P, P1)
    X_P2 = algebraic_triangulation(img_points1, img_points2, P, P2)
    X_P3 = algebraic_triangulation(img_points1, img_points2, P, P3)
    X_P4 = algebraic_triangulation(img_points1, img_points2, P, P4)


    
    # Computing Image Coordinates for all the Triangulated Points in Image 1 and Image 2
    x1_1 = K_inverse @ P @ X_P1.T
    x2_1 = K_inverse @P1 @ X_P1.T

    x1_2 =K_inverse @ P @ X_P2.T
    x2_2 =K_inverse @ P2 @ X_P2.T

    x1_3 = K_inverse @ P @ X_P3.T
    x2_3 = K_inverse @ P3 @ X_P3.T

    x1_4 = K_inverse @ P @ X_P4.T
    x2_4 = K_inverse @ P4 @ X_P4.T

    # Computing the depth of all the reprojected image points
    d1_1 =  x1_1[2,:]
    d2_1 =  x2_1[2,:]
    score_1 = (d1_1 > 0) & (d2_1 > 0)
    score_1 = np.sum(score_1)

    d1_2 =  x1_2[2,:]
    d2_2 =  x2_2[2,:]
    score_2 = (d1_2 > 0) & (d2_2 > 0)
    score_2 = np.sum(score_2)

    d1_3 =  x1_3[2,:]
    d2_3 =  x2_3[2,:]
    score_3 = (d1_3 > 0) & (d2_3 > 0)
    score_3 = np.sum(score_3)

    d1_4 =  x1_4[2,:]
    d2_4 =  x2_4[2,:]
    score_4 = (d1_4 > 0) & (d2_4 > 0)
    score_4 = np.sum(score_4)

    index = np.argmax(np.array([score_1, score_2, score_3, score_4]))
    rotation = np.mat([])
    translation = np.mat([])
    if index == 0:
        rotation = R1
        translation = T
        Pactual = P1
    elif index == 1:
        rotation = R1
        translation = -T
        Pactual = P2
    elif index == 2:
        rotation = R2
        translation = T
        Pactual = P3        
    elif index == 3:
        rotation = R2
        translation = -T
        Pactual = P4




    # numPointsInfrontOfCam = calculatePointsInfrontOfCam(P, P1, Points3dP1)

    # maxNumPointsInfrontOfCam = numPointsInfrontOfCam
    # rotation = r1
    # translation = t
    # Pactual = P1
    # r = np.sqrt(np.sum((Points3dP1[0,:] - Points3dP1[1,:])**2))
    

    # numPointsInfrontOfCam = calculatePointsInfrontOfCam(P, P2, Points3dP2)

    # if numPointsInfrontOfCam > maxNumPointsInfrontOfCam:
    #     rotation = r1
    #     translation = -t
    #     Pactual = P2
    #     r = np.sqrt(np.sum((Points3dP2[0,:] - Points3dP2[1,:])**2))
    #     maxNumPointsInfrontOfCam = numPointsInfrontOfCam

    # numPointsInfrontOfCam = calculatePointsInfrontOfCam(P, P3, Points3dP3)

    # if numPointsInfrontOfCam > maxNumPointsInfrontOfCam:
    #     rotation = r2
    #     translation = t
    #     Pactual = P3
    #     r = np.sqrt(np.sum((Points3dP3[0,:] - Points3dP3[1,:])**2))
    #     maxNumPointsInfrontOfCam = numPointsInfrontOfCam
    

    # numPointsInfrontOfCam = calculatePointsInfrontOfCam(P, P4, Points3dP4)

    # if numPointsInfrontOfCam > maxNumPointsInfrontOfCam:
    #     rotation = r2
    #     translation = -t
    #     Pactual = P4
    #     r = np.sqrt(np.sum((Points3dP4[0,:] - Points3dP4[1,:])**2))
    #     maxNumPointsInfrontOfCam = numPointsInfrontOfCam

    # return rotation, translation, r
    
    return rotation, translation, P, Pactual 



def compute_essential_matrix(F, K):
    E = K.T @ F @ K
    u, s, vh = np.linalg.svd(E, full_matrices=True)
    E = u @ np.diag(np.array([1,1,0])) @ vh
    return E