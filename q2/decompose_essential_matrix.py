import numpy as np



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
    
    return X

def calculatePointsInfrontOfCam(P,P2, points3D):
    
    
    PointsInfrontOfCam = 0

    for i in range(len(points3D)):
        if points3D[i,2] > 0:
            PointsInfrontOfCam = PointsInfrontOfCam + 1

    return PointsInfrontOfCam

    

def decompose_essential_matrix(E, K, img_points1, img_points2):
    w = np.array([
        [ 0, -1, 0],
        [ 1,  0, 0],
        [ 0,  0, 1],
    ])

    z = np.array([
        [ 0, 1, 0],
        [-1, 0, 0],
        [ 0, 0, 0],
    ])

    u, s, vh = np.linalg.svd(E, full_matrices=True)

    t = u[:,2]
    r1 = u @ w @ vh.T
    r2 = u @ w.T @ vh.T

    P =  K @ np.concatenate((np.eye(3),np.zeros((3,1))),axis = 1)
    P1 = K @ np.concatenate((r1,t),axis=1)
    P2 = K @ np.concatenate((r1,-t),axis=1)
    P3 = K @ np.concatenate((r2,t),axis=1)
    P4 = K @ np.concatenate((r2,-t),axis=1)


    Points3dP1 = algebraic_triangulation(img_points1, img_points2, P, P1)
    Points3dP2 = algebraic_triangulation(img_points1, img_points2, P, P2)
    Points3dP3 = algebraic_triangulation(img_points1, img_points2, P, P3)
    Points3dP4 = algebraic_triangulation(img_points1, img_points2, P, P4)


    numPointsInfrontOfCam = calculatePointsInfrontOfCam(P, P1, Points3dP1)

    maxNumPointsInfrontOfCam = numPointsInfrontOfCam
    rotation = r1
    translation = t
    Pactual = P1
    r = np.sqrt(np.sum((Points3dP1[0,:] - Points3dP1[1,:])**2))
    

    numPointsInfrontOfCam = calculatePointsInfrontOfCam(P, P2, Points3dP2)

    if numPointsInfrontOfCam > maxNumPointsInfrontOfCam:
        rotation = r1
        translation = -t
        Pactual = P2
        r = np.sqrt(np.sum((Points3dP2[0,:] - Points3dP2[1,:])**2))

    numPointsInfrontOfCam = calculatePointsInfrontOfCam(P, P3, Points3dP3)

    if numPointsInfrontOfCam > maxNumPointsInfrontOfCam:
        rotation = r2
        translation = t
        Pactual = P3
        r = np.sqrt(np.sum((Points3dP3[0,:] - Points3dP3[1,:])**2))
    

    numPointsInfrontOfCam = calculatePointsInfrontOfCam(P, P4, Points3dP4)

    if numPointsInfrontOfCam > maxNumPointsInfrontOfCam:
        rotation = r2
        translation = -t
        Pactual = P4
        r = np.sqrt(np.sum((Points3dP4[0,:] - Points3dP4[1,:])**2))

    return rotation, translation, r



def compute_essential_matrix(F, K):
    E = K.T @ F @ K
    u, s, vh = np.linalg.svd(E, full_matrices=True)
    E = u @ np.diag(np.array([1,1,0])) @ vh
    return E