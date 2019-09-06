def decompose_essential_matrix(E, K):
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

    t = U[:,2]
    r1 = U @ W @ V.T
    r2 = U @ W.T @ V.T

    P =  K @ np.concatenate((np.eye(3),np.zeros(3,1)),axis=1)
    P1 = K @ np.concatenate((r1,t),axis=1)
    P2 = K @ np.concatenate((r1,-t),axis=1)
    P3 = K @ np.concatenate((r2,t),axis=1)
    P4 = K @ np.concatenate((r2,-t),axis=1)


def algebraic_triangulation(x1, x2, P1, P2):

    X = np.zeros((len(x1),4))

    for i in range(len(x1)):
        J = np.zeros((1,4))
        J[:,0] = x1[i,0] * P1[2,:] - P1[0,:]
        J[:,1] = x1[i,1] * P1[2,:] - P1[1,:]
        J[:,2] = x2[i,0] * P2[2,:] - P2[0,:]
        J[:,3] = x2[i,1] * P2[2,:] - P2[1,:]

        u, s, vh = np.linalg.svd(J, full_matrices=False)
        X[i,:] = v[3,:]
    
    return X

def compute_essential_matrix(F, K):
    E = K.T @ F @ K
    u, s, vh = np.linalg.svd(E, full_matrices=True)
    E = u @ np.diag(np.array([1,1,0])) @ vh
    return E