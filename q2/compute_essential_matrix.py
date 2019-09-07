import numpy as np 

def compute_essential_matrix(F, K):
    E = K.T @ F @ K
    u, s, vh = np.linalg.svd(E, full_matrices=True)
    E = u @ np.diag(np.array([1,1,0])) @ vh
    return E
