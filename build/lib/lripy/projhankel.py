#PROJHANKEL Orthogonal projection onto the subspace of Hankel matrices
#
#   X = PROJHANKEL(Z) determines the orthogonal projection of 
#   Z onto the subspaces of Hankel matrices, i.e., X is the 
#   proximal mapping of (i_{Hankel}(X)), where i_{Hankel} 
#   is the indicator function of the set of Hankel matrices.
#
#  X = PROJHANKEL(Z,H,gamma) allows to shift Z by -gamma*H, i.e., 
#  X is projection of Z-gamma*H onto the subspaces of Hankel matrices and is
#  therefore the proximal mapping of gamma*(i_{Hankel}(X)+trace(X'H)).

def projhankel(Z,H = 0,gamma = 0):
    
    import numpy as np
    from scipy.linalg import hankel 
    
    Z = Z - gamma*H
    dim = Z.shape
    if dim[0] < dim[1]:
        Z = Z.T
        transp_Z = 1 # Flag transpositon of Z
        dim = np.sort(dim,axis = -1)[::-1]
    else:
        transp_Z = 0 # Flag Z is not transposed
    
    n = dim[0]
    m = dim[1]

    # Determine the means k[0],...,k[n+m-2] along the anti-diagonals
    B = np.zeros((m+n,m))
    B[0:n,:] = Z
    N = np.sum(np.reshape(B.flatten('F')[0:(n+m-1)*m],(n+m-1,m),order='F'),axis = 1)
    D = np.append(np.append(np.arange(1,m), m*np.ones((1,n-m+1))),np.arange(1,m)[::-1])
    k = np.divide(N,D)
    # Arrange means as Hankel matrix
    X = hankel(k[0:n],k[n-1:])

    if transp_Z == 1:
        X = X.T
    
    return X