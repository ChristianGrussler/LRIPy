def projindex(Z,N,Index,H = 0,gamma = 0):
    """
    Orthogonal projection onto the subspace of known entries

    X = PROJHANKEL(Z,N,Index) determines the orthogonal projection of 
    Z onto the subspaces of matrices with entries N(Index), i.e., 
    X is the proximal mapping of the function

    i_{X(Index) = N(Index)}(X),

    where i_{X(Index) = N(Index)} is the convex indicator function 
    for the linear constraint X(Index) = N(Index).   

    X = PROJINDEX(Z,H,gamma) allows to shift Z by -gamma*H, i.e., 
    X is projection of Z-gamma*H onto the subspaces of matrices with 
    entries N(Index) and is therefore the proximal mapping of 
    gamma*(i_{X(Index) = N(Index)}(X)+trace(X'H)).
    """
    X = Z - gamma*H
    X[Index] = N[Index]    
    return X