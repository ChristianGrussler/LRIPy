#PROXNONCONV_SQUARE Non-convex prox of half of the squared Frobenius 
#   and spectral norm + indicator function of a matrix with at most rank r.
#  
#   X = PROXNONCONV_SQUARE(Z,r,p,gamma) determines the prox of 
#   0.5*gamma*||X||_{\ell_p}^2+i_{rank(X)<=r}(X), where i_{rank(X)<=r} is the
#   idicator function of all matrices with at most rank r. This is, X is a 
#   solution to the optimization problem: 
#       
#       minimize_X 0.5*gamma*||X||_{\ell_p}^2+0.5*||X-Z||_{\ell_2}^2
#             s.t. rank(X) <= r
#
#   where p=2 or p='inf'.
#  
#   X = PROXNONCONV(Z,r,p,gamma,option) allows us to specify furhter options:
#       1. X = PROXNONCONV(Z,r,p,gamma,...,mode='vec') is used to flag that the
#       vector-valued problem is to be solved.
#       2. X = PROXNONCONV(Z,r,p,gamma,...,tol=tol_val) sets the relative
#       tolerance of the deciding about zeros e.g. if Z is matrix then
#       for all i: \sigma(X)_i = 0 if |\sigma(X)_i| <= tol_val ||Z||_{\ell_p,r). 
#       Default value: tol_val = 1e-12.
#
#############
# References:
#   - C. Grussler and A. Rantzer and P. Giselsson (2018): 
#   "Low-Rank Optimization with Convex Constraints", 
#   IEEE Transactions on Automatic Control, DOI: 10.1109/TAC.2018.2813009.
#
#   - C. Grussler and P. Giselsson (2016):
#   "Low-Rank Inducing Norms With Optimality Interpreations", 
#   arXiv:1612.03186v1.
#
#   - C. Grussler and P. Giselsson (2017):
#   "Local convergence of proximal splitting methods for rank constrained
#   problems", pp. 702-708, IEEE 56th Annual Conference on Decision and Control
#   (CDC), DOI: 10.1109/CDC.2017.8263743.
#
#   - C. Grussler (2017):
#   "Rank reduction with convex constraints", PhD Thesis, 
#   Department of Automatic Control, Lund Institute of Technology, 
#   Lund University, ISBN 978-91-7753-081-7.
#############
def proxnonconv_square(Z,r,p,gamma,mode = None,tol = 1e-12):
    
    from lripy.proxnormrast_square import proxnormrast_square
    import numpy as np
    
    ## Check if p = 2 or p = 'inf'
    if p != 2 and p !='inf':
        raise ValueError("p can only be equal to 2 or inf")

    ## Check options
    dim = Z.shape
    max_mn = np.max(dim)
    min_mn = np.min(dim)

    ## Check dimensions
    if mode == 'vec':
        if r > max_mn:
            raise ValueError("r is larger than max(Z.shape)")
    elif r > min_mn:
            raise ValueError("r is larger than min(Z.shape)")

    # Compute rank-r approximation of Z
    U,S,V = np.linalg.svd(Z,full_matrices=False)
    Z = U[:,0:r].dot(np.diag(S[0:r])).dot(V[0:r,:])
    X,_ = proxnormrast_square(Z,r,p,gamma,mode,tol = tol)

    return X