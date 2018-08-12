def drcomplete(N,Index,r,p,solver = None, gamma = 1,rho = 1,Z0 = None,tol = None):
    """
    Douglas-Rachford proximal splitting for low-rank 
    completion through the low-rank inducing Frobenius/spectral norm and 
    non-convex Douglas-Rachford for integer-valued r.
        
    M,rankM,err,D,Z_fix,iter = DRCOMPLETE(N,Index,r,p) determines a low-rank matrix completion
    solution M though the low-rank inducing Frobenius/spectral norm, i.e., M
    is a solution to 

    minimize ||M||_{ell_p,r*}
        s.t.   M(Index) = N(Index)
    for p = 2 or p = 'inf', where Index is the logical incident matrix of the 
    known entries in N. Therefore, N should contain the correct known entries,
    whereas the unknowns can be arbitrary. Further,   
        1. rankM = rank(M)
        2. err = norm(N-M,'fro') (This is only to compare the performance) 
        with other methods, when N is explicitly known. 
        3. D = solution of the dual problem
        4. Z_fix = fix point of the Douglas-Rachford iterations
        5. iter = total number of Douglas-Rachford iterations

    ... = DRCOMPLETE(N,Index,r,p,option) allows to specify further options:
        1. ... = DRCOMPLETE(N,Index,r,p,...,solver = 'NDR',...) sets the 
        solver to the non-convex Douglas-Rachford.
        2. ... = DRCOMPLETE(N,Index,r,p,...,gamma = gamma_val,...) multiplies 
        the objective functions with gamma when determining the prox of them.
        The default value is set to gamma = 1.
        3. ... = DRCOMPLETE(N,Index,r,p,...,rho = rho_val,...) set the step 
        length update of the fix-point update, i.e. 
        Z_{k+1} = Z_k + rho_val*(Y_k - X_k), where 0 < rho_val < 2.
        Default value: rho_val = 1.
        4. ... = DRCOMPLETE(N,Index,r,p,...,Z0 = Z0_val,...) sets the initial 
        value of the fix-point iteration, i.e. Z_0 = Z0_val. 
        Default choice: Z0 = randn(size(N)).
        5. ... = DRCOMPLETE(N,Index,r,p,...,tol=tol_val,...) sets the relative
        tolerance of:
            + The numerical rank: rankM = rank(M/norm(N0,'fro'),tol_val)
            + Iterations stop: Stop if (Y_k -X_k)/norm(N0,'fro') < tol_val
            + Zero values: E.g., D(abs(D/norm(N0,'fro')) < tol_val) = 0
        where N0[Index] = N[Index] and zero otherwise. 
        Default value: tol_val = sqrt(eps), where eps denotes the machine precision constant.

    --------------
    References:
    - C. Grussler and A. Rantzer and P. Giselsson (2018): 
    "Low-Rank Optimization with Convex Constraints", 
    IEEE Transactions on Automatic Control, DOI: 10.1109/TAC.2018.2813009.

    - C. Grussler and P. Giselsson (2016):
    "Low-Rank Inducing Norms With Optimality Interpreations", 
    arXiv:1612.03186v1.

    - C. Grussler and P. Giselsson (2017):
    "Local convergence of proximal splitting methods for rank constrained
    problems", pp. 702-708, IEEE 56th Annual Conference on Decision and Control
    (CDC), DOI: 10.1109/CDC.2017.8263743.

    - C. Grussler (2017):
    "Rank reduction with convex constraints", PhD Thesis, 
    Department of Automatic Control, Lund Institute of Technology, 
    Lund University, ISBN 978-91-7753-081-7.
    """
    
    import numpy as np
    from functools import partial
    from lripy.projindex import projindex
    from lripy.proxnonconv import proxnonconv
    from lripy.proxnormrast import proxnormrast
    from lripy.dr import dr

    # Define default tolerance value
    if tol is None:
        tol = np.sqrt(np.finfo(float).eps)
    
    dim = N.shape # Dimension of the problem 
    
    # Define Z0 for DR iteration
    if Z0 is None:
        Z0 = np.random.randn(dim[0],dim[1])
    
    if r > min(dim):
        ValueError('r is larger than min(N.shape)')

    # Define matrix with known entries of N and zeros otherwise
    N0 = np.zeros(dim)
    N0[Index] = N[Index]

    # Set absolut tolerance 
    tol_N0 = np.linalg.norm(N0)*tol

    ## Start Douglas-Rachford iterations

    # Choose between convex and non-convex Douglas-Rachford
    if  solver == 'NDR':
        M,Z_fix,iter,D = dr(partial(proxnonconv,r=r,p=p),partial(projindex,N = N,Index = Index),dim,tol=tol_N0,gamma = gamma,Z0 = Z0,rho = rho)
        
    else:
        M,Z_fix,iter,D = dr(partial(proxnormrast,r=r,p=p),partial(projindex,N = N,Index = Index),dim,tol=tol_N0,gamma = gamma,Z0 = Z0,rho = rho)
    

    # Compute error and rank of the approximation
    err = np.linalg.norm(M-N)
    rankM = np.linalg.matrix_rank(M,tol_N0)
    
    return M,rankM,err,D,Z_fix,iter