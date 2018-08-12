def drhankelapprox(H,r,solver = None, gamma = 1,rho = 1,Z0 = None,tol = None):
    """
    Douglas-Rachford proximal splitting for low-rank 
    approximation with Hankel constraint through the low-rank inducing 
    Frobenius norm and non-convex Douglas-Rachford.
            
    M,rankM,e,D,Z_fix,iter = DRHANKELAPPROX(H,r) determines a Frobenius norm 
    low-rank Hankel approximation M through the Frobenius norm low-rank
    inducing norm and Douglas-Rachford splitting, i.e., M is the solution 
    to
    
    minimize 0.5*||H||_{ell_2}^2 - trace(M'H) + 0.5*||M||_{ell_2,r*}^2 
            s.t. M is Hankel
    
    Further,
        1. rankM = rank(M)
        2. err = norm(H-M,'fro')
        3. D = solution of the dual problem
        4. Z_fix = fix point of the Douglas-Rachford iterations
        5. iter = total number of Douglas-Rachford iterations

    ... = DRHANKELAPPROX(H,r,option) allows to specify further options:
        1. ... = DRHANKELAPPROX(H,r,...,solver = 'NDR',...) changes solver 
        to use the non-convex Douglas-Rachford.
        2. ... = DRHANKELAPPROX(H,r,...,gamma = gamma_val,...) multiplies the
        objective functions with gamma_val when determining the prox of them.
        Default value: gamma_val = 1.
        3. ... = DRHANKELAPPROX(H,r,...,rho = rho_val,...) set the step length
        update of the fix-point update, i.e. 
        Z_{k+1} = Z_k + rho_val*(Y_k - X_k), where 0 < rho_val < 2.
        Default value: rho_val = 1.
        4. ... = DRHANKELAPPROX(H,r,...,Z0 = Z0_val,...) sets the initial value of
        the fix-point iteration, i.e., Z_{0} = Z0_val. 
        Default choice: Z0 = zeros(H.shape).
        5. ... = DRHANKELAPPROX(H,r,...,tol = tol_val,...) sets the relative
        tolerance of:
            + The numerical rank: rankM = rank(M/norm(H,'fro'),tol_val)
            + Iterations stop: Stop if (Y_k -X_k)/norm(H,'fro') < tol_val
            + Zero values: E.g., D(abs(D/norm(H,'fro')) < tol_val) = 0
        Default value: tol_val = sqrt(eps), where eps denotes the machine precision constant.

    ---------------
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
    from lripy.projhankel import projhankel
    from lripy.proxnonconv_square import proxnonconv_square
    from lripy.proxnormrast_square import proxnormrast_square
    from lripy.dr import dr

    # Define default tolerance value
    if tol is None:
        tol = np.sqrt(np.finfo(float).eps)

    dim = H.shape

    if r > min(dim):
        ValueError('r is larger than min(H.shape)')

    # Define absolute tolerance
    tol_H = np.linalg.norm(H)*tol 

    ## Start Douglas-Rachford iterations
    # Choose between convex and non-convex Douglas-Rachford
    if  solver == 'NDR': 
        M,Z_fix,iter,D = dr(partial(proxnonconv_square,r=r,p=2),partial(projhankel,H = -H),dim,tol =tol_H,gamma = gamma,Z0 = Z0,rho = rho)       
    else:
        M,Z_fix,iter,D = dr(partial(proxnormrast_square,r=r,p=2),partial(projhankel,H = -H),dim,tol =tol_H,gamma = gamma,Z0 = Z0,rho = rho)

    # Compute error and rank of the approximation
    err = np.linalg.norm(M-H)
    rankM = np.linalg.matrix_rank(M,tol_H)

    return M,rankM,err,D,Z_fix,iter