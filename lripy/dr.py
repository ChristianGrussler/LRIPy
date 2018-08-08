#DR  Douglas-Rachford proximal splitting algorithm with two proximal 
#    mappings as inputs 
#  
#   X,_,_,_ = DR(prox_f,prox_g,dim) runs the Douglas-Rachford splitting algorithm for the objective function 
#   f + g, where:
#       1. prox_f(Z,gamma) and prox_g(Z,gamma) are funcionts that derive the proximal mappings of 
#       gamma*f and gamma*g evalutaed in Z. 
#       2. dim is the dimension of the output of f and g. 
#   
#   Note: If f and g are convex, then X = argmin(f(X) + g(X)).
#   
#   X,Z_fix,iter,D = DR(prox_f,prox_g,dim) also returns:
#       1. Z_fix = fix point of that Douglas-Rachford iterations
#       2. iter = total number of Douglas-Rachford iterations
#       3. D = Z_fix - X, which is the solution of the dual problem if f
#       and g are convex. 
#
#   ... = DR(prox_f,prox_g,dim,option) allows to specify further options:
#       1. ... = DR(prox_f,prox_g,dim,...,rho = rho_val,...) set the step
#       length update of the fix-point update, i.e.,
#       Z_{k+1} = Z_k + rho*(Y_k - X_k), where 0 < rho < 2.
#       The default value is rho = 1.
#       2. ... = DR(prox_f,prox_g,dim,...,gamma = gamma_val,...) sets gamma 
#       to another value than the default gamma = 1.
#       3. ... = DR(prox_f,prox_g,dim,...,Z0 = Z0_val,...) sets the 
#       initial value of the fix-point iteration.
#       The default choice is Z0 = 0.
#       4. ... = DR(prox_f,prox_g,dim,...,tol=tol_val,...) 
#       sets the tolerance for zero entries as well as to stop the 
#       iteratrions once norm(Y_k-X_k,'fro') < tol
#       The default tol-value is sqrt(eps).

def dr(prox_f,prox_g,dim,gamma = 1,rho = 1,Z0 = None, tol = None):

    import numpy as np
    

    # Standard value for Z0 and tol
    if Z0 is None:
        Z = np.zeros(dim)
    else:
        Z = Z0
        
    if tol is None:
        tol = np.sqrt(np.finfo(float).eps)

    # Initialize iteration counter
    iter = 0

    # Initialize X, Y and error between X and Y   
    X = np.zeros(dim)
    Y = np.zeros(dim)
    err_XY = tol+1 # Larger than the tol so that loop starts

    # Set step-size for printing err_XY
    err_bound = 1e-1 # Error bound step-size for display 

    # Compute the Douglas-Rachford iterations

    while err_XY >= tol:

        iter += 1 # Increase counter
                    
        # Display error between X_iter and Y_iter
        if err_XY <= err_bound:
            print('Error between X_'+str(iter)+' and '+'Y_'+str(iter)+' <= '+str(err_bound))
            err_bound /= 10

        ## Compute Douglas-Rachford steps 
        X = prox_f(Z=Z,gamma=gamma)  
        # Check if X is tuple in case of multiple outputs
        if isinstance(X, tuple):
            X = X[0]
        # Check if Y is tuple in case of multiple outputs
        Y = prox_g(Z=2*X-Z,gamma=gamma)
        if isinstance(Y, tuple):
            Y = Y[0]
        Z = Z+rho*(Y-X)
                
        # Update iteration error ||X_k - Y_k||_F
        err_XY = np.linalg.norm(X-Y,ord = 'fro')         

    # Set the final solution
    X = Y

    # Compute dual variable D
    D = (Z-X)/gamma
    Z[(np.absolute(Z) < 0)] = 0
    
    # Set fix point
    if iter > 0:
        Z_fix = Z
    else:
        # Set fix point if iter = 0 and D = 0
        Z_fix = X

    X[np.absolute(X) < tol] = 0

    return X,Z_fix,iter,D