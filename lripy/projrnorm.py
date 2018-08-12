def projrnorm(z,zv,r,p,gamma, search = {"t": 1, "s": 1, "k": 1}, init ={"t": 1, "s": 0,"k": 1},tol = 1e-12):
    """
    Projection onto the unit-ball or negative epi-graph of the 
    scaled truncated ell_2 norm and ell_1 norms for integer-valued r.

    y,w,final = PROJRNORM(z,zv,r,p,gamma) computes the projection of z onto 
    a) the unit-ball of the truncated ell_p norm/gamma: if zv = None.
    b) the negative epi-graph of the truncated ell_p norm/gamma: if zv is real.
    This means that y is the solution to the optimization problems: 

        a) minimize_X ||y-z||_{ell_2} s.t. ||y||_{ell_p,r}/gamma <= 1,

        b) minimize_X ||y-z||_{ell_2}+(w-zv)^2 s.t. ||y||_{ell_p,r}/gamma <= -w,

    where p=1 or p=2 and r is an integer. If zv = None, then w = None.
    c) final.t and final.s are the final values of parameters to the two 
    nested search. For p = 1, final.k is the final value of the third inner 
    search parameter. Note,
    if zv = None and ||z||_{ell_p,r}/gamma <= 1, then t = s = k = None;
    if zv is real and ||z||_{ell_p,r}/gamma <= -zv, then t = s = k = None;
    Furthmore, k=None if the local iteration lies within the polar cone.

    ... = PROJRNORM(z,zv,r,p,gamma,option) allows us to specify furhter options:
        1. ... = PROJRNORM(z,zv,r,p,...,tol = tol_val) sets the relative
        tolerance of the deciding about zeros e.g.
        For all i: y_i = 0 if |y_i| <= tol*||y||_{ell_p,r). 
        Default value: tol_val = 1e-12.
        2. ... = PROJRNORM(z,zv,r,p,gamma,...,search = search_val) changes 
        from default binary search to linear search (i.e. incremental
        increase of s and t) over
        a) t if search_val["t"] = 0.
        b) s if search_val["s"] = 0.
        c) k if search_val["k"] = 0 (only for p =1).
        3. ... = PROJRNORM(z,zv,r,p,gamma,...,init = init_val) changes 
        from default binary search start values t_0 = 1, k_0 = 1, s_0 = 0
        to 
        a) t_0 if init_val["t"] = t_0.
        b) s_0 if init_val["s"] = s_0.
        c) k_0 if init_val["k"] = k_0 (only for p =1).

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
    
    n = np.max(z.shape)
    
    #Intialize final values for t,s,k
    final = {"t": None, "s": None}
    
    # Set zv = -1 for projection onto unit-ball and flag this case

    if zv is None:
        zv = -1 
        mode_ball = 1 # Flag as ball case
    else:
        mode_ball = 0 # Flag as epi graph case

    # Check what parameters have not been defined
    if search.get("t") is None:
        search["t"] = 1
    if search.get("s") is None:
        search["s"] = 1
    if search.get("k") is None:
        search["k"] = 1

    if init.get("t") is None:
        init["t"] = 1
    if search.get("s") is None:
        init["s"] = 1
    if search.get("k") is None:
        init["k"] = 1

    # Intialize y
    y = np.zeros(n)

    # Store original sign of z.
    I_sign = (z < 0)

    # Replace z by sorted abs(z) and store orgininal sorting in Io.
    Io = np.argsort(abs(z), axis=-1, kind='mergesort')[::-1]
    z  = abs(z)[Io]
    
        
    
    # Pre-compute necessary sums dependend on t
    if p == 2:

        # Compute and save sum(z[0:i]^2) for all i <=r
        sum_z_square = np.zeros(r)
        sum_z_square[0] = np.square(z[0])
        for i in range(1,r):
            sum_z_square[i] = sum_z_square[i-1]+np.square(z[i])
            
        # Relative toleranz for the value 0
        tol_rel = np.sqrt(sum_z_square[r-1])*tol 

        # Check if (z,zv) is a solution
        if np.sqrt(sum_z_square[r-1])/gamma + zv <= tol_rel: 
            y = z
            w = zv
            run_search = 0 # Flag: no search needed 
        else:
            run_search = 1 # Flag to search

    elif p == 1:
    
        # Compute sum(z[r-t:r]) for all 1 <= t <= r
        sum_z_t = np.zeros(r)
        sum_z_t[0] = z[r-1] # t == 1

        for i in range(1,r):
            sum_z_t[i] = sum_z_t[i-1]+z[r-i-1]
            
        # Relative toleranz for the value 0
        tol_rel = sum_z_t[r-1]*tol
        
        # Check if (z,zv) is a solution
        if sum_z_t[r-1]/gamma+zv <= tol_rel: 
            y = z
            w = zv
            final["t"] = None
            final["s"] = None
            final["k"] = None
            run_search = 0 # Flag: no search needed 
        else:
            run_search = 1 # Flag to search

    if run_search == 1:

        # Initialize binary search summation for computing \tilde{z}_{r-t+1} based on
        # changes in s (and t for p == 2)
        if search["t"] == 1 and p == 2:
            sum_z_t = np.zeros(r)
            t_old = 1 # largest summation index of all previous iterations
            sum_z_t[0] = z[r-1]

        
        if search["s"] == 1:
            s_old = 1 # largest summaton index of all previous iterations
            sum_z_s = np.zeros(n-r+2) # sum z[r:i] for i=r+1,...,s
            if r+1<=n:
                sum_z_s[1] = z[r]


        ## Search for t
        if search["t"] == 0:
            init["t"] = 1
            sum_zt = z[r-1] #sum_zt for linear search
            
        t = init["t"]
        indt = 0

        t_min = 1 # Lower limit on t
        t_max = r # Upper limit on t


        while indt == 0:

            if t_min == t_max:
                indt = 1
                t = t_min

            if p == 2:
                if search["t"] == 0:
                    if t_min != t_max and t > init["t"]: #sum_zt was already updated in the iteration before
                        sum_zt = sum_zt + z[r-t]
                    
                else:
                    # Binary search summation for sum_zt
                    if t_old < t:
                        for i in range(t_old,t):
                            sum_z_t[i] = sum_z_t[i-1] + z[r-i-1]
                    t_old = t
                    sum_zt = sum_z_t[t-1]
            elif p == 1:
                sum_zt = sum_z_t[t-1]

            ## Binary search for s
            if search["s"] == 0:
                init["s"] = 0
                sum_zs = 0


            s_min = 0
            s_max = n-r
            s = init["s"]
            inds = 0


            while inds == 0:

                if s_min == s_max:
                    inds = 1
                    s = s_max

                ## Summation style for sum_z_s 
                if search["s"] == 0: # Incremental summation for sum_z_s
                    if s_max != s_min and s>init["s"]: # Only sums from r+1,...,s and sum_zs has not already been updated in the previous iteration
                        sum_zs = sum_zs + z[r-1+s]
                else: # Binary search summatoin
                    if s_old < s:
                        for i in range(s_old+1,s+1):
                            sum_z_s[i] = sum_z_s[i-1] + z[r+i-1]

                        s_old = s

                    sum_zs = sum_z_s[s]

                # Determine c_2 for p = 2 and \tilde{z}_{r-t+1} for both cases
                c_2 = (sum_zt + sum_zs)
                z_tilde = c_2/np.sqrt(t+s) #\tilde{z}_{r-t+1}

                # Determine mu    
                if p == 2:
                    c_1 = 0
                    if r-t > 0:
                        c_1 = sum_z_square[r-t-1]

                    #Check if mu > 0, i.e. if (\tilde{z},z_v) does not fulfil the constraints
                    if np.sqrt((c_1+t/(s+t)*np.square(z_tilde)))/gamma + zv <= tol_rel: #Check: (\tilde{z},z_v) in cone
                        mu = 0
                        is_polar = 0 # Flag not polar cone case
                    elif np.sqrt((c_1+(s+t)/t*np.square(z_tilde)))*gamma - zv <= tol_rel: #Check: (\tilde{z},z_v) in polar cone
                        is_polar = 1 # Flag that in polar cone
                    else:
                        is_polar = 0 # Flag not polar cone case
                        if mode_ball == 1:
                            # Polynomial for projection onto unit-ball
                            polynom = np.array([np.square(t/np.square(gamma)), 2*np.square(t/gamma) + (2*t*(s + t))/np.square(gamma), (4*t*(s + t) + np.square(s + t) - (np.square(c_2)*t)/np.square(gamma) - (np.square(t)*(- np.square(gamma) + c_1))/np.square(gamma)), (2*np.square(gamma*(s + t)) - 2*np.square(c_2)*t - 2*t*(s + t)*(- np.square(gamma) + c_1)), - np.square(c_2*gamma)*t - np.square(gamma*(s + t))*(- np.square(gamma) + c_1)])
                            
                        else:
                            # Polynomial for projection onto epi-graph
                            polynom = np.array([np.square((gamma*(s + t) + t/gamma)*(gamma + 1/gamma)), (- 2*gamma*zv*np.square(gamma*(s + t) + t/gamma)*(gamma + 1/gamma) - 2*gamma*zv*(gamma*(s + t) + t/gamma)*(s + t)*np.square(gamma + 1/gamma)), (np.square(gamma*zv*(s + t)*(gamma + 1/gamma)) - t*np.square((gamma + 1/gamma)*c_2) - np.square(gamma*(s + t) + t/gamma)*(- np.square(gamma*zv) + c_1) + 4*np.square(gamma*zv)*(gamma*(s + t) + t/gamma)*(s + t)*(gamma + 1/gamma)), (2*np.square(c_2)*gamma*t*zv*(gamma + 1/gamma) - 2*np.power(gamma*zv,3)*np.square(s + t)*(gamma + 1/gamma) + 2*gamma*zv*(gamma*(s + t) + t/gamma)*(s + t)*(- np.square(gamma*zv) + c_1)), - np.square(c_2*gamma*zv)*t - np.square(gamma*zv*(s + t))*(- np.square(gamma*zv) + c_1)])

                        R = np.roots(polynom)
                        I = (np.abs(np.imag(R)) <= tol_rel)
                        mu = np.real(np.max(R[I]))

                if p == 1: 

                # Check if mu > 0 for p==1:
                    if (sum_z_t[r-1]-sum_z_t[t-1]+t/np.sqrt(s+t)*z_tilde)/gamma + zv <= tol_rel: #Check: (\tilde{z},z_v) in cone
                        mu = 0
                        k = None
                        is_polar = 0 # Flag that not in polar cone
                    elif np.max(np.array([z[0],np.sqrt(t+s)/t*z_tilde]))*gamma - zv <= tol_rel: # Check if (\tilde{z},z_v) in polar cone
                        is_polar = 1 # Flag that in polar cone
                        k = None
                    else:
                        is_polar = 0 # Flag that not in polar cone

                        ## Do search for k

                        # Sort according to the break points:

                        # Define z_hat
                        z_hat = np.zeros(r)
                        z_hat[0:r-t] = z[0:r-t] #\tilde{z}_{1:r-t} = z_{1:r-t}
                        z_hat[r-t] = t/(np.sqrt(t+s))*z_tilde

                        # Break points
                        z_break = z_hat.copy()
                        z_break[r-t] = (np.sqrt(t+s)/t)*z_tilde

                        I_alpha = np.argsort(z_break[0:r-t+1], axis=-1, kind='mergesort')[::-1]

                        z_hat[0:r-t+1] = z_hat[I_alpha]
                        z_hat = z_hat/gamma

                        alpha = np.ones(r)
                        alpha[r-t] = np.square(t)/(t+s)
                        alpha = alpha/np.square(gamma)
                        alpha[0:r-t+1] = alpha[I_alpha]

                        ## Binary search for k

                        # Initialize binary search variables
                        if search["k"] == 0:
                            init["k"] = 1
                            sum_z_hat = z_hat[0]
                            sum_alpha = alpha[0]
                        else:
                            # Intialize binary search summation of z_hat
                            sum_z_hat_k = np.zeros(r)
                            sum_z_hat_k[0] = z_hat[0]

                            # Intialize binary search summation for alpha
                            sum_alpha_k = np.zeros(r)
                            sum_alpha_k[0] = alpha[0]

                        indk = 0
                        k_min = 1
                        k_max = r-t+1
                        k = init["k"]
                        k_old = 1


                        while indk == 0:

                            if k_min == k_max:
                                indk = 1
                                k = k_max


                            # Search summation 
                            if search["k"] == 0: # Incremental search
                                if k_max != k_min and k>init["k"]:
                                    sum_z_hat = sum_z_hat + z_hat[k-1]
                                    sum_alpha = sum_alpha + alpha[k-1]

                            else: # Binary search summation for sum_z_hat_k and sum_alpha
                                if k_old < k:   
                                    for i in range(k_old,k):
                                        sum_z_hat_k[i] = sum_z_hat_k[i-1]+z_hat[i]
                                        sum_alpha_k[i] = sum_alpha_k[i-1]+alpha[i]

                                    k_old = k

                                sum_alpha = sum_alpha_k[k-1]
                                sum_z_hat = sum_z_hat_k[k-1]


                            if mode_ball == 1:
                                mu = (sum_z_hat+zv)/(sum_alpha)
                            else:
                                mu = (sum_z_hat+zv)/(1+sum_alpha)
                            

                            # Search rules for k
                            if (z_hat[k-1] - alpha[k-1]*mu) >= -tol_rel:
                                if mu >= -tol_rel:
                                    if search["k"] == 0:
                                        k_min = k
                                        k = k_min+1
                                    else:
                                        k_min = k
                                        k = k_min+int(np.ceil((k_max-k_min)/2)) #ceil makes sure to not stay

                                else:
                                    k_min = k+1
                                    if search["k"] == 0:
                                        k = k_min
                                    else:
                                        k = k_min+int(np.floor((k_max-k_min)/2)) #floor makes sure not to jump too far


                            else:
                                if search["k"] == 0:
                                    k_min = k_max
                                else:
                                    k_max = k-1
                                    k = k_max-int(np.floor((k_max-k_min)/2)) #floor makes sure to not jump too far, because k_max = k-1

                # Determine w^{(s,t)}
                if is_polar == 1:
                    w = 0
                elif mode_ball == 1:
                    w = -1
                else:
                    w = zv-mu
            

                # Determine [y_{r+s}^(s,t),y_{r+s+1}^(s,t)] = [y_tilde_{r-t+1}/sqrt{s+t},z_{r+s+1}].

                if p == 2:
                    if is_polar == 1:
                        y[r+s-1] = 0
                    else:
                        y[r+s-1] = (z_tilde/(1-mu*t/(np.square(gamma)*w*(s+t))))/np.sqrt(s+t)

                elif p == 1:
                    if is_polar == 1:
                        y[r+s-1] = 0
                    else:
                        y[r+s-1] = np.max(np.array([z_tilde-t*mu/(gamma*np.sqrt(t+s)),0]))/np.sqrt(s+t)


                if r+s < n:
                    if is_polar == 1:
                        y[r+s] = 0
                    else:
                        y[r+s] = z[r+s]
                        
                #print("t: "+str(t)+"s: "+str(s))
                #pdb.set_trace()
                # Search rules for s
                if r+s < n:
                    if y[r+s-1] - y[r+s] >= -tol_rel:
                        if search["s"] == 0:
                            s_max = s_min
                        else:
                            s_max = s
                            s = s_max - int(np.ceil((s_max-s_min)/2))
                    else:
                        s_min = s+1
                        
                        if search["s"] == 0:
                            s = s_min
                        else:
                            s = s_min + int(np.floor((s_max-s_min)/2))
                else:
                    if search["s"] == 0:
                        s_min = s_max
                    else:
                        s_max = s
                        s = s_max - int(np.ceil((s_max-s_min)/2))
                

            # Determine [y_{r-t}^(s,t), y_{r-t+1}^(s,t)] = [y_tilde_{r-t},y_tilde_{r-t+1}/sqrt{s+t}]
        
            if r-t>0: # In case r = t there is nothing to do
                if p == 2:
                    if is_polar == 1:
                        y[r-t-1] = 0
                    else:
                        y[r-t-1] = z[r-t-1]/(1-mu/(np.square(gamma)*w))
                elif p == 1:
                    if is_polar == 1:
                        y[r-t-1] = 0
                    else:
                        y[r-t-1] = np.max(np.array([z[r-t-1]-mu/gamma,0]))
      
            y[r-t] = y[r+s-1]

            # Search rules for t
            if r-t > 0:

                if y[r-t-1] - y[r-t] >= -tol_rel:
                    if search["t"] ==0:
                        t_max = t_min #Tried all smaller t => t is optimal
                    else:
                        t_max = t
                        t = t_max - int(np.ceil((t_max-t_min)/2)) #t may be too large
                else: 
                    t_min = t+1

                    if search["t"] == 0:
                        t = t_min # Tried all previous ones => Only solution reached
                    else:
                        t = t_min + int(np.floor((t_max-t_min)/2)) # t may be too large
            else:
                if search["t"] == 0:
                    t_min = t_max
                else:
                    t_max = t
                    t = t_max - int(np.ceil((t_max-t_min)/2)) 
        

        ## Get full solution
        if p == 2:
            if is_polar == 1:
                y[:] = 0
            else:
                
                
                y[0:r-t] = z[0:r-t]/(1-mu/(np.square(gamma)*w))
                y[r-t:r+s] = (z_tilde/(1-mu*t/(np.square(gamma)*w*(s+t))))/np.sqrt(s+t)
                y[r+s:] = z[r+s:]
        elif p == 1:
            if is_polar == 1:
                y[:] = 0
            else:
                y[0:r-t] = np.max(np.append(z[0:r-t]-mu/gamma,0))
                y[r-t:r+s] = np.max(z_tilde-t*mu/(gamma*np.sqrt(t+s)),0)/np.sqrt(s+t)
                y[r+s:] = z[r+s:]

        final["t"] = t
        final["s"] = s
        if p == 1:
            final["k"] = k

    y[y<tol_rel] = 0


    Iback = np.argsort(Io, axis=-1, kind='mergesort')
    y = y[Iback]

    y[I_sign] = -y[I_sign]

    if mode_ball == 1:
         w = None
            
    return y,w,final