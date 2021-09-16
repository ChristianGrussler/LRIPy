# Example: Low-rank approximation with Hankel constraint low-rank inducing Frobenius norm
# Compare results of DR and NDR to illustrate their handlings
#
#############
# References: 
#   - C. Grussler and P. Giselsson (2017):
#   "Local convergence of proximal splitting methods for rank constrained
#   problems", pp. 702-708, IEEE 56th Annual Conference on Decision and Control
#   (CDC), DOI: 10.1109/CDC.2017.8263743.
#
#   - C. Grussler and P. Giselsson (2018):
#   "Low-Rank Inducing Norms With Optimality Interpreations", 
#   SIAM J. Optim., 28(4), pp. 3057–3078. 
#
#   - C. Grussler (2017):
#   "Rank reduction with convex constraints", PhD Thesis, 
#   Department of Automatic Control, Lund Institute of Technology, 
#   Lund University, ISBN 978-91-7753-081-7.  
#############
#%%
import numpy as np
from scipy.linalg import hankel
import time
from lripy import drhankelapprox

H = hankel(np.arange(1,11,1),np.arange(10,0,-1)) # Hankel matrix
r = 5 # Desired rank of the approximation
dim = H.shape
## Compute the different solutions:

# Set tolerance for deciding about multiple singular values
tol_val = 1e-10

# Low-rank inducing Frobenius norm with Douglas-Rachford
tic = time.perf_counter()
M_dr,rankM_dr,err_dr,D_dr,Z_fix_dr,iter_dr = drhankelapprox(H,r,tol=tol_val)
toc = time.perf_counter()
t_dr = toc - tic

# Non-convex Douglas-Rachford
tic = time.perf_counter()
M_ndr,rankM_ndr,err_ndr,D_ndr,Z_fix_ndr,iter_ndr = drhankelapprox(H,r,solver = 'NDR',tol=tol_val)
toc = time.perf_counter()
t_ndr = toc - tic

# Display summary
print('\n---Rank of the solutions---')
print('Dougals-Rachford: '+str(rankM_dr))
print('Non-convex Dougals-Rachford: '+str(rankM_ndr))

print('\n---Relative Erros of the solutions---')
print('Dougals-Rachford: ' + str(err_dr/np.linalg.norm(H)))
print('Non-convex Dougals-Rachford: '+str(err_ndr/np.linalg.norm(H)))

print('\n---Elapse time of solvers---')
print('Dougals-Rachford: '+str(t_dr)+' sec')
print('Non-convex Dougals-Rachford: '+str(t_ndr)+' sec')
