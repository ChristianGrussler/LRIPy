# Example: Matrix completion by low-rank Frobenius-norm mimization
# N = svd_5(H)
# Known entries: positive elements in N
# Unkown entries: non-positive elements in N
# Compare results of DR and NDR to illustrate their handlings
#
#############
# References: 
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
#%%
import numpy as np
from scipy.linalg import hankel
import time
from lrinorm import drcomplete

H = hankel(np.ones(10))
r = 5
U,S,V = np.linalg.svd(H,full_matrices=False)
N = U[:,0:r].dot(np.diag(S[0:r])).dot(V[0:r,:])
dim = N.shape
# Define known entries
Index = (N > 0)

# Intialize Douglas-Rachford iterations in the same point: Z0 = 0
Z0_val = np.zeros(dim)
# Set iteration tolerance smaller than default
tol_val = 1e-9

## Compute the different solutions:

# Douglas-Rachford solving the convexified problem
tic = time.clock()
M_dr,rankM_dr,err_dr,D_dr,Z_fix_dr,iter_dr = drcomplete(N,Index,r,2,Z0=Z0_val,tol=tol_val)
toc = time.clock()
t_dr = toc - tic

# Non-convex Douglas-Rachford
tic = time.clock()
M_ndr,rankM_ndr,err_ndr,D_ndr,Z_fix_ndr,iter_ndr = drcomplete(N,Index,r,2,solver='NDR',Z0=Z0_val,tol=tol_val)
toc = time.clock()
t_ndr = toc-tic 

## Display summary

print('\n---Rank of the solutions---')
print('Dougals-Rachford: '+str(rankM_dr))
print('Non-convex Dougals-Rachford: '+str(rankM_ndr))

print('\n---Relative Erros of the solutions---')
print('Dougals-Rachford: ' + str(err_dr/np.linalg.norm(N)))
print('Non-convex Dougals-Rachford: '+str(err_ndr/np.linalg.norm(N)))

print('\n---Elapse time of solvers---')
print('Dougals-Rachford: '+str(t_dr)+' sec')
print('Non-convex Dougals-Rachford: '+str(t_ndr)+' sec')


