# LRIPy
A Python3 package for rank constrained optimization by low-rank inducing norms and non-convex proximal splitting methods.

## Purpose:
Low-rank rank inducing norms and non-convex Proximal Splitting Algoriths attempt to find exact rank/cardinality-r solutions to minimization problems with convex loss functions, i.e., avoiding of regularzation heuristics. LRIPy provides Python implementations for the proximal mappings of the low-rank inducing Frobenius and Spectral norms, as well as, their epi-graph projections and non-convex counter parts.

## Literature:

### Optimization with low-rank inducing norms: 
* [Low-rank Inducing Norms with Optimality Interpretations](https://epubs.siam.org/doi/abs/10.1137/17M1115770)
* [Low-rank Optimization with Convex Constraints](https://doi.org/10.1109/TAC.2018.2813009)
* [The Use of the r* Heuristic in Covariance Completion Problems](https://doi.org/10.1109/CDC.2016.7798554)
* [Rank Reduction with Convex Constraints](https://lup.lub.lu.se/search/publication/54cb814f-59fe-4bc9-a7ef-773cbcf06889)
* [On optimal low-rank approximation of non-negative matirces](https://doi.org/10.1109/CDC.2015.7403045)

### Proximal mapping computation for low-rank inducing norms:
* [Efficient Proximal Mapping Computation for Unitarily Invariant Low-Rank Inducing Norms](https://arxiv.org/abs/1810.07570)

### Non-convex counter parts:
* [Local Convergence of Proximal Splittinge Methods for Rank Constrained Problems](https://ieeexplore.ieee.org/document/8263743)

## Installation

The easiest way to install the package is to run ``pip install lripy``. Or for the latest changes, install the package from source by either running ``python setup.py install`` or ``pip install .`` in the main folder.  

## Documentation
In the following it holds that
* for the low-rank inducing Frobenius norm: ``p = 2``
* for the low-rank inducing Spectral norm:  ``p = 'inf'``

### Examples
There are two examples in the "examples" folder:

1. Exact Matrix Completion
2. Low-rank approximation with Hankel constraint

### Optimization

LRIPy contains Douglas-Rachford splitting implementations for "Exact Matrix Completion" and "Low-rank Hankel Approximation", both with low-rank inducing norms, as well as, non-convex Douglas-Rachford splitting. It is easy to modify these functions for other constraints! 

#### Exact Matrix completion

Let N be a matrix and Index be a binary matrix of the same size, where the ones indicate the known entries N. We attempt to find a rank-r completion M:

```
# Import the Douglas-Rachford Completion function:

from lripy import drcomplete

# Low-rank inducing norms with Douglas-Rachford splitting:

M = drcomplete(N,Index,r,p)[0]

# Non-convex Douglas-Rachford splitting:

M = drcomplete(N,Index,r,p,solver = 'NDR')[0]
```

#### Low-rank Hankel Approximation

Let H be a matrix. We attempt to find a rank-r Hankel approximation M that minimizes the Frobenius norm:

```
# Import the Douglas-Rachford Hankel Approximation function:

from lripy import drhankelapprox

# Low-rank inducing norms with Douglas-Rachford splitting:

M = drhankelapprox(H,r)[0]

# Non-convex Douglas-Rachford splitting:

M = drhankelapprox(H,r,solver = 'NDR')[0]
```

### Proximal Mappings
LRIPy provides Python implemenations for the proximal mappings to the low-rank inducing Frobenius and Spectral norm as well as their epi-graph projections and non-convex counter parts. In the following, we only discuss the matrix-valued case, but notice that for the vector-valued case, i.e., sparsity inducing, it is only required to add ``mode = 'vec'`` as an input argument. 

#### Low-rank inducing Spectral and Frobenius norms: 

Proximal mapping of the low-rank inducing norms at Z with parameter r and scaling factor gamma:
```
X = proxnormrast(Z,r,p,gamma)[0]
```
#### Squared Low-rank inducing Spectral and Frobenius norms: 
Proximal mapping of the SQUARED low-rank inducing norms at Z with parameter r and scaling factor gamma:
```
X = proxnormrast_square(Z,r,p,gamma)[0]
```
#### Projection onto the epi-graph of the low-rank inducing norms: 
Projection of (Z,zv) on the epi-graph of the low-rank inducing norms with parameter r and scaling factor gamma:
```
X,xv = projrast(Z,zv,r,p,gamma)[0:2]
```

#### Non-convex proximal mappings for Frobenius and Spectral norm: 

Non-convex proximal mapping of at Z with parameter r and scaling factor gamma:
```
X = proxnonconv(Z,r,p,gamma)
```
#### Non-convex proximal mappings for squared Frobenius and Spectral norm:
Non-convex proximal mapping for the SQUARED norms at Z with parameter r and scaling factor gamma:
```
X = proxnonconv_square(Z,r,p,gamma)
```
