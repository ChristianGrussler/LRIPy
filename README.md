# LRIPy
Python code for Low-rank optimization by Low-Rank Inducing Norms as well as non-convex Douglas-Rachford. 

## Purpose:
Low-rank rank inducing norms and non-convex Proximal Splitting Algoriths attempt to find exact rank/cardinality-r solutions to minimization problems with convex loss functions, i.e., avoiding of regularzation heuristics. LRIPy provides Python implementations for the proximal mappings of the low-rank inducing Frobenius and Spectral norms, as well as, their epi-graph projections and non-convex counter parts.

## Literature:

### Low-rank inducing norms: 
* [Rank Reduction with Convex Constraints](https://lup.lub.lu.se/search/publication/54cb814f-59fe-4bc9-a7ef-773cbcf06889)
* [Low-rank Inducing Norms with Optimality Interpretations](https://arxiv.org/abs/1612.03186)
* [Low-rank Optimization with Convex Constraints](https://arxiv.org/abs/1606.01793)
* [The Use of the r* Heuristic in Covariance Completion Problems](http://www.control.lth.se/index.php?mact=ReglerPublicationsB,cntnt01,showpublication,0&cntnt01LUPid=a61669c7-29b9-41ee-82da-9c825b08f8d8&cntnt01returnid=60)
* [On optimal low-rank approximation of non-negative matirces](http://lup.lub.lu.se/search/ws/files/21812505/2015cdcGrusslerRantzer.pdf)

### Non-convex counter parts:
* [Local Convergence of Proximal Splittinge Methods for Rank Constrained Problems](https://arxiv.org/abs/1710.04248)

## Installation

The easiest way to install the package is to run ``pip install lripy``. To install the package from source, run ``python setup.py install`` in the main folder.  

## Documentation
In the following it holds that
* for the low-rank inducing Frobenius norm: p = 2
* for the low-rank inducing Spectral norm:  p = 'inf'

### Examples
There are two examples in the "example" folder:

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
LRIPy provides Python implemenations for the proximal mappings to the low-rank inducing Frobenius and Spectral norm as well as their epi-graph projections and non-convex counter parts.

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
#### Projection on the epi-graph of the low-rank inducing norms: 
Projection of (Z,zv) on the epi-graph of the low-rank inducing norms with parameter r and scaling factor gamma:
```
[X,xv] = projrast(Z,zv,r,p,gamma)[0]
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