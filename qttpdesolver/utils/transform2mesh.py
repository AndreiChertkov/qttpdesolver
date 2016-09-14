# -*- coding: utf-8 -*- 
import numpy as np

import tt

from general import is_mode_tt, define_mode, mvec, ttround
from block_and_space import kronm
from splitting import eo_sub_array
from spec_matr import interpol_1d
        
def transform2finer(x, dim, tau=None, reps=1):
    ''' 
    Transform a given vector x of size N=2^{d*dim}, that is assumed to be
    defined on appropriate spatial grid, to a finer grid with
    N1=2^{(d+1)*dim} nodes. The transformation is repeated reps-times.
    '''
    if is_mode_tt(x):
        d = x.d / dim
    else:
        d = int(np.log2(x.size)) / dim
        if 2**(d*dim) != x.size:
            raise ValueError('Vector should have a size 2^{d*dim}.')
    res = x.copy()
    for i in range(reps):
        d_new = d + i
        n_new = 2**d_new
        if dim == 1:
            res = transform2finer_1d(res, d_new, n_new, tau)
        if dim == 2:
            res = transform2finer_2d(res, d_new, n_new, tau)
        if dim == 3:
            res = transform2finer_3d(res, d_new, n_new, tau)
    return res
    
def transform2coarser(x, dim, tau=None, reps=1):
    ''' 
    Transform a given vector x of size N=2^{d*dim}, that is assumed to be
    defined on appropriate spatial grid, to a coarser grid with
    N1=2^{(d-1)*dim} nodes. The transformation is repeated reps-times.
    '''
    if is_mode_tt(x):
        d = x.d / dim
    else:
        d = int(np.log2(x.size)) / dim
        if 2**(d*dim) != x.size:
            raise ValueError('Vector should have a size 2^{d*dim}.')
    res = x.copy()
    for i in range(reps):
        d_new= d - i
        n_new = 2**d_new
        if dim == 1:
            res = transform2coarser_1d(res, d_new, n_new, tau)
        if dim == 2:
            res = transform2coarser_2d(res, d_new, n_new, tau)
        if dim == 3:
            res = transform2coarser_3d(res, d_new, n_new, tau)
    return res
    
def transform2finer_1d(x, d, n, tau=None):
    ''' 
    For a given vector x of size N=n=2^d, that is defined on a 1D grid
    with constant step, constructs a new array y of size 2^{d+1} and form:
    x[0]/2, x[0], (x[0]+x[1])/2, x[1],..., (x[n-2]+x[n-1])/2, x[n-1].
    '''
    M = interpol_1d(d, n, tau, define_mode(x))
    e1 = np.array([1., 0.])
    if is_mode_tt(x):
        e1 = tt.tensor(e1)
    x2 = kronm([x, e1], tau)
    return mvec(M, x2, tau)
    
def transform2coarser_1d(x, d, n, tau=None):
    ''' 
    For a given vector x of size N=n=2^d, that is defined on a 1D grid
    with constant step, constructs a new array y of size 2^{d-1} and form:
    x[1], x[3], ..., x[n-1]
    '''
    return eo_sub_array(x, 1, tau)
    
def transform2finer_2d(x, d, n, tau=None):
    ''' 
    For a given vector x of size N=n^2=2^(2*d), that is defined on a 2D grid
    with constant step, constructs a new array y of size 2^{2*d+2}
    that is interpolation to the finer 2d grid.
    '''
    M = interpol_1d(d, n, tau, define_mode(x))
    M = kronm([M, M], tau)
    e1 = np.array([1., 0.])
        
    if not is_mode_tt(x):
        x1 = x.reshape((n, n))
        x1 = np.hstack((x1, np.zeros((n, n))))
        x1 = x1.flatten()
    else:
        e1 = tt.tensor(e1)
        G = tt.tensor.to_list(x)
        r = G[d-1].shape[2]
        Gnew = np.zeros((r, 2, r))
        Gnew[:, 0, :] = np.eye(r)
        G.insert(d, Gnew)
        x1 = ttround(tt.tensor.from_list(G), tau)
    x2 = kronm([x1, e1], tau)
    return mvec(M, x2, tau)
 
def transform2coarser_2d(x, d, n, tau=None):
    ''' 
    For a given vector x of size N=n^2=2^(2*d), that is defined on a 2D grid
    with constant step, constructs a new array y of size 2^{2*d+2}
    that is interpolation to the finer 2d grid.
    '''
    if not is_mode_tt(x):
        return x.reshape((n, n), order='F')[1:, 1:][::2, ::2].flatten('F')
    GG_in  = [G.copy() for G in tt.tensor.to_list(x)]
    GG_out = []
    for i in range(2):
        GG = GG_in[d*i+1:d*(i+1)]
        Gf = GG_in[d*i]
        n = GG[0].shape[1]
        rf = Gf.shape[0]
        GG[0] = GG[0].reshape((GG[0].shape[0], -1))
        GG[0] = Gf[:, 1, :].dot(GG[0])
        GG[0] = GG[0].reshape((rf, n, -1))
        GG_out.extend([G.copy() for G in GG])
    return ttround(tt.tensor.from_list(GG_out), tau)
   
def transform2finer_3d(x, d, n, tau=None):
    raise NotImplementedError('Transformation for 3d case is not implemented.')
    
def transform2coarser_3d(x, d, n, tau=None):
    raise NotImplementedError('Transformation for 3d case is not implemented.')
    
### OLD 
    
#def transform2finer_1d_slow(x, d, n, tau=None, mode=MODE_NP):
#    ''' 
#    For given array x of size n=2^d with constant mesh step constructs a new
#    array y of size 2^{d+1} and form:
#    x[0]/2, x[0], (x[0]+x[1])/2, x[1],..., (x[n-2]+x[n-1])/2, x[n-1].
#    '''
#    e1 = np.array([1., 0.])
#    e2 = np.array([0., 1.])
#    if mode == MODE_NP:
#        Md = sp_diag([np.ones(n-1)], [-1], format='csr')
#    else:
#        e1 = tt.tensor(e1)
#        e2 = tt.tensor(e2)
#        Md = tt.Toeplitz(tt.mkron([e2] * d), kind='U')
#    x1 = mvec(Md, x, tau)
#    x1 = msum([x1, x], tau) * 0.5
#    x1 = kronm([x1, e1], tau)
#    x2 = kronm([x , e2], tau)
#    return msum([x1, x2], tau)
#    
#def transform2finer_2d_slow(x, d, n, tau=None, mode=MODE_NP):
#    rows_new = []
#    for j in range(n):
#        row_old = part_array(x, j, tau)
#        row = transform2finer_1d(row_old, d, n, tau, mode)
#        if j>0:
#            rows_new[-1] = msum([rows_new[-1], 0.5*row], tau)
#        else:
#            rows_new.append(0.5*row)
#        rows_new.append(row)
#        if j<n-1:
#            rows_new.append(0.5*row)
#    return vblock(rows_new, tau)