# -*- coding: utf-8 -*- 
import numpy as np
from scipy.sparse import diags as sp_diag
from numba import jit

import tt

from general import MODE_NP, MODE_TT, MODE_SP, msum, ttround
from block_and_space import kronm

@jit(nopython=True)
def toeplitz(c, r):
    '''
    Toeplitz matrix with c is the first column and
    r[1:] is the first row (except the first element).
    '''
    n = c.shape[0]
    a = np.zeros((n, n))
    for i in xrange(n):
        for j in xrange(i+1):
            a[i, j] = c[i - j]
        for j in xrange(i+1, n):
            a[i, j] = r[j - i]
    return a  
    
def eye(d, n, mode=MODE_NP):
    ''' 
    Eye matrix: diag(1, 1,..., 1).
    '''
    if mode == MODE_NP:
        return np.eye(n)
    if mode == MODE_TT:
        return tt.eye(2, d)
    if mode == MODE_SP:
        return sp_diag([np.ones(n)], [0], format='csr')
    raise ValueError('Incorrect mode.')
        
def volterra(d, n, h, tau=None, mode=MODE_NP):
    '''
    Volterra integral 1D matrix
    [i, j] = h if i>=j and = 0 otherwise.
    '''
    if mode == MODE_NP or mode == MODE_SP:
        B = h * toeplitz(c=np.ones(n), r=np.zeros(n))
        return B
    if mode == MODE_TT:
        B = h * (tt.Toeplitz(tt.ones(2, d), kind='U') + tt.eye(2, d))
        return ttround(B, tau)
    if mode == MODE_SP:
        raise ValueError('Is not implemented for MODE_SP.')
    raise ValueError('Incorrect mode.')

def findif(d, n, h, tau=None, mode=MODE_NP):
    '''
    Finite difference 1D matrix
    [i, j] = 1/h if i=j, [i, j] =-1/h if i=j+1 and = 0 otherwise.
    '''
    if mode == MODE_NP:
        B = 1./h*sp_diag([np.ones(n), -np.ones(n-1)], [0, -1], format='csr')
        return B.toarray()
    if mode == MODE_TT:
        e1 = tt.tensor(np.array([0.0, 1.0]))
        e2 = tt.mkron([e1] * d)
        B = (1./h)*(tt.Toeplitz(-e2, kind='U') + tt.eye(2, d))
        return ttround(B, tau)
    if mode == MODE_SP:
        B = 1./h*sp_diag([np.ones(n), -np.ones(n-1)], [0, -1], format='csr')
        return B
    raise ValueError('Incorrect mode.')
    
def shift(d, n, tau=None, mode=MODE_NP):
    '''
    Shift 1D matrix
    [i, j] = 1 if i=j+1, [i, j] = 0 otherwise.
    '''
    return msum([eye(d, n, mode), (-1.)*findif(d, n, 1., tau, mode)], tau)
    
def interpol_1d(d, n, tau=None, mode=MODE_NP):
    '''
    1D interpolation matrix.
    Is valid only for mesh x_j = h+hj, j=0,1,...,n-1.
    '''
    if mode == MODE_NP:
        M = np.zeros((2*n, 2*n))
        for j in range(n):
            M[2*j+0, j*2] = 1
            M[2*j+1, j*2] = 2
            if j<n-1:
                M[2*j+2, j*2] = 1
        return 0.5*M
    if mode == MODE_SP:
        raise ValueError('Is not implemented for MODE_SP.')
    if mode == MODE_TT:
        e1 = tt.matrix(np.array([[1., 0.], [2., 0.]]))
        e2 = tt.tensor(np.array([0.0, 1.0]))
        e3 = tt.matrix(np.array([[1., 0.], [0., 0.]]))
        M1 = kronm([eye(d, n, mode), e1], tau)
        M2 = tt.Toeplitz(tt.mkron([e2] * d), kind='U')
        M2 = kronm([M2, e3], tau)
        return 0.5*msum([M1, M2], tau)
    raise ValueError('Incorrect mode.')

def vzeros_except_one(d, ind, mode=MODE_NP, value=1.):
    '''
    Construct a vector in the given mode of length 2**d with only one
    nonzero element in position ind, that is equal to a given value.
    '''
    if mode!=MODE_TT:
        res = np.zeros(2**d)
        res[ind] = value
        return res
    n = np.array([2] * d, dtype=np.int32)
    if ind < 0:
        cind = [0] * d
    else:
        cind = []
        for i in xrange(d):
            cind.append(ind % n[i])
            ind /= n[i]
        if ind > 0:
            cind = [0] * d
    cr = []
    for i in xrange(d):
        cur_core = np.zeros((1, n[i], 1))
        cur_core[0, cind[i], 0] = 1
        cr.append(cur_core)
    return value*tt.vector.from_list(cr)
    
# Are not used!        
#
#from scipy.sparse import coo_matrix
#       
#def eye_first(d, n, mode=MODE_NP, use_csr=False):
#    ''' 
#    Diagonal matrix: diag(1, 0,..., 0, 0).
#    '''
#    if mode == MODE_NP:
#        M = coo_matrix(([1], ([0], [0])), shape=(n, n)).tocsr()
#        if not use_csr:
#            return M.toarray()
#        else:
#            return M
#    else:
#        G = tt.matrix.to_list(tt.eye(2, d))
#        for i in range(len(G)):
#            G[i][:, 1, 1, :] = 0. 
#        return tt.matrix.from_list(G)
#        
#def eye_last(d, n, mode=MODE_NP, use_csr=False):
#    ''' 
#    Diagonal matrix: diag(0, 0,..., 0, 1).
#    '''
#    if mode == MODE_NP:
#        M = coo_matrix(([1], ([n-1], [n-1])), shape=(n, n)).tocsr()
#        if not use_csr:
#            return M.toarray()
#        else:
#            return M
#    else:
#        G = tt.matrix.to_list(tt.eye(2, d))
#        for i in range(len(G)):
#            G[i][:, 0, 0, :] = 0. 
#        return tt.matrix.from_list(G)
#        
#def eye_except_first(d, n, mode=MODE_NP, use_csr=False):
#    ''' 
#    Diagonal matrix: diag(0, 1,..., 1, 1).
#    '''
#    return eye(d, n, mode, use_csr)-eye_first(d, n, mode, use_csr)
#        
#def eye_except_last(d, n, mode=MODE_NP, use_csr=False):
#    ''' 
#    Diagonal matrix: diag(0, 1,..., 1, 1).
#    '''
#    return eye(d, n, mode, use_csr)-eye_last(d, n, mode, use_csr)
#    
#def eye_p1_first(d, n, mode=MODE_NP, use_csr=False):
#    ''' 
#    Matrix: A[i, j] = 0, A[0, 1] = 1.
#    '''
#    if mode == MODE_NP:
#        M = coo_matrix(([1], ([0], [1])), shape=(n, n)).tocsr()
#        if not use_csr:
#            return M.toarray()
#        else:
#            return M
#    else:
#        G = tt.matrix.to_list(tt.eye(2, d))
#        for i in range(len(G)):
#            G[i][:, 1, 1, :] = 0. 
#        G[0][:, 0, 1, :] = 1.
#        return tt.matrix.from_list(G)-eye_first(d, 2**d, MODE_TT)
#    
#def eye_p1_last(d, n, mode=MODE_NP, use_csr=False):
#    ''' 
#    Matrix: A[i, j] = 0, A[-1, -2] = 1.
#    '''
#    if mode == MODE_NP:
#        M = coo_matrix(([1], ([n-2], [n-1])), shape=(n, n)).tocsr()
#        if not use_csr:
#            return M.toarray()
#        else:
#            return M
#    else:
#        G = tt.matrix.to_list(tt.eye(2, d))
#        for i in range(len(G)):
#            G[i][:, 0, 0, :] = 0. 
#        G[0][:, 0, 1, :] = 1.
#        return tt.matrix.from_list(G)-eye_last(d, 2**d, MODE_TT)
#        
#def ones(d, n, mode=MODE_NP):
#    ''' 
#    Ones vector: [1, 1,..., 1, 1] (shape=(n, ) or qtt-tensor)
#    '''
#    if mode == MODE_NP:
#        return np.ones(n)
#    else:
#        return tt.ones(2, d)
#
#def ones_first(d, n, mode=MODE_NP, use_csr=False):
#    ''' 
#    Vector: [1, 0,..., 0, 0] (shape=(n, ) or qtt-tensor)
#    '''
#    if mode == MODE_NP:
#        x = np.zeros(n); x[0] = 1.
#    else:
#        e = tt.tensor(np.array([1., 0.]))
#        x = tt.mkron([e]*d)
#    return x
#
#def ones_last(d, n, mode=MODE_NP):
#    ''' 
#    Vector: [0, 0,..., 0, 1] (shape=(n, ) or qtt-tensor)
#    '''
#    if mode == MODE_NP:
#        x = np.zeros(n); x[-1] = 1.
#    else:
#        e = tt.tensor(np.array([0., 1.]))
#        x = tt.mkron([e]*d)
#    return x
#
#def ones_except_first(d, n, mode=MODE_NP):
#    ''' 
#    Vector: [0, 1,..., 1, 1] (shape=(n, ) or qtt-tensor)
#    '''
#    return ones(d, n, mode) - ones_first(d, n, mode)
#
#def ones_except_last(d, n, mode=MODE_NP):
#    ''' 
#    Vector: [1, 1,..., 1, 0] (shape=(n, ) or qtt-tensor)
#    '''
#    return ones(d, n, mode) - ones_last(d, n, mode)
#    
#def zeros_except_one(d, n, i, j=None, mode=MODE_NP, value=1.):
#    '''
#    Construct a vector (if j is None) or matrix (if is not None)
#    with only one nonzero element in position [i, j] ([i]) with given value.
#    '''
#    if mode == MODE_NP:
#        if j is not None:
#            X = np.zeros((n, n))
#            X[i, j] = 1.
#        else:
#            X = np.zeros(n)
#            X[i] = 1.
#        return X
#    ii = ind2sub([2]*d, i)
#    if j is not None:
#        jj = ind2sub([2]*d, j)
#    GG = []
#    for k in range(d):
#        if j is not None:
#            GG.append(np.zeros((1, 2, 2, 1)))
#            GG[k][0, ii[k], jj[k], 0] = value
#        else:
#            GG.append(np.zeros((1, 2, 1)))
#            GG[k][0, ii[k], 0] = value
#    if j is not None:
#        return tt.matrix.from_list(GG)
#    else:
#        return tt.vector.from_list(GG)