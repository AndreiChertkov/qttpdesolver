# -*- coding: utf-8 -*- 
import numpy as np
from numpy import kron as np_kron
from scipy.sparse import kron as sp_kron
from scipy.sparse import diags as sp_diag
from scipy.linalg import block_diag as sp_block_diag

import tt

from general import is_mode_np, is_mode_tt, is_mode_sp
from general import ttround, msum, vdiag

def mblock(lst, tau=None):
    '''
    Construct rectangular block matrix from list (or list of lists) lst,
    that contains matrices of equal shape and maybe some (not all!)
    None/int/float entries. For None/int/float entries elements in the 
    corresponding block will be filled by zeros.
            Input:
    lst     - a given list of matrices
              type: list or list of lists of np.ndarray/tt.matrix/scipy csr
    tau     - is an accuracy for TT-rounding operation
              (if it's None, then rounding will not be performed)
              type: float or None (default: None)
            Output:
    res     - rectangular matrix
              type: is the same as type of matrices in lst
    '''
    if not isinstance(lst, list):
        raise ValueError('This should be a list.')
    n = None # number of rows in lst
    m = None # number of columns in lst
    for i in range(len(lst)): 
        if isinstance(lst[i], list):
            m = len(lst[i])
            break
    if m is None:
        m = len(lst)
        lst = [lst]
    n = len(lst)
    res = None
    for i in range(n):
        if not isinstance(lst[i], list):
            raise ValueError('List of lists should contain only lists.')
        if len(lst[i]) != m:
            raise ValueError('This should be a rectangular block matrix.')
        for j in range(m):
            if lst[i][j] is None or isinstance(lst[i][j], (int, float)):
                continue
            ek = np.zeros((n, m))
            ek[i, j] = 1
            if is_mode_tt(lst[i][j]):
                ek = tt.matrix(ek)
            res_curr = kronm([ek, lst[i][j]])
            if res is None:
                res = res_curr
            else:
                res = msum([res, res_curr], tau)
    return res
    
def vblock(lst, tau=None):
    '''
    Concatenate vectors from a list lst. All vectors must have the same length.
    List may contain some (not all!) int/float or None entries, that will be
    filled by zero-vectors.
            Input:
    lst     - a given list of vectors
              type: list of np.ndarray(1D) or tt.vector
    tau     - is an accuracy for rounding operation
              (if it's None, then rounding will not be performed)
              type: float or None (default: None)
            Output:
    res     - concatenation
              type: is the same as type of vectors in lst
    '''
    if not isinstance(lst, list):
        raise ValueError('This should be a list.')
    n = len(lst)
    res = None
    for i in range(n):
        if lst[i] is None or isinstance(lst[i], (int, float)):
            continue
        ek = np.zeros(n)
        ek[i] = 1
        if is_mode_tt(lst[i]):
            ek = tt.tensor(ek)
        res_curr = kronm([ek, lst[i]], tau)
        if res is None:
            res = res_curr
        else:
            res = msum([res, res_curr], tau)
    return res
   
def kronm(lst, tau=None):
    '''
    Compute Kronecker products for given list of matrices or vectors.
    (If tt-format is used then list is reversed.)
    '''
    if is_mode_tt(lst[0]):
        lst.reverse()
    res = lst[0].copy()
    for x in lst[1:]:
        if is_mode_np(x):
            res = np_kron(res, x)
        elif is_mode_tt(x):
            res = ttround(tt.kron(res, x), tau)
        elif is_mode_sp(x):
            res = sp_kron(res, x)
        else:
            raise ValueError('Incorrect element type in list.')  
    return res
    
def space_kron(x, axis, d, n, dim, tau=None):
    '''
    Prepare multidimensional (dim) version of the 1d operator x (in a matrix
    form) or 1d vector by corresponding Kronecker products with edentity matrix
    or edentity vector respectively for given axis.
    '''
    if dim==1:
        if axis == 0:
            return x
        raise ValueError('Incorrect axis number')
    if isinstance(x, tt.vector):
        I = tt.ones(2, d)
    elif isinstance(x, tt.matrix):
        I = tt.eye(2, d)
    else:
        if len(x.shape)==1:
            I = np.ones(n)  
        elif isinstance(x, np.ndarray):
            I = np.eye(n)
        else: #isinstance(A, csr_matrix):
            I = sp_diag([np.ones(n)], [0], format='csr')
    if dim==2:
        if axis==0:
            return kronm([I, x], tau)
        if axis==1:
            return kronm([x, I], tau)
        raise ValueError('Incorrect axis number')
    if dim==3:
        if axis==0:
            return kronm([I, I, x], tau)
        if axis==1:
            return kronm([I, x, I], tau)  
        if axis==2:
            return kronm([x, I, I], tau)
    raise ValueError('Incorrect dimension')
    
def sum_out(x, axis, d, n, dim, tau=None):
    ''' 
    Fast calculation of the special product for dim=2,3 (for =1 x is returned) 
    S[axis] diag(x) S[axis].T, where x is a vector of size (n**dim) and
    for dim==2: S[0] = I \kron e.T
                S[1] = e.T \kron I
    for dim==3: S[0] = I \kron I \kron e.T
                S[1] = I \kron e.T \kron I
                S[2] = e.T \kron I \kron I
    where I is an (n, n) eye matrix and e is an (n, 1) vector of ones.
    The result is a diagonal matrix, and the function returns it's diagonal,
    which is a vector of size (n**(dim-1)).
    '''
    if dim==1:
        if axis==0:
            return x
        raise ValueError('Incorrect axis number')
    if dim==2:
        if not axis in [0, 1]:
            raise ValueError('Incorrect axis number')
    elif dim==3:
        if not axis in [0, 1, 2]:
            raise ValueError('Incorrect axis number')
    else:
        raise ValueError('Incorrect dimension')
    if not is_mode_tt(x):
        res = np.sum(x.reshape(tuple([n]*dim), order='F'), axis=axis)
        res = res.flatten('F')
        return res
    axs = [np.arange(d), np.arange(d, 2*d), np.arange(2*d, 3*d)][axis]
    res = _sum_out_axis(x, axs, tau)
    return res 
    
def _sum_out_axis(A, axis, tau=None):
    '''
    Sums out the indices in A given by the array from axis.
    Works only for TT !!!
    '''
    cores = tt.tensor.to_list(A)
    for i in axis:
        cores[i] = np.sum(cores[i], axis=1)
    cr0 = None
    new_cores = []
    for i, cr in enumerate(cores):
        if i in axis:
            if cr0 is None:
                cr0 = cr.copy()
            else:
                cr0 = np.tensordot(cr0, cr, 1)
        else:
            if cr0 is not None:
                new_cores.append(np.tensordot(cr0, cr, 1))
                cr0 = None
            else:
                new_cores.append(cr)
    if cr0 is not None:
        new_cores[-1] = np.tensordot(new_cores[-1], cr0, 1)
    B = tt.tensor.from_list(new_cores)
    return ttround(B, tau)
   
def kron_out(x, axis, d, n, dim, tau=None):
    '''
    Fast calculation of the special product for dim=2,3 (for =1 A is returned) 
    S[axis].T diag(x) S[axis], where x is a vector of size (n**(dim-1)) and
    for dim==2: S[0] = I \kron e.T
                S[1] = e.T \kron I
    for dim==3: S[0] = I \kron I \kron e.T
                S[1] = I \kron e.T \kron I
                S[2] = e.T \kron I \kron I
    where I is an (n, n) eye matrix and e is an (n) vector of ones.
    The function return resulting matrix of shape (n**dim, n**dim).
    '''
    if dim==1:
        if axis==0:
            return x
        raise ValueError('Incorrect axis number')
    if not is_mode_tt(x):
        ee = np.ones((n, n))
    else:
        ee = tt.eye(2, d); ee.tt = tt.ones(4, d)
    if dim==2:
        if axis==0:
            return kronm([vdiag(x), ee], tau)
        if axis==1:
            return kronm([ee, vdiag(x)], tau)
        raise ValueError('Incorrect axis number')
    if dim==3:
        if axis==0:
            return kronm([vdiag(x), ee], tau)
        if axis==1:
            return _kron_block(ee, vdiag(x), d, n, tau)  
        if axis==2:
            return kronm([ee, vdiag(x)], tau)
        raise ValueError('Incorrect axis number')
    raise ValueError('Incorrect dimension')
    
def _kron_block(A, B, d, n, tau=None):
    '''
    Construct kron product of the special form.
    '''
    if not is_mode_tt(A):
        N = B.shape[0]
        n = int(N**0.5)
        blocks = []
        for i in xrange(n):
            b = B[n*i: n*(i+1), n*i: n*(i+1)]
            blocks.append(np.kron(A, b))
        return sp_block_diag(*blocks) 
        
    G = tt.matrix.to_list(B)
    for j in range(d):
        i = d
        r = G[i-1].shape[-1]
        Gnew = np.zeros((r, 2, 2, r))
        Gnew[:, 0, 0, :] = np.eye(r)
        Gnew[:, 1, 0, :] = np.eye(r)
        Gnew[:, 0, 1, :] = np.eye(r)
        Gnew[:, 1, 1, :] = np.eye(r)
        G.insert(i, Gnew)
    return tt.matrix.from_list(G)
    
#
#                        Is not ready (and don't used in main packages):
#
#def vfelem(x, mode=MODE_NP):
#    ''' 
#    Return a first element of the vector.
#    '''
#    if mode == MODE_NP:
#        return x[0]
#    else:
#        return vfelem_tt(x)
#
#def vlelem(x, mode=MODE_NP):
#    ''' 
#    Return a last element of the vector.
#    '''
#    if mode == MODE_NP:
#        return x[-1]
#    else:
#        return vlelem_tt(x)
    
#def _kron_block_old(A, B, d, n, tau=None, mode=MODE_NP):
#    '''
#    Construct kron product of the special form.
#    '''
#    if mode == MODE_NP:
#        N = B.shape[0]
#        n = int(N**0.5)
#        blocks = []
#        for i in xrange(n):
#            b = B[n*i: n*(i+1), n*i: n*(i+1)]
#            blocks.append(np.kron(A, b))
#        return sp_block_diag(*blocks) 
#    from splitting import part_array
#    res = None
#    C = tt.diag(B)
#    for i in xrange(n):
#        q = part_array(C, i, tau, mode)
#        q = kronm([A, tt.diag(q)], tau, mode)
#        g = np.zeros(n); g[i] = 1
#        if tau is not None:
#            g = tt.tensor(g.reshape(tuple([2]*(d)), order='F'), tau)
#        else:
#            g = tt.tensor(g.reshape(tuple([2]*(d)), order='F'))
#        g = vdiag(g, tau, mode)
#        res_curr = kronm([g, q], tau, mode)
#        if res is None:
#            res = res_curr
#        else:
#            res = msum([res, res_curr], tau, mode)
#    return res
#  
#def ind2sub(siz, num):
#    '''
#    Convert index (num) of the vector to a QTT form with shape siz.
#    Have sense only for TT !!!
#    '''
#    n = len(siz)
#    ind = np.zeros(n, dtype=np.int)
#    ind[0] = num%2
#    num = (num - ind[0])/2
#    for i in xrange(1, n):
#        ind[i] = num%2
#        num = (num - ind[i])/2
#    return ind
#    
#New matvec
#from tt.amen.amen_mv import amen_mv
#res, z = amen_mv(A, x, tau) 