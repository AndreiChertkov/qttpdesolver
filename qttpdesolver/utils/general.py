# -*- coding: utf-8 -*- 
MODE_NP = 0
MODE_TT = 1
MODE_SP = 2

import numpy as np
from numpy.linalg import norm as np_norm
from scipy.sparse.linalg import norm as sp_norm
from scipy.sparse import diags as sp_diag
from scipy.sparse.csr import csr_matrix
from scipy.sparse.coo import coo_matrix

import tt
import tt.multifuncrs2 as multifuncrs2
     
def is_mode_np(x):
    '''
    Return True if x is in MODE_NP and False otherwise.
    '''
    if isinstance(x, (int, float, bool, np.ndarray)):
        return True
    return False
 
def is_mode_tt(x):
    '''
    Return True if x is in MODE_TT and False otherwise.
    '''
    if isinstance(x, (tt.vector, tt.matrix)):
        return True
    return False
        
def is_mode_sp(x):
    '''
    Return True if x is in MODE_SP and False otherwise.
    '''
    if isinstance(x, (csr_matrix, coo_matrix)):
        return True
    return False
        
def define_mode(x):
    '''
    Return MODE_NP / MODE_TT / MODE_SP or None according to the type of x.
    '''
    if is_mode_np(x):
        return MODE_NP
    if is_mode_tt(x):
        return MODE_TT
    if is_mode_sp(x):
        return MODE_SP
    return None
        
def define_mode_list(lst, num_as_none=False):
    '''
    Return MODE_NP / MODE_TT / MODE_SP or None according to the type of
    the first non-None list element.
    If num_as_none==True, then elements of int/float will be passed (as None).
    '''
    for x in lst:
        if isinstance(x, list):
            mode = define_mode_list(x, num_as_none)
            if mode is not None:
                return mode
        if num_as_none and isinstance(x, (int, float)):
            continue
        mode = define_mode(x)
        if mode is not None:
            return mode
    return None

def norm(x):
    '''
    Return ||x|| if x != None or return None otherwise.
    '''
    if x is None:
        return None
    if is_mode_np(x):
        return np_norm(x)
    if is_mode_sp(x):
        return sp_norm(x)
    if is_mode_tt(x):
        return x.norm()
    return None
        
def rel_err(x1, x2):
    '''
    Return ||x1-x2||/||x1|| if x1 and x2 != None or return None otherwise.
    '''
    if x1 is None or x2 is None:
        return None
    if define_mode(x1) != define_mode(x2):
        raise ValueError('Both operands should be in the same format.')
    return norm(x1-x2)/norm(x1)

def ttround(x=None, tau=None):
    ''' 
    Round tt-vector or matrix to accuracy tau,
    or return x if it is None or mode is MODE_NP or MODE_SP or if tau is None.
    '''
    if x is None or not is_mode_tt(x) or tau is None:
        return x
    return x.round(tau)
        
def vdiag(x, tau=None, to_sp=False):
    ''' 
    Construct a diagonal matrix from a vector (if x is a vector) or
    construct matrix diagonal (if x is a matrix).
    If x is a vector in MODE_NP and result is needed in MODE_SP, then
    to_sp should be set to True.
    '''
    mode = define_mode(x)
    if mode == MODE_NP:
        if to_sp:
            return sp_diag([x], [0], format='csr')
        return np.diag(x)
    if mode == MODE_TT:
        return ttround(tt.diag(x), tau)
    if mode == MODE_SP:
        return x.diagonal()
    return None
 
def msum(lst, tau=None):
    ''' 
    Calculate a sum of a list of vectors or matrices.
    '''
    res = lst[0].copy()
    for x in lst[1:]:
        res = ttround(res + x, tau)
    return res
    
def vsum(x, tau=None):
    ''' 
    Calculate a sum of vector's elements.
    '''
    if not is_mode_tt(x):
        return np.sum(x)
    return tt.sum(x)
    
def mprod(lst, tau=None):
    ''' 
    Calculate matrix-dot product for a list of matrices m.
    '''
    res = lst[0].copy()
    for x in lst[1:]:
        if not is_mode_tt(x):
            res = res.dot(x)
        else:
            res = ttround(res * x, tau)
    return res
    
def vprod(lst, tau=None):
    ''' 
    Calculate elemenwise product for a list of vectors v.
    '''
    res = lst[0].copy()
    for x in lst[1:]:
        res = ttround(res * x, tau)
    return res
    
def mvec(A, x, tau=None):
    ''' 
    Calculate a matrix-vector product for matrix A and vector x.
    '''
    if not is_mode_tt(A):
        return A.dot(x)
    return ttround(tt.matvec(A, x), tau)
    
def vinv(x, eps=1.E-8, tau=None, x0=None, abs_val=False,
         name='Unknown vector', verb=False):
    ''' 
    For a given vector x construct vector 1/x or 1|x| (if abs_val==True).
            Input:
    x       - a given vector x
              type: np.ndarray or tt.vector
    eps     - is a cross approximation accuracy
              type: float (default: 1.E-8)
    tau     - is an accuracy for rounding operation of result
              (if it's None, then rounding will not be performed)
              type: float or None (default: None)
    x0      - is an initial guess for cross approximation
              (if it's None, then it will be set equal to x value)
              type: tt.vector or None (default: None)
    abs_val - if is True, then 1/|x| (instead of 1/x) will be calculated
              type: bool (default: False)
    name    - a string to present process
              type: str (defalt:'Unknown vector')
    verb    - if is True, then cross approximation ouput will be printed
              type: bool (default: False)
            Output:
    ix      - vector 1/x or 1/|x|
              type: the same as of input x
    '''
    if not is_mode_tt(x):
        if not abs_val:
            return 1./x
        else:
            return 1./np.abs(x)    
    if verb:
        print '  Construction of inverse for %s'%name
    if x0 is None:
        x0 = x
    if not abs_val:
        ix = multifuncrs2([x], lambda x: 1./x, eps, verb=verb, y0=x0)
    else:
        ix = multifuncrs2([x], lambda x: 1./np.abs(x), eps, verb=verb, y0=x0)
    if tau is not None:
        ix = ix.round(tau)
    return ix