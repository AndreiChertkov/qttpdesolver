# -*- coding: utf-8 -*- 
import numpy as np

import tt

from general import is_mode_tt, ttround

def half_array(x, ind, tau=None):
    ''' 
    Get a first (ind=0) or second (ind=1) half of array. In tt-case transform
    d-dim array A to (d-1)-dim array.
    ** For MODE_TT it is assumed that x is a QTT-tensor with standard
    mode size 2. If it's wrong, and mode size is n!=2, then for ind=0 function
    returns x[:len(x)/n] and so on.
    '''
    if not is_mode_tt(x):
        if ind==0:
            return x[:x.shape[0]/2]
        elif ind==1:
            return x[x.shape[0]/2:]
        else:
            raise ValueError('Only 0,1 values for index are valid!')
    GG = [G.copy() for G in tt.tensor.to_list(x)[:-1]]
    Gl = tt.tensor.to_list(x)[-1]
    GG[-1] = np.tensordot(GG[-1], Gl[:, ind, :], axes=(-1, 0))
    return ttround(tt.tensor.from_list(GG), tau)

def eo_sub_array(x, ind, tau=None):
    ''' 
    Get a subarray x[0::2] of even x elements (ind=0) or
                   x[1::2] of odd  x elements (ind=1).
    ** For MODE_TT it transforms d-dim array x to (d-1)-dim array.
    '''
    if not is_mode_tt(x):
        if ind==0:
            return x[0::2]
        elif ind==1:
            return x[1::2]
        else:
            raise ValueError('Only 0,1 values for index are valid!')
    GG = [G.copy() for G in tt.tensor.to_list(x)[1:]]
    Gf = tt.tensor.to_list(x)[0]
    
    GG[0] = np.tensordot(Gf[:, ind, :], GG[0], axes=(-1, 0))
    return ttround(tt.tensor.from_list(GG), tau)
    
def part_array(x, num, tau=None):
    ''' 
    For an array of length 2^(2d) get a part (num = 0, 1, ..., 2**d - 1) 
    of length 2^d.
    *** It works also for arrays of len 2^(2d+1) in that case
        part has len 2^(d+1) and num 0, 1, ..., 2**d - 1.
    '''
    if not is_mode_tt(x):
        n = int(np.sqrt(x.shape[0]))
        return x[num*n:(num+1)*n]
    n = x.d/2
    ind = np.zeros(n, dtype=np.int)
    ind[0] = num%2
    num = (num - ind[0])/2
    for i in xrange(1, n):
        ind[i] = num%2
        num = (num - ind[i])/2
    ind = ind[::-1]
    q = half_array(x, ind[0], tau)
    for i in ind[1:]:
        q = half_array(q, i, tau)
    return q