# -*- coding: utf-8 -*-
MODE_NP = 'np'
MODE_TT = 'tt'
MODE_SP = 'sp'
DEF_TAU = 1.E-14
DEF_TENSOR_NAME = 'Unknown tensor'
DEF_VECTOR_NAME = 'Unknown vector'
DEF_MATRIX_NAME = 'Unknown matrix'

import numpy as np
 
from numpy.linalg import norm as np_norm
from scipy.sparse.linalg import norm as sp_norm

from numpy import kron as np_kron
from tt import kron as tt_kron
from scipy.sparse import kron as sp_kron

class TensorBase(object):
    
    def __init__(self, x=None, d=None, mode=MODE_NP, tau=DEF_TAU,
                 conv2mode=True, name=DEF_TENSOR_NAME):
        self.x = x
        self.d = d
        self.mode = mode
        if tau is not None:
            self.tau = tau
        else:
            self.tau = DEF_TAU
        if conv2mode:
            self.conv2mode()
        self.name = name
        self.type = 'tensor'

    @property
    def isnone(self):
        return (self.x is None)
        
    @property
    def isnotnone(self):
        return (not self.isnone)
        
    @property
    def n(self):
        if self.d is None:
            return 0
        return 2**self.d
        
    @property
    def r(self):
        if self.mode != MODE_TT or self.isnone:
            return None
        return self.x.r
        
    @property
    def erank(self):
        if self.mode != MODE_TT or self.isnone:
            return None
        return self.x.erank

    @property  
    def to_np(self):
        return self.full().x
        
    @property
    def to_tt(self):
        res = self.copy()
        res.conv2mode(MODE_TT)
        return res.x
        
    @property           
    def to_sp(self):
        res = self.copy()
        res.conv2mode(MODE_SP)
        return res.x

    def round(self, tau=None):
        res = self.copy(copy_x=False)
        if tau is not None:
            if isinstance(tau, (list, np.ndarray)):
                tau = _max_tau(tau)
            res.tau = tau
        if self.mode == MODE_TT and self.isnotnone:
            res.x = self.x.round(res.tau)
        else:
            res.x = self.x.copy()
        return res
      
    def norm(self):
        if self.isnone:
            return None
        if self.mode == MODE_NP or (self.mode == MODE_SP and self.type == 'vector'):
            return np_norm(self.x)
        if self.mode == MODE_TT:
            return self.x.norm()
        if self.mode == MODE_SP and self.type != 'vector':
            return sp_norm(self.x)
            
    def rel_err(self, other):
        '''
        Return ||x1-x2||/||x1|| if x1 and x2 != None or return None otherwise.
        '''
        if self.isnone or other.isnone:
            return None
        return (self - other).norm() / self.norm()
        
    def kron(self, other):
        '''
        Return self x other, where "x" is a Kronecker product operation.
        '''
        if self.isnone or other.isnone or self.mode != other.mode:
            raise ValueError('Incorrect input.')
        res = self.copy(copy_x=False)
        res.d = self.d + other.d
        if self.mode == MODE_NP or (self.mode == MODE_SP and self.type == 'vector'):
            res.x = np_kron(self.x, other.x)
        if self.mode == MODE_TT:
            res.x = tt_kron(other.x, self.x)
            res = res.round([self.tau, other.tau])
        if self.mode == MODE_SP and self.type != 'vector':
            res.x = sp_kron(self.x, other.x)
        return res
   
    def __add__(self, other):
        res = self.copy(copy_x=False)
        if isinstance(other, (int, float)) and self.isnotnone:
            res.x = self.x + other
            return res.round()
        if self.isnone or other.isnone or self.mode != other.mode or self.d != other.d:
            raise ValueError('Incorrect input.')
        res.x = self.x + other.x
        return res.round([self.tau, other.tau])
        
    def __radd__(self, other):
        return self.__add__(other)
        
    def __sub__(self, other):
        res = self.copy(copy_x=False)
        if isinstance(other, (int, float)) and self.isnotnone:
            res.x = self.x - other
            return res.round()
        if self.isnone or other.isnone or self.mode != other.mode or self.d != other.d:
            raise ValueError('Incorrect input.')
        res.x = self.x - other.x
        return res.round([self.tau, other.tau])
        
    def __mul__(self, other):
        res = self.copy(copy_x=False)
        if isinstance(other, (int, float)) and self.isnotnone:
            res.x = self.x * other
            return res
        if self.isnone or other.isnone or self.mode != other.mode or self.d != other.d:
            raise ValueError('Incorrect input.')
        raise NotImplementedError()
    
    def __rmul__(self, other):
        return self.__mul__(other)
        
    def __str__(self):
        s = ' %s-%s'%(self.mode, self.type)
        s+= ' %-20s'%('|%s|'%self.name)
        s+= ' [tau=%-8.2e]'%(self.tau)
        if self.d is not None:
            s+= ' {d=%2d}'%self.d
        if self.erank is not None:
            s+= ' (erank=%5.1f)'%self.erank
        if self.isnone:
            s+= ' x = None'
        return s
    
def _max_tau(tau_list):
    if not isinstance(tau_list, (list, np.ndarray)) or len(tau_list)==0:
        raise ValueError('Incorrect tau.')
    return np.max(tau_list)
    
def _n2d(n):
    d = int(np.log2(n))
    if 2**d != n:
        raise ValueError('Incorrect size.')
    return d
    
def _ind2mind(d, ind):
    if ind < 0:
        if ind == -1:
            mind = [1] * d
        else:
            raise ValueError('Only "-1" is supported for negaive indices.')
    else:
        mind = []
        for i in xrange(d):
            mind.append(ind % 2)
            ind /= 2
        if ind > 0:
            raise ValueError('Index is out of range.')
    return mind