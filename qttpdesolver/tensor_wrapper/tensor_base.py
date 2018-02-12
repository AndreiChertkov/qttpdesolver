# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import norm as np_norm
from scipy.sparse.linalg import norm as sp_norm
from numpy import kron as np_kron
from tt import kron as tt_kron
from scipy.sparse import kron as sp_kron

MODE_NP, MODE_TT, MODE_SP = 'np', 'tt', 'sp'
TENSOR_BASE = 'tensor_base'
TENSOR = 'tensor'
VECTOR = 'vector'
MATRIX = 'matrix'
DEF_TAU = 1.E-14

class TensorBase(object):
    kind = TENSOR_BASE

    def __init__(self, x=None, d=None, mode=MODE_NP, tau=DEF_TAU, conv2mode=True, name=TENSOR_BASE):
        self.x = x
        self.d = d
        self.mode = mode
        self.tau = tau if tau is not None else DEF_TAU
        self.name = name
        if conv2mode:
            self.conv2mode()
        if name in [TENSOR_BASE, TENSOR, MATRIX]:
            self.name = self.kind

    @property
    def isnone(self):
        return self.x is None

    @property
    def isnotnone(self):
        return not self.isnone

    @property
    def n(self):
        if self.d is None:
            return 0
        return 1 << self.d # 2**self.d

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

    def conv2mode(self, mode=None, tau=None):
        raise NotImplementedError()

    def copy(self, copy_x=True):
        raise NotImplementedError()

    def full(self):
        res = self.copy(copy_x=False)
        if self.mode == MODE_NP and self.isnotnone:
            res.x = self.x.copy()
        if self.mode == MODE_TT and self.isnotnone:
            res.x = self.x.full()
            if self.kind == VECTOR:
                res.x = res.x.flatten('F')
        if self.mode == MODE_SP and self.isnotnone:
            res.x = self.x.copy()
            if self.kind == MATRIX:
                res.x = res.x.toarray()
        res.mode = MODE_NP
        return res

    def round(self, tau=None):
        res = self.copy(copy_x=False)
        if tau is not None:
            if isinstance(tau, (list, np.ndarray)):
                tau = _max_tau(tau)
            res.tau = tau
        if self.isnotnone:
            if self.mode == MODE_TT:
                res.x = self.x.round(res.tau)
            else:
                res.x = self.x.copy()
        return res

    def norm(self):
        if self.isnone:
            return None
        if self.mode == MODE_NP:
            return np_norm(self.x)
        if self.mode == MODE_TT:
            return self.x.norm()
        if self.mode == MODE_SP and self.kind == VECTOR:
            return np_norm(self.x)
        if self.mode == MODE_SP and self.kind == MATRIX:
            return sp_norm(self.x)

    def rel_err(self, other):
        '''
        Return || self - other || / || self ||
        (if self is None or other is None it returns None).
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
        if self.mode == MODE_NP:
            res.x = np_kron(self.x, other.x)
        if self.mode == MODE_TT:
            res.x = tt_kron(other.x, self.x)
            res = res.round([self.tau, other.tau])
        if self.mode == MODE_SP and self.kind == VECTOR:
            res.x = np_kron(self.x, other.x)
        if self.mode == MODE_SP and self.kind == MATRIX:
            res.x = sp_kron(self.x, other.x)
        return res

    def __add__(self, other):
        res = self.copy(copy_x=False)
        if isinstance(other, (int, float)) and self.isnotnone:
            res.x = self.x + other
            return res.round()
        if self.isnone or other.isnone or self.mode != other.mode:
            raise ValueError('Incorrect input.')
        if self.d != other.d:
            raise ValueError('Mismatch in input data sizes')
        res.x = self.x + other.x
        if self.mode == MODE_TT:
            res = res.round([self.tau, other.tau])
        return res

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        res = self.copy(copy_x=False)
        if isinstance(other, (int, float)) and self.isnotnone:
            res.x = self.x - other
            return res.round()
        if self.isnone or other.isnone or self.mode != other.mode:
            raise ValueError('Incorrect input.')
        if self.d != other.d:
            raise ValueError('Mismatch in input data sizes')
        res.x = self.x - other.x
        if self.mode == MODE_TT:
            res = res.round([self.tau, other.tau])
        return res

    def __rsub__(self, other):
        if not isinstance(other, (int, float)):
            raise ValueError('Incorrect input.')
        return (-1.)*self.__sub__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __unicode__(self):
        s = '> %s-%s'%(self.mode, self.kind)
        s+= ' %-20s'%('|%s|'%self.name)
        s+= ' [tau=%-8.2e]'%(self.tau)
        s+= ' {d =%2d}'%self.d if self.d else ''
        s+= ' (erank =%5.1f)'%self.erank if self.erank else ''
        s+= ' x = None' if self.isnone else ''
        return s

    def __str__(self):
        return self.__unicode__()

    @staticmethod
    def _max_tau(tau_list):
        return _max_tau(tau_list)

def _max_tau(tau_list):
    if not isinstance(tau_list, (list, np.ndarray)) or len(tau_list)==0:
        raise ValueError('Incorrect tau.')
    return np.max(tau_list)

def _n2d(n):
    d = n.bit_length() - 1
    return d # np.log2(n)

def _ind2nneg(d, i):
    n = 1 << d
    if i >= n or i < -n:
        raise ValueError('Incorrect index.')
    if i < 0:
        i = n + i
    return i

def _ind2mind(d, ind):
    if ind < 0:
        if ind == -1:
            mind = [1] * d
        else:
            raise ValueError('Only "-1" is supported for negative indices.')
    else:
        mind = []
        for i in xrange(d):
            mind.append(ind % 2)
            ind /= 2
        if ind > 0:
            raise ValueError('Index is out of range.')
    return mind
