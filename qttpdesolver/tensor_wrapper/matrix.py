# -*- coding: utf-8 -*-
from numba import jit
import numpy as np
import tt
from numpy import diag as np_diag
from tt import diag as tt_diag
from scipy.sparse import diags as sp_diag
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix

from .tensor_base import MODE_NP, MODE_TT, MODE_SP, DEF_TAU, MATRIX
from .tensor_base import TensorBase, _ind2mind, _ind2nneg, _n2d
from .vector import Vector

class Matrix(TensorBase):
    kind = MATRIX

    @property
    def T(self):
        res = self.copy(copy_x=False)
        res.x = self.x.T
        return res

    def conv2mode(self, mode=None, tau=None):
        if mode is not None:
            self.mode = mode
        if tau is not None:
            self.tau = tau
        if self.x is None:
            return
        elif isinstance(self.x, np.ndarray):
            if len(self.x.shape) != 2 or self.x.shape[0] != self.x.shape[1]:
                raise ValueError('Incorrect shape.')
            self.d = _n2d(self.x.shape[0])
            if self.mode == MODE_TT:
                self.x = tt.matrix(self.x.reshape([2]*(2*self.d), order='F'), eps=self.tau)
            if self.mode == MODE_SP:
                raise NotImplementedError()
        elif isinstance(self.x, tt.matrix):
            self.d = self.x.tt.d
            if self.mode == MODE_NP:
                self.x = self.x.full()
            if self.mode == MODE_SP:
                raise NotImplementedError()
        elif isinstance(self.x, csr_matrix):
            if self.mode == MODE_NP or self.mode == MODE_TT:
                self.x = self.x.toarray()
                self.conv2mode()
            else:
                if len(self.x.shape) != 2 or self.x.shape[0] != self.x.shape[1]:
                    raise ValueError('Incorrect shape.')
                self.d = _n2d(self.x.shape[0])
        else:
            raise ValueError('Incorrect matrix.')

    def copy(self, copy_x=True):
        x = self.x.copy() if copy_x and self.isnotnone else None
        return Matrix(x, self.d, self.mode, self.tau, False, self.name)

    def diag(self):
        res = Vector(None, self.d, self.mode, self.tau, False, self.name)
        if self.isnone:
            return res
        if self.mode == MODE_NP:
            res.x = np_diag(self.x)
        if self.mode == MODE_TT:
            res.x = tt_diag(self.x)
        if self.mode == MODE_SP:
            res.x = sp_diag(self.x)
        return res

    def dot(self, other):
        if isinstance(other, Vector):
            return self.matvec(other)
        if not isinstance(other, Matrix):
            raise NotImplementedError()
        if self.isnone or other.isnone or self.mode != other.mode:
            raise ValueError('Incorrect input.')
        if self.d != other.d:
            raise ValueError('Mismatch in input data sizes')
        res = self.copy(copy_x=False)
        if self.mode == MODE_NP or self.mode == MODE_SP:
            res.x = self.x.dot(other.x)
        if self.mode == MODE_TT:
            res.x = self.x * other.x
            res = res.round([self.tau, other.tau])
        return res

    def matvec(self, other):
        if not isinstance(other, Vector):
            raise ValueError('Incorrect input.')
        if self.isnone or other.isnone or self.mode != other.mode:
            raise ValueError('Incorrect input.')
        if self.d != other.d:
            raise ValueError('Mismatch in input data sizes')
        res = other.copy(copy_x=False)
        if self.mode == MODE_NP or self.mode == MODE_SP:
            res.x = self.x.dot(other.x)
        if self.mode == MODE_TT:
            res.x = tt.matvec(self.x, other.x)
            res = res.round([self.tau, other.tau])
        return res

    def __mul__(self, other):
        res = self.copy(copy_x=False)
        if isinstance(other, (int, float)) and self.isnotnone:
            res.x = self.x * other
            return res
        if self.isnone or other.isnone or self.mode != other.mode:
            raise ValueError('Incorrect input.')
        if self.d != other.d:
            raise ValueError('Mismatch in input data sizes')
        if self.mode == MODE_NP or self.mode == MODE_SP:
            res.x = self.x * other.x
        if self.mode == MODE_TT:
            a = tt.vector.from_list([G.reshape((G.shape[0], 4, G.shape[-1]))
                                     for G in tt.matrix.to_list(self.x)])
            b = tt.vector.from_list([G.reshape((G.shape[0], 4, G.shape[-1]))
                                     for G in tt.matrix.to_list(other.x)])
            C = tt.matrix.from_list([G.reshape((G.shape[0], 2, 2, G.shape[-1]))
                                     for G in tt.vector.to_list(a*b)])
            res.x = C
            res = res.round([self.tau, other.tau])
        return res

    @staticmethod
    def unit(d, mode=MODE_NP, tau=None, i=0, j=0, val=1., name=MATRIX):
        '''
        Construct a matrix in the given mode of shape 2**d with only one
        nonzero element in position [i, j], that is equal to a given value val.
        '''
        i = _ind2nneg(d, i)
        j = _ind2nneg(d, j)
        res = Matrix(None, d, mode, tau, False, name=name)
        if mode == MODE_NP or mode == MODE_SP:
            res.x = coo_matrix(([val], ([i], [j])), shape=(res.n, res.n))
            res.x = res.x.tocsr()
            if mode == MODE_NP:
                res.x = res.x.toarray()
        if mode == MODE_TT:
            mind_i = _ind2mind(d, i)
            mind_j = _ind2mind(d, j)
            Gl = []
            for k in xrange(d):
                G = np.zeros((1, 2, 2, 1))
                G[0, mind_i[k], mind_j[k], 0] = 1.
                Gl.append(G)
            res.x = val*tt.matrix.from_list(Gl)
        return res

    @staticmethod
    def eye(d, mode=MODE_NP, tau=None, name=MATRIX):
        '''
        Eye matrix: diag([1, 1,..., 1]).
        '''
        res = Matrix(None, d, mode, tau, False, name=name)
        if mode == MODE_NP:
            res.x = np.eye(res.n)
        if mode == MODE_TT:
            res.x = tt.eye(2, d)
        if mode == MODE_SP:
            res.x = sp_diag([np.ones(res.n)], [0], format='csr')
        return res

    @staticmethod
    def ones(d, mode=MODE_NP, tau=None, name=MATRIX):
        '''
        Matrix of all ones.
        '''
        res = Matrix(None, d, mode, tau, False, name=name)
        if mode == MODE_NP:
            res.x = np.ones((res.n, res.n))
        if mode == MODE_TT:
            res.x = tt.eye(2, d)
            res.x.tt = tt.ones(4, d)
        if mode == MODE_SP:
            raise NotImplementedError()
        return res

    @staticmethod
    def volterra(d, mode=MODE_NP, tau=None, h=1., name=MATRIX, isperiodic=False):
        '''
        Volterra integral 1D matrix
        [i, j] = h if i>=j and = 0 otherwise.
        '''
        res = Matrix(None, d, mode, tau, False, name)
        if mode == MODE_NP:
            res.x = h * toeplitz(c=np.ones(res.n), r=np.zeros(res.n))
        if mode == MODE_TT:
            res.x = h * (tt.Toeplitz(tt.ones(2, d), kind='U') + tt.eye(2, d))
        if mode == MODE_SP:
            raise NotImplementedError()
        if isperiodic:
            E = Matrix.ones(d, mode, tau)
            res = res - h*(E.dot(res) + res.dot(E)) + h*(h+1)/2 * E
            #from ...tensor_wrapper.matrix import toeplitz
            #c = 1.5 + 0.5*h - h*np.arange(n+1, 2*n+1, 1.)
            #r = 0.5 + 0.5*h - h*np.arange(n+1, 1, -1.)
            #B = Matrix(h*toeplitz(c=c, r=r), d, mode, tau)
        if mode == MODE_TT:
            res = res.round()
        return res

    @staticmethod
    def findif(d, mode=MODE_NP, tau=None, h=1., name=MATRIX, isperiodic=False):
        '''
        Finite difference 1D matrix
        [i, j] = 1/h if i=j, [i, j] =-1/h if i=j+1 and = 0 otherwise.
        '''
        res = Matrix(None, d, mode, tau, False, name)
        if mode == MODE_NP or mode == MODE_SP:
            res.x = sp_diag([np.ones(res.n), -np.ones(res.n-1)], [0, -1], format='csr')
            res.x*= 1./h
            if mode == MODE_NP:
                res.x = res.x.toarray()
        if mode == MODE_TT:
            e1 = tt.tensor(np.array([0., 1.]))
            e2 = tt.mkron([e1] * d)
            res.x = tt.Toeplitz(-e2, kind='U') + tt.eye(2, d)
            res.x*= (1./h)
            res = res.round()
        if isperiodic:
            res = res + Matrix.unit(d, mode, tau, i=0, j=-1, val=-1./h)
        return res

    @staticmethod
    def findifp(d, mode=MODE_NP, tau=None, h=1., name=MATRIX):
        '''
        Periodic finite difference 1D matrix
        [i, j] = 1/h if i=j, [i, j] =-1/h if i=j+1 or i=0,j=-1 and = 0 otherwise.
        '''
        return Matrix.findif(d, mode, tau, h, name, True)

    @staticmethod
    def shift(d, mode=MODE_NP, tau=None, name=MATRIX):
        '''
        Shift 1D matrix
        [i, j] = 1 if i=j+1, [i, j] = 0 otherwise.
        '''
        I = Matrix.eye(d, mode, tau, name=name)
        B = Matrix.findif(d, mode, tau, name=name)
        return (-1.)*B + I

    @staticmethod
    def block(mlist, name=MATRIX):
        '''
        Construct square block matrix from list of lists, that contains matrices
        of equal shape and maybe some (not all!) None/int/float entries.
        For None/int/float entries elements in the corresponding block
        will be filled by zeros.
        '''
        res = None
        if not isinstance(mlist, list):
            raise ValueError('This should be a list.')
        d0 = _n2d(len(mlist))
        n0 = 2**d0
        for i, mrow in enumerate(mlist):
            if not isinstance(mrow, list):
                raise ValueError('List of lists should contain only lists.')
            if not n0 == len(mrow):
                raise ValueError('The length of the list and sub-lists should be equal.')
            for j, m in enumerate(mrow):
                if (m is None or isinstance(m, (int, float))) or \
                   (isinstance(m, Matrix) and m.isnone):
                    continue
                e = Matrix.unit(d0, m.mode, m.tau, i, j)
                if res is None:
                    res = e.kron(m)
                else:
                    res+= e.kron(m)
        res.name = name
        return res

#    @staticmethod
#    def interpol_1d(d=None, mode=MODE_NP, tau=None, name=MATRIX):
#        '''
#        1D interpolation matrix.
#        Is valid only for mesh x_j = h+hj, j=0,1,...,n-1.
#        '''
#        raise NotImplementedError()
#        res = Matrix._prep_spec_matrix(d, mode, tau, name)
#        if mode == MODE_NP:
#            x = np.zeros((2*res.n, 2*res.n))
#            for j in range(res.n):
#                x[2*j+0, j*2] = 1
#                x[2*j+1, j*2] = 2
#                if j<res.n-1:
#                    x[2*j+2, j*2] = 1
#            x = 0.5*x
#        if mode == MODE_TT:
#            e1 = tt.matrix(np.array([[1., 0.], [2., 0.]]))
#            e2 = tt.tensor(np.array([0.0, 1.0]))
#            e3 = tt.matrix(np.array([[1., 0.], [0., 0.]]))
#            x1 = kronm([Matrix.eye(d, mode, tau), e1], tau)
#            x2 = tt.Toeplitz(tt.mkron([e2] * d), kind='U')
#            x2 = kronm([x2, e3], tau)
#            x = 0.5*(x1+x2)
#        if mode == MODE_SP:
#            raise NotImplementedError()
#        res.x = x
#        return res

#    @staticmethod
#    def eye_except_last(d, mode=MODE_NP, tau=None, name=MATRIX):
#        '''
#        Eye matrix with zero last element: diag(1, 1,..., 1, 0).
#        '''
#        return (-1.)*Matrix.zeros_except_last(d, mode, tau, name=name)\
#                    +Matrix.eye(d, mode, tau, name=name)
#        res = Matrix(None, d, mode, tau, False, name)
#        if mode == MODE_NP or mode == MODE_SP:
#            res.x = coo_matrix((np.ones(res.n-1), (np.arange(res.n-1), np.arange(res.n-1))),
#                               shape=(res.n, res.n)).tocsr()
#            if mode == MODE_NP:
#                res.x = res.x.toarray()
#        if mode == MODE_TT:
#            e1 = tt.tensor(np.array([0.0, 1.0]))
#            e2 = tt.mkron([e1] * d)
#            res.x = tt.diag(tt.ones(2, d)-e2)
#            res = res.round()
#        return res

#    @staticmethod
#    def zeros_except_last(d, mode=MODE_NP, tau=None, name=DEF_MATRIX_NAME):
#        '''
#        Zero-matrix with nonzero last element: diag(0, 0,..., 0, 1).
#        '''
#        return Matrix.zeros_except_one(d, mode, tau, 2**d-1, 2**d-1, 1., name)
#        res = Matrix(None, d, mode, tau, False, name)
#        if mode == MODE_NP or mode == MODE_SP:
#            res.x = coo_matrix(([1], ([res.n-1], [res.n-1])),
#                               shape=(res.n, res.n)).tocsr()
#            if mode == MODE_NP:
#                res.x = res.x.toarray()
#        if mode == MODE_TT:
#            e1 = tt.tensor(np.array([0.0, 1.0]))
#            e2 = tt.mkron([e1] * d)
#            res.x = tt.diag(e2)
#            res = res.round()
#        return res

@jit(nopython=True)
def toeplitz(c, r):
    '''
    Toeplitz matrix with c is the first column and
    r[1:] is the first row (c[0] is used instead of r[0]).
    '''
    n = c.shape[0]
    a = np.zeros((n, n))
    for i in xrange(n):
        for j in xrange(i+1):
            a[i, j] = c[i - j]
        for j in xrange(i+1, n):
            a[i, j] = r[j - i]
    return a
