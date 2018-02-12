# -*- coding: utf-8 -*-
import numpy as np
import tt
from numpy import diag as np_diag
from tt import diag as tt_diag
from scipy.sparse import diags as sp_diag
from scipy.linalg import block_diag as sp_block_diag

from .tensor_base import MODE_NP, MODE_TT, MODE_SP, DEF_TAU, VECTOR
from .tensor_base import TensorBase, _ind2nneg, _ind2mind, _n2d, _max_tau

class Vector(TensorBase):
    kind = VECTOR

    def conv2mode(self, mode=None, tau=None):
        if mode is not None:
            self.mode = mode
        if tau is not None:
            self.tau = tau
        if self.x is None:
            return
        elif isinstance(self.x, np.ndarray):
            if len(self.x.shape) != 1:
                raise ValueError('Incorrect shape.')
            self.d = _n2d(self.x.shape[0])
            if self.mode == MODE_TT:
                self.x = tt.vector(self.x.reshape([2]*self.d, order='F'), eps=self.tau)
        elif isinstance(self.x, tt.vector):
            self.d = self.x.d
            if self.mode in [MODE_NP, MODE_SP]:
                self.x = self.x.full().flatten('F')
        else:
            raise ValueError('Incorrect vector.')

    def copy(self, copy_x=True):
        x = self.x.copy() if copy_x and self.isnotnone else None
        return Vector(x, self.d, self.mode, self.tau, False, self.name)

    def diag(self):
        from matrix import Matrix
        res = Matrix(None, self.d, self.mode, self.tau, False, self.name)
        if self.isnone:
            return res
        if self.mode == MODE_NP:
            res.x = np_diag(self.x)
        if self.mode == MODE_TT:
            res.x = tt_diag(self.x)
        if self.mode == MODE_SP:
            res.x = sp_diag([self.x], [0], format='csr')
        return res

    def sum(self):
        if self.isnone:
            return None
        if self.mode in [MODE_NP, MODE_SP]:
            return np.sum(self.x)
        if self.mode == MODE_TT:
            return tt.sum(self.x)

    def sum_out(self, dim, axis):
        '''
        Reshape vector into dim-dimensional array, construct sum under given
        axis number and return the vector that is the flatten result.
        It should be d%dim == 0; dim=2, 3; axis=0, 1, 2 and axis<dim.
        '''
        if self.isnone:
            return None
        if self.d%dim != 0 or not dim in [2, 3] or not axis in [0, 1, 2] or not axis < dim:
            raise ValueError('Incorrect input.')
        res = self.copy(copy_x=False)
        dd = self.d/dim
        res.d = dd *(dim-1)
        if self.mode in [MODE_NP, MODE_SP]:
            x_new = self.x.reshape(tuple([2**dd]*dim), order='F')
            res.x = np.sum(x_new, axis=axis).flatten('F')
        if self.mode == MODE_TT:
            axs = np.arange(dd*axis, dd*(axis+1))
            Gl, Gl_new, G0 = tt.tensor.to_list(self.x), [], None
            for i, G in enumerate(Gl):
                if i in axs:
                    G = np.sum(G, axis=1)
                    if G0 is None:
                        G0 = G.copy()
                    else:
                        G0 = np.tensordot(G0, G, 1)
                else:
                    if G0 is not None:
                        Gl_new.append(np.tensordot(G0, G, 1))
                        G0 = None
                    else:
                        Gl_new.append(G)
            if G0 is not None:
                Gl_new[-1] = np.tensordot(Gl_new[-1], G0, 1)
            res.x = tt.tensor.from_list(Gl_new)
            res = res.round()
        return res

    def kron2e(self):
        '''
        Function construct special kron product (for 3D FS-QTT-solver)
        '''
        d = self.d/2
        n = 2**d
        from matrix import Matrix
        res = Matrix(None, d*3, self.mode, self.tau, False)
        if self.mode in [MODE_NP, MODE_SP]:
            blocks = []
            for i in xrange(n):
                blocks.append(np.kron(np.ones((n, n)),
                                      np.diag(self.x[n*i: n*(i+1)])))
            res.x = sp_block_diag(*blocks)
        else:
            GG = tt.matrix.to_list(self.diag().x)
            for j in range(d):
                i = d
                r = GG[i-1].shape[-1]
                GGnew = np.zeros((r, 2, 2, r))
                GGnew[:, 0, 0, :] = np.eye(r)
                GGnew[:, 1, 0, :] = np.eye(r)
                GGnew[:, 0, 1, :] = np.eye(r)
                GGnew[:, 1, 1, :] = np.eye(r)
                GG.insert(i, GGnew)
            res.x = tt.matrix.from_list(GG)
        return res

    def half(self, ind):
        '''
        Transform d-dim array to (d-1)-dim array,
        that is the first (ind=0) or the second (ind=1) half of the origin array.
        '''
        if not ind in [0, 1]:
            raise ValueError('Only 0,1 values for index are valid!')
        res = self.copy(copy_x=False)
        res.d-= 1
        if self.mode in [MODE_NP, MODE_SP]:
            if ind==0:
                res.x = self.x[:self.n/2]
            else:
                res.x = self.x[self.n/2:]
        if self.mode == MODE_TT:
            Gl_all = tt.tensor.to_list(self.x)
            Gl = Gl_all[:-1]
            Gl[-1] = np.tensordot(Gl[-1], Gl_all[-1][:, ind, :], axes=(-1, 0))
            res.x = tt.tensor.from_list(Gl)
        return res

    def outer(self, v):
        '''
        Construct outer vector product u.dot(v^T) where u=self.
        '''
        if not isinstance(v, Vector) or self.isnone or v.isnone or self.mode != v.mode or self.d != v.d:
            raise ValueError('Incorrect input.')
        from matrix import Matrix
        res = Matrix(None, self.d, self.mode, self.tau, False, self.name)
        if self.mode == MODE_NP:
            res.x = np.outer(self.x, v.x)
        if self.mode == MODE_TT:
            GG = tt.vector.to_list(self.x)
            for i in range(self.d):
                r0, n, r1 = GG[i].shape
                GG[i] = GG[i].reshape((r0, n, 1, r1))
            U1 = tt.matrix.from_list(GG)
            GG = tt.vector.to_list(v.x)
            for i in range(self.d):
                r0, n, r1 = GG[i].shape
                GG[i] = GG[i].reshape((r0, 1, n, r1))
            V1 = tt.matrix.from_list(GG)
            res.x = U1 * V1
        if self.mode == MODE_SP:
            raise ValueError('It is not work for MODE_SP.')
        return res.round([self.tau, v.tau])

    def inv(self, v0=None, verb=False, name=VECTOR):
        '''
        Return inverse to the vector.
                Input:
        v0      - is a Vector that is an initial guess for 1/x (it has sense
                  only for MODE_TT while cross appr. If v0=None, then v0=self)
        verb    - if is True, then output will be printed (only for MODE_TT)
        name    - is the name for the new vector
                Output:
        iv      - inverse vector (type: Vector) or None (if self.isnone)
        '''
        if self.isnone:
            return None
        return self.func([self], lambda x: 1./x, v0, verb, name)

    def __mul__(self, other):
        res = self.copy(copy_x=False)
        if isinstance(other, (int, float)) and self.isnotnone:
            res.x = self.x * other
            return res
        if self.isnone or other.isnone or self.mode != other.mode:
            raise ValueError('Incorrect input.')
        if self.d != other.d:
            raise ValueError('Mismatch in input data sizes')
        res.x = self.x * other.x
        if self.mode == MODE_TT:
            res = res.round([self.tau, other.tau])
        return res

    @staticmethod
    def func(vlist, func, v0=None, verb=False, name=VECTOR, inv=False):
        '''
        Apply the given function (func) to a list of Vectors (vlist).
        Initial guess (v0) may be set (if is None, then v0=vlist[0])
        For MODE_NP and MODE_SP function should expects input in the form
            vlist[0].x, vlist[1].x,..., vlist[-1].x,
            where args are float or numpy arrays,
        For MODE_TT function should expects input in the form
            r, where r is a 2-d array
            (r[0, :] - is corresponding to vlist[0],
             r[1, :] - is corresponding to vlist[1], ...)
        '''
        if not isinstance(vlist, list) or not isinstance(vlist[0], Vector):
            raise ValueError('Incorrect input.')
        res = vlist[0].copy(copy_x=False)
        res.name = name
        for v in vlist:
            if not isinstance(v, Vector) or res.d != v.d or res.mode != v.mode:
                raise ValueError('Incorrect input.')
        res.tau = _max_tau([v.tau for v in vlist])
        if res.mode == MODE_NP or res.mode == MODE_SP:
            if not inv:
                res.x = func(*[v.x for v in vlist])
            else:
                res.x = 1./func(*[v.x for v in vlist])
        if res.mode == MODE_TT:
            if verb:
                print '  Construction of %s'%res.name
            if v0 is None:
                v0 = vlist[0].copy()
            if not inv:
                res.x = tt.multifuncrs2([v.x for v in vlist], func, res.tau, verb=verb, y0=v0.x)
            else:
                res.x = tt.multifuncrs2([v.x for v in vlist], lambda x: 1./func(x), res.tau, verb=verb, y0=v0.x)
            res = res.round()
        return res

    @staticmethod
    def unit(d, mode=MODE_NP, tau=None, i=0, val=1., name=VECTOR):
        '''
        Construct a vector in the given mode of length 2**d with only one
        nonzero element in position i, that is equal to a given value val.
        '''
        i = _ind2nneg(d, i)
        res = Vector(None, d, mode, tau, False, name)
        if mode == MODE_NP or mode == MODE_SP:
            res.x = np.zeros(res.n)
            res.x[i] = val
        if mode == MODE_TT:
            mind = _ind2mind(d, i)
            Gl = []
            for k in xrange(d):
                G = np.zeros((1, 2, 1))
                G[0, mind[k], 0] = 1.
                Gl.append(G)
            res.x = val*tt.vector.from_list(Gl)
        return res

    @staticmethod
    def ones(d, mode=MODE_NP, tau=None, name=VECTOR):
        res = Vector(None, d, mode, tau, False, name=name)
        if mode == MODE_NP or mode == MODE_SP:
            res.x = np.ones(res.n)
        if mode == MODE_TT:
            res.x = tt.ones(2, d)
        return res

    @staticmethod
    def arange(d, mode=MODE_NP, tau=None, name=VECTOR):
        res = Vector(None, d, mode, tau, False, name=name)
        if mode == MODE_NP or mode == MODE_SP:
            res.x = np.arange(res.n)
        if mode == MODE_TT:
            res.x = tt.xfun(2, d)
        return res

    @staticmethod
    def block(vlist, name=VECTOR):
        '''
        Concatenate vectors from a list. All vectors must have the same size.
        List may contain some (not all!) int/float or None or None-Vector
        entries, that will be filled by zero-vectors.
        '''
        res = None
        if not isinstance(vlist, list):
            raise ValueError('This should be a list.')
        d0 = _n2d(len(vlist))
        for i, v in enumerate(vlist):
            if (v is None or isinstance(v, (int, float))) or (isinstance(v, Vector) and v.isnone):
                continue
            e = Vector.unit(d0, v.mode, v.tau, i)
            if res is None:
                res = e.kron(v)
            else:
                res+= e.kron(v)
        res.name = name
        return res
