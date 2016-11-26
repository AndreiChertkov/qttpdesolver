# -*- coding: utf-8 -*-
import numpy as np
import tt

from numpy import diag as np_diag
from tt import diag as tt_diag
from scipy.sparse import diags as sp_diag

from .tensor_base import MODE_NP, MODE_TT, MODE_SP, DEF_TAU
from .tensor_base import DEF_TENSOR_NAME, DEF_VECTOR_NAME, DEF_MATRIX_NAME
from .tensor_base import TensorBase, _ind2mind, _n2d, _max_tau

class Vector(TensorBase):
    
    def __init__(self, x=None, d=None, mode=MODE_NP, tau=DEF_TAU,
                 conv2mode=True, name=DEF_VECTOR_NAME):
        TensorBase.__init__(self, x, d, mode, tau, conv2mode, name)
        if self.name == DEF_TENSOR_NAME or self.name == DEF_MATRIX_NAME:
            self.name = DEF_VECTOR_NAME
        self.type = 'vector'

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
                self.x = tt.vector(self.x.reshape([2]*self.d, order='F'), 
                                   eps=self.tau)
        elif isinstance(self.x, tt.vector):
            self.d = self.x.d
            if self.mode == MODE_NP or self.mode == MODE_SP:
                self.x = self.x.full().flatten('F')
        else:
            raise ValueError('Incorrect vector.')
            
    def copy(self, copy_x=True):
        x = None
        if copy_x and self.isnotnone:
            x = self.x.copy()
        return Vector(x, self.d, self.mode, self.tau, False, self.name)
    
    def full(self):
        res = self.copy(copy_x=False)
        res.mode = MODE_NP
        if self.mode == MODE_TT and self.isnotnone:
            res.x = self.x.full().flatten('F')
        return res
    
    def diag(self):
        from matrix import Matrix
        res = Matrix(None, self.d, self.mode, self.tau, False)
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
        if self.mode == MODE_NP or self.mode == MODE_SP:
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
        res.d = self.d / dim *(dim-1)
        if self.mode == MODE_NP or self.mode == MODE_SP:
            x_new = self.x.reshape(tuple([2**(self.d/dim)]*dim), order='F')
            res.x = np.sum(x_new, axis=axis).flatten('F')
        if self.mode == MODE_TT:
            axs = [np.arange(self.d/dim), np.arange(self.d/dim, self.d/dim*2), 
                   np.arange(self.d/dim*2, self.d/dim*3)][axis]
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

    def half(self, ind):
        ''' 
        Get a first (ind=0) or second (ind=1) half of array.
        It transform d-dim array to (d-1)-dim array.
        '''
        if ind != 0 and ind != 1:
            raise ValueError('Only 0,1 values for index are valid!')
        res = self.copy(copy_x=False)
        res.d-= 1
        if self.mode == MODE_NP or self.mode == MODE_SP:
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
   
    def inv(self, v0=None, verb=False, name=DEF_VECTOR_NAME):
        ''' 
        Return inverse to the vector.
                Input:
        v0      - is a Vector that is an initial guess for 1/x (it has sense
                  only for MODE_TT while cross appr. If v0=None, then v0=self)
        verb    - if is True, then ouput will be printed (only for MODE_TT)
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
        if self.isnone or other.isnone or self.mode != other.mode or self.d != other.d:
            raise ValueError('Incorrect input.')
        res.x = self.x * other.x
        return res.round([self.tau, other.tau])
        
    @staticmethod
    def func(vlist, func, v0=None, verb=False, name=DEF_VECTOR_NAME, inv=False):
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
                res.x = tt.multifuncrs2([v.x for v in vlist], 
                                        func, res.tau, 
                                        verb=verb, y0=v0.x)
            else:
                res.x = tt.multifuncrs2([v.x for v in vlist], 
                                        lambda x: 1./func(x), res.tau, 
                                        verb=verb, y0=v0.x)
            res = res.round()
        return res

    @staticmethod
    def unit(d, mode=MODE_NP, tau=None, i=0, val=1., name=DEF_VECTOR_NAME):
        '''
        Construct a vector in the given mode of length 2**d with only one
        nonzero element in position i, that is equal to a given value val.
        '''
        res = Vector(None, d, mode, tau, False, name=name)
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
    def ones(d, mode=MODE_NP, tau=None, name=DEF_VECTOR_NAME):
        res = Vector(None, d, mode, tau, False, name=name)
        if mode == MODE_NP or mode == MODE_SP:
            res.x = np.ones(res.n)
        if mode == MODE_TT:
            res.x = tt.ones(2, d)
        return res
        
    @staticmethod
    def arange(d, mode=MODE_NP, tau=None, name=DEF_VECTOR_NAME):
        res = Vector(None, d, mode, tau, False, name=name)
        if mode == MODE_NP or mode == MODE_SP:
            res.x = np.arange(res.n)
        if mode == MODE_TT:
            res.x = tt.xfun(2, d)
        return res  
        
    @staticmethod
    def block(vlist, name=DEF_VECTOR_NAME):
        '''
        Concatenate vectors from a list. All vectors must have the same size..
        List may contain some (not all!) int/float or None entries,
        that will be filled by zero-vectors.
        '''
        res = None
        if not isinstance(vlist, list):
            raise ValueError('This should be a list.')
        d0 = _n2d(len(vlist))
        for i, v in enumerate(vlist):
            if (v is None or isinstance(v, (int, float))) or \
               (isinstance(v, Vector) and v.isnone):
                continue
            e = Vector.unit(d0, v.mode, v.tau, i)
            if res is None:
                res = e.kron(v)
            else:
                res+= e.kron(v)
        res.name = name
        return res       