# -*- coding: utf-8 -*-
import time
import numpy as np
import tt
from tt.cross import rect_cross

from . import MODE_NP, MODE_TT, MODE_SP, Vector
from ..utils.capture_output import CaptureCross

class Func(CaptureCross):

    def __init__(self, dim, name='func', desc='', func=None, expr='', args={}, isXYZ=True):
        self.dim = dim
        self.set_name(name)
        self.set_desc(desc)
        self.set_func(func)
        self.set_expr(expr)
        self.set_args(args)
        self.isXYZ = isXYZ
        self.time = None
        self.evals_real = None

    @property
    def desc_full(self):
        return '%-10s = %s'%(self.name, self.desc)

    def set_name(self, name='func'):
        self.name = name
        return self

    def set_desc(self, desc=''):
        self.desc = desc.replace('np.', '')
        return self

    def set_func(self, func=None):
        self.func = func
        return self

    def set_expr(self, expr=''):
        self.expr = expr
        if not self.desc:
            self.set_desc(self.expr)
        return self

    def set_args(self, *args, **kwargs):
        self.args = {}
        if args:
            self.args.update(args[0])
        if kwargs:
            self.args.update(kwargs)
        return self

    def add_arg(self, name, value):
        self.args[name] = value
        return self

    def get(self, r):
        '''
        Input r is a 2-d array [n_poi, n_dim], where
            r[:, 0] - is corresponding to x,
            r[:, 1] - is corresponding to y, ...
        '''
        if len(r) != self.dim:
            r = [r[:, i] for i in range(r.shape[1])]
        if(self.isXYZ):
            args = {['x', 'y', 'z'][i]: r[i] for i in range(len(r))}
        else:
            args = {'r': r}
        args.update(self.args)
        if self.func:
            return self.func(**args)
        if self.expr:
            args.update({'np': np})
            return eval(self.expr, args)
        raise ValueError('Func or expr should be set.')

    def get_inv(self, *r):
        return 1./self.get(*r)

    def build(self, vlist, f0=None, verb=False, inv=False, to_diag=False):
        '''
        Apply the function to a list of Vectors (vlist).
        If initial guess (f0) is not set, it is selected as vlist[0]
        For MODE_NP and MODE_SP function should expects input in the form
            vlist[0].x, vlist[1].x,..., vlist[-1].x,
            where args are float or numpy arrays,
        For MODE_TT function should expects input in the form
            r, where r is a 2-d array
            (r[0, :] - is corresponding to vlist[0],
             r[1, :] - is corresponding to vlist[1], ...)
        '''
        self.time = time.time()
        if not isinstance(vlist, list) or not isinstance(vlist[0], Vector):
            raise ValueError('Incorrect input.')
        f = vlist[0].copy(copy_x=False)
        for v in vlist:
            if not isinstance(v, Vector) or f.d != v.d or f.mode != v.mode:
                raise ValueError('Incorrect input.')
        f.tau = Vector._max_tau([v.tau for v in vlist])
        f.name = self.name
        get = self.get if not inv else self.get_inv
        if f.mode == MODE_NP or f.mode == MODE_SP:
            inp = np.vstack(([v.x for v in vlist])).T
            f.x = get(inp)
        if f.mode == MODE_TT:
            inp = [v.x for v in vlist]
            if verb:
                print '  Construction of %s'%f.name
            f0 = vlist[0].copy() if f0 is None else f0
            self.start_capture()
            f.x = tt.multifuncrs2(inp, get, f.tau, verb=True, y0=f0.x)
            self.stop_capture()
            f = f.round()
        self.time = time.time() - self.time
        if verb:
            print self.present(f)
        return f if not to_diag else f.diag()

    def buildt(self, sh, lim=None, fgrid='cheb', eps=1.E-8, F0=None, verb=False, i0=0):
        '''
        ___DESCRIPTION
        Interpolate function on given dim-dimensional (dim>=1) mesh.
        The mesh should be given as a function (fgrid) that returns
        dim-dimensional point for dim-dimensional index.
        If fgrid is 'cheb', then interpolation is performed on Chebyshev mesh:
        points for every axis i are calculated as x = cos(ind[i]*pi/(sh[i]+1))
        are scaled according to the given limits lim[i, :].
        ___INPUT
        sh    - total numbers of points for every dimension
                type: ndarray [dim] of int
        lim   - [None] min-max values of variable for each dimension
                type: ndarray [dim, 2] of float
        fgrid - ['cheb'] type of grid or function that return multidim point
                type:
                      string ('cheb') or
                      function
                      (ndarray [dim-i0] of int) -> ndarray [dim-i0] of float
        eps   - [1.E-8] accuracy of approximation
                type: float
        F0    - [None] initial guess for result
                type: tensor_wrapper.Tensor
        verb  - [False] if true, the interpolation result will be presented
                type: bool
        i0    - [0] number of dimensions (from the first) for which
                simple indexation is used (index is not transforming to mesh)
                type: int, <= dim
        ___OUTPUT
        F     - the interpolated tensor
                type: tensor_wrapper.Tensor
        '''
        calc_num = [0]
        self.time = time.time()
        def _fgrid(ind):
            if callable(fgrid):
                return fgrid(ind)
            if fgrid == 'cheb':
                t = np.cos(np.pi*ind/(sh[i0:] - 1))
                t*= (lim[i0:, 1]-lim[i0:, 0])/2.
                t+= (lim[i0:, 1]+lim[i0:, 0])/2.
                return t
            raise ValueError('Incorrect grid type "%s" (fgrid)'%fgrid)

        def fvals(ind):
            f_all = np.zeros(ind.shape[0])
            for i in xrange(ind.shape[0]):
                f = tmp.get(tuple(ind[i, :]))
                if f is None:
                    x = np.hstack((ind[i, :i0], _fgrid(ind[i, i0:])))
                    f = self.get(x)
                    tmp[tuple(ind[i, :])] = f
                    calc_num[0] += 1
                f_all[i] = f
            return f_all

        tmp = {}
        F0 = tt.rand(sh, sh.shape[0], 1) if not F0 else F0
        self.start_capture()
        F = rect_cross(fvals, F0, eps=eps, verbose=True)
        self.time = time.time() - self.time
        self.evals_real = calc_num[0]
        self.stop_capture()
        if verb:
            print self.present(F)
        return F

    def present(self, F=None):
        cores_size = 0
        if F and hasattr(F, 'x'):
            F = F.x
        if F and hasattr(F, 'r'):
            G = tt.tensor.to_list(F)
            for GG in G:
                cores_size += GG.size
        s = ''
        if self.evals:
            s+= 'Cross evals    :  %d    \n'%(self.evals)
        if self.evals_real:
            s+= 'Function evals :  %d    \n'%(self.evals_real)
        if cores_size:
            s+= 'TT cores size  :  %d    \n'%(cores_size)
            s+= 'TT ranks       :  %s    \n'%(F.r)
        if self.erank:
            s+= 'Eff TT rank    :  %-6.1f\n'%(self.erank)
        if self.sweep:
            s+= 'Iters          :  %d    \n'%(self.sweep)
        if self.err_rel:
            s+= 'Err rel        :  %-8.2e\n'%(self.err_rel)
        if self.err_abs:
            s+= 'Err abs        :  %-8.2e\n'%(self.err_abs)
        if self.err_dy:
            s+= 'Max dy         :  %-8.2e\n'%(self.err_dy)
        if self.time:
            s+= 'Total time (s) :  %-6.3f\n'%(self.time)
        return s
