# -*- coding: utf-8 -*-
from ..solver import Solver
from ...tensor_wrapper import Vector, Matrix

class SolverFS_1d(Solver):
    '''
    1D Finite Sum (FS) solver for elliptic PDEs of the form 
    -div(k grad u) = f. See parent class Solver for more details.
    '''
    
    def _gen_coefficients(self, PDE, GRD, d, n, h, dim, mode, tau, verb):
        self.iKx = Vector.func([GRD.xc], PDE.k_func, None, 
                               verb, 'iKx', inv=True).diag()
        self.f   = Vector.func([GRD.xr], PDE.f_func, None,
                               verb, 'f')

    def _gen_matrices(self, d, n, h, dim, mode, tau, verb):
        self.Bx = Matrix.volterra(d, mode, tau, h)

    def _gen_system(self, d, n, h, dim, mode, tau, verb):
        pass
    
    def _gen_solution(self, PDE, LSS, d, n, h, dim, mode, eps, tau, verb):
        iKx, f, Bx = self.iKx.diag(), self.f, self.Bx
        g = Bx.T.dot(f) * iKx
        s = g.sum()/iKx.sum()
        self.ux = g - s*iKx
        self.u  = Bx.dot(self.ux)