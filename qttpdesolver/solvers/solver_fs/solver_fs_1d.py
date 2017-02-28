# -*- coding: utf-8 -*-
from ..solver import BC_PR, Solver
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
        B = Matrix.volterra(d, mode, tau, h)
        if self.PDE.bc == BC_PR:      
            E = Matrix.ones(d, mode, tau)
            B = B - h*(E.dot(B) + B.dot(E)) + h*(h+1)/2 * E
            #from ...tensor_wrapper.matrix import toeplitz
            #c = 1.5 + 0.5*h - h*np.arange(n+1, 2*n+1, 1.)
            #r = 0.5 + 0.5*h - h*np.arange(n+1, 1, -1.)
            #B = Matrix(h*toeplitz(c=c, r=r), d, mode, tau)
        self.Bx = B
        
    def _gen_system(self, d, n, h, dim, mode, tau, verb):
        pass
    
    def _gen_solution(self, PDE, LSS, d, n, h, dim, mode, eps, tau, verb):
        ikx, f, Bx = self.iKx.diag(), self.f, self.Bx
        g = Bx.T.dot(f) * ikx
        s = g.sum()/ikx.sum()
        self.ux = g - s*ikx
        self.u  = Bx.dot(self.ux)