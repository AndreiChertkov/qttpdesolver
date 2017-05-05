# -*- coding: utf-8 -*-
from ..solver import BC_PR, Solver
from ...tensor_wrapper import Vector, Matrix

class SolverFS_NH_2d(Solver):
    '''
    2D Finite Sum (FS) solver for numerical homogenization
    of multiscale elliptic PDEs of the form 
    -div(k grad u) = f. See parent class Solver for more details.
    '''
    
    def _gen_coefficients(self, PDE, GRD, d, n, h, dim, mode, tau, verb):
        self.Kx1  = Vector.func([GRD.xc*PDE.params[0]], PDE.k_func, None, 
                                verb, 'iKx1')
        self.iKx1 = self.Kx1.inv(None, verb, 'iKx1').diag()
        self.Kx1  = self.Kx1.diag()
        self.f    = Vector.func([GRD.xr], PDE.f_func, None,
                                verb, 'f')
            
    def _gen_matrices(self, d, n, h, dim, mode, tau, verb):
        E = Matrix.ones(d, mode, tau)

        B = Matrix.volterra(d, mode, tau, h)     
        
        G = B - h*(E.dot(B) + B.dot(E)) + h*(h+1)/2 * E
        #from ...tensor_wrapper.matrix import toeplitz
        #c = 1.5 + 0.5*h - h*np.arange(n+1, 2*n+1, 1.)
        #r = 0.5 + 0.5*h - h*np.arange(n+1, 1, -1.)
        #B = Matrix(h*toeplitz(c=c, r=r), d, mode, tau)

        iB = Matrix.findif(d, mode, tau, h)
        
        self.Bx = B
        self.Gx = G 
        self.iGx = iB + Matrix.unit(d, mode, tau, i=0, j=-1, val=-1./h)
        
    def _gen_system(self, d, n, h, dim, mode, tau, verb):
        pass
    
    def _gen_solution(self, PDE, LSS, d, n, h, dim, mode, eps, tau, verb):
        kx1, ikx1 = self.Kx1.diag(), self.iKx1.diag()
        
        f0 = -1.*self.iGx.T.dot(kx1)
        g0 = self.Gx.T.dot(f0)*ikx1
        s0 = g0.sum()/ikx1.sum()
        ux_cell = g0 - s0*ikx1
               
        e = Vector.ones(PDE.d, PDE.mode, PDE.tau)
        kx_hom = (kx1*(e + ux_cell)).sum()*h**dim
        ikx = 1./kx_hom * e
        
        if self.PDE.bc != BC_PR: 
            Bx = self.Bx
        else:
            Bx = self.Gx
            
        g = Bx.T.dot(self.f) * ikx
        s = g.sum()/ikx.sum()
        self.ux = g - s*ikx
        self.u  = Bx.dot(self.ux)
        self.ux*= e + ux_cell