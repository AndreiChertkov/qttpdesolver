# -*- coding: utf-8 -*-
from ..solver import Solver
from ...tensor_wrapper import Vector, Matrix

class SolverFD_2d(Solver):
    '''
    2D Finite difference (FD) solver for elliptic PDEs of the form 
    -div(k grad u) = f. See parent class Solver for more details.
    '''
    
    def _gen_coefficients(self, PDE, GRD, d, n, h, dim, mode, tau, verb):
        self.Kx = Vector.func([GRD.xc, GRD.yr], PDE.k_func, None, 
                              verb, 'Kx').diag()
        self.Ky = Vector.func([GRD.xr, GRD.yc], PDE.k_func, None, 
                              verb, 'Ky').diag()
        self.f  = Vector.func([GRD.xr, GRD.yr], PDE.f_func, None,
                              verb, 'f')
    
    def _gen_matrices(self, d, n, h, dim, mode, tau, verb):
        I = Matrix.eye(d, mode, tau)
        
        iB = Matrix.findif(d, mode, tau, h)        
        self.iBx = I.kron(iB)   
        self.iBy = iB.kron(I) 

        Z = Matrix.unit(d, mode, tau, -1, -1)
        self.Zx = I.kron(Z) 
        self.Zy = Z.kron(I)
        
        I2 = Matrix.eye(2*d, mode, tau)
        self.Sx = I2 - self.Zx 
        self.Sy = I2 - self.Zy   
           
    def _gen_system(self, d, n, h, dim, mode, tau, verb):           
        self.A = self.Sx.dot(self.iBx.T).dot(self.Kx).dot(self.iBx).dot(self.Sx)
        self.A+= self.Sy.dot(self.iBy.T).dot(self.Ky).dot(self.iBy).dot(self.Sy)
        self.A+= self.Zx + self.Zy
        self.rhs = self.Sy.dot(self.Sx.dot(self.f))
        
    def _gen_solution(self, PDE, LSS, d, n, h, dim, mode, eps, tau, verb):
        self.u = LSS.solve(self.A, self.rhs, eps, tau, PDE.sol0, verb)        
        self.ux = self.iBx.dot(self.u)
        self.uy = self.iBy.dot(self.u)