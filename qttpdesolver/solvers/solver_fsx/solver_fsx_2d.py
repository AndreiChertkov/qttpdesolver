# -*- coding: utf-8 -*-
from ..solver import BC_HD, BC_PR, Solver
from ...tensor_wrapper import Vector, Matrix

class SolverFSX_2d(Solver):
    '''
    2D Finite Sum (FS) solver for elliptic PDEs of the form 
    -div(k grad u) = f. See parent class Solver for more details.
    '''
    
    def _gen_coefficients(self, PDE, GRD, d, n, h, dim, mode, tau, verb):
        self.iKx = Vector.func([GRD.xc, GRD.yr], PDE.kx_func, None, 
                               verb, 'iKx', inv=True).diag()
        self.iKy = Vector.func([GRD.xr, GRD.yc], PDE.ky_func, None, 
                               verb, 'iKy', inv=True).diag()
        self.Kxy = Vector.func([GRD.xr, GRD.yc], PDE.kxy_func, None, 
                               verb, 'iKy', inv=False).diag()
        self.f   = Vector.func([GRD.xr, GRD.yr], PDE.f_func, None,
                               verb, 'f')
    
    def _gen_matrices(self, d, n, h, dim, mode, tau, verb):
        I = Matrix.eye(d, mode, tau)
        
        B = Matrix.volterra(d, mode, tau, h)
        if self.PDE.bc == BC_PR:      
            E = Matrix.ones(d, mode, tau)
            B = B - h*(E.dot(B) + B.dot(E)) + h*(h+1)/2 * E
        self.Bx = I.kron(B) 
        self.By = B.kron(I)  

        self.iqx = self.iKx.diag().sum_out(dim, 0)
        self.iqy = self.iKy.diag().sum_out(dim, 1)
        
        self.qx = self.iqx.inv(v0=None, verb=verb, name = 'qx')
        self.qy = self.iqy.inv(v0=None, verb=verb, name = 'qy')
        
        E = Matrix.ones(d, mode, tau)
        self.Wx = self.qx.diag().kron(E)
        self.Wy = E.kron(self.qy.diag())
        
        I2 = Matrix.eye(d*dim, mode, tau)      
        self.Rx = self.iKx.dot(I2-self.Wx.dot(self.iKx))
        self.Ry = self.iKy.dot(I2-self.Wy.dot(self.iKy))
        self.R0 = -1. * self.Rx.dot(self.Kxy).dot(self.Ry)
        
        self.Hx = self.Bx.dot(self.Rx).dot(self.Bx.T)
        self.Hy = self.By.dot(self.Ry).dot(self.By.T)
        self.H0 = self.Bx.dot(self.R0).dot(self.By.T)
        
    def _gen_system(self, d, n, h, dim, mode, tau, verb):           
        self.A = self.Hx + self.Hy + self.H0
        self.rhs = (self.Hy-self.H0).dot(self.f)
        
    def _gen_solution(self, PDE, LSS, d, n, h, dim, mode, eps, tau, verb):
        sol = LSS.solve(self.A.T.dot(self.A), self.A.T.dot(self.rhs), eps, tau, PDE.sol0, verb)        
        self.wx = self.Bx.T.dot(sol)
        self.wy = self.f - self.By.T.dot(sol)
        self.ux = self.Rx.dot(self.wx) + self.R0.dot(self.wy)
        self.uy = self.Ry.dot(self.wy)
        self.u  = -1.*self.Bx.dot(self.ux)     