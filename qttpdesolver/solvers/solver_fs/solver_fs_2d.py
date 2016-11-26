# -*- coding: utf-8 -*-
from ..solver import Solver
from ...tensor_wrapper.vector import Vector
from ...tensor_wrapper.matrix import Matrix

class SolverFS_2d(Solver):
    '''
    2D Finite Sum (FS) solver for elliptic PDEs of the form -div(k grad u) = f.
    See parent class Solver for more details.
    '''
    
    def _gen_coefficients(self, PDE, GRD, d, n, h, dim, mode, tau, verb):
        self.iKx = Vector.func([GRD.xc, GRD.yr], PDE.k_func, None, 
                               verb, 'iKx', inv=True).diag()
        self.iKy = Vector.func([GRD.xr, GRD.yc], PDE.k_func, None, 
                               verb, 'iKy', inv=True).diag()
        self.f   = Vector.func([GRD.xr, GRD.yr], PDE.f_func, None,
                               verb, 'f')
    
    def _gen_matrices(self, d, n, h, dim, mode, tau, verb):
        I = Matrix.eye(d, mode, tau)
        B = Matrix.volterra(d, mode, tau, h)
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
        self.Rx = self.iKx.dot(I2-self.Wx.dot(self.iKx)).dot(self.Bx.T)
        self.Ry = self.iKy.dot(I2-self.Wy.dot(self.iKy)).dot(self.By.T)
        
        self.Hx = self.Bx.dot(self.Rx)
        self.Hy = self.By.dot(self.Ry)
           
    def _gen_system(self, d, n, h, dim, mode, tau, verb):           
        self.A = self.Hx + self.Hy
        self.rhs = self.Hy.dot(self.f)
        
    def _gen_solution(self, PDE, LSS, d, n, h, dim, mode, eps, tau, verb):
        sol = LSS.solve(self.A, self.rhs, eps, tau, PDE.sol0, verb)        
        self.wx = sol
        self.ux = self.Rx.dot(self.wx)
        self.uy = self.Ry.dot(self.f - self.wx)
        self.u  = self.Hx.dot(self.wx)     