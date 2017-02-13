# -*- coding: utf-8 -*-
from ..solver import Solver
from ...tensor_wrapper import Vector, Matrix

class SolverFD_3d(Solver):
    '''
    3D Finite difference (FD) solver for elliptic PDEs of the form 
    -div(k grad u) = f. See parent class Solver for more details.
    '''
     
    def _gen_coefficients(self, PDE, GRD, d, n, h, dim, mode, tau, verb):
        self.Kx = Vector.func([GRD.xc, GRD.yr, GRD.zr], PDE.k_func, None, 
                              verb, 'Kx').diag()
        self.Ky = Vector.func([GRD.xr, GRD.yc, GRD.zr], PDE.k_func, None, 
                              verb, 'Ky').diag()
        self.Kz = Vector.func([GRD.xr, GRD.yr, GRD.zc], PDE.k_func, None, 
                              verb, 'Kz').diag()
        self.f  = Vector.func([GRD.xr, GRD.yr, GRD.zr], PDE.f_func, None,
                              verb, 'f')
            
    def _gen_matrices(self, d, n, h, dim, mode, tau, verb):
        I = Matrix.eye(d, mode, tau)
        
        iB = Matrix.findif(d, mode, tau, h)        
        self.iBx = I.kron(I).kron(iB) 
        self.iBy = I.kron(iB).kron(I)   
        self.iBz = iB.kron(I).kron(I)  

        Z = Matrix.unit(d, mode, tau, -1, -1)
        self.Zx = I.kron(I).kron(Z) 
        self.Zy = I.kron(Z).kron(I)   
        self.Zz = Z.kron(I).kron(I)
        
        I3 = Matrix.eye(3*d, mode, tau)
        self.Sx = I3 - self.Zx 
        self.Sy = I3 - self.Zy 
        self.Sz = I3 - self.Zz         

    def _gen_system(self, d, n, h, dim, mode, tau, verb):
        self.A = self.Sx.dot(self.iBx.T).dot(self.Kx).dot(self.iBx).dot(self.Sx)
        self.A+= self.Sy.dot(self.iBy.T).dot(self.Ky).dot(self.iBy).dot(self.Sy)
        self.A+= self.Sz.dot(self.iBz.T).dot(self.Kz).dot(self.iBz).dot(self.Sz)
        self.A+= self.Zx + self.Zy + self.Zz
        self.rhs = self.Sz.dot(self.Sy.dot(self.Sx.dot(self.f)))
        
    def _gen_solution(self, PDE, LSS, d, n, h, dim, mode, eps, tau, verb):
        self.u = LSS.solve(self.A, self.rhs, eps, tau, PDE.sol0, verb)        
        self.ux = self.iBx.dot(self.u)
        self.uy = self.iBy.dot(self.u)
        self.uz = self.iBz.dot(self.u)