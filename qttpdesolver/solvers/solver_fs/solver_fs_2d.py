# -*- coding: utf-8 -*-
from ..solver import BC_HD, BC_PR, Solver
from ...tensor_wrapper import Vector, Matrix

class SolverFS_2d(Solver):
    '''
    2D Finite Sum (FS) solver for elliptic PDEs of the form
    -div(k grad u) = f. See parent class Solver for more details.
    '''

    def _gen_coefficients(self, PDE, GRD, d, n, h, dim, mode, tau, verb):
        self.iKx = PDE.Kx.build([GRD.xc, GRD.yr], verb=verb, inv=True, to_diag=True)
        self.iKy = PDE.Ky.build([GRD.xr, GRD.yc], verb=verb, inv=True, to_diag=True)

        self.f = PDE.F.build([GRD.xr, GRD.yr], verb=verb)

        if 1 < 0:
            return
            e = Vector.ones(2*d, mode=mode, tau=tau)
            I = Matrix.eye(d, mode=mode, tau=tau)
            Z = Matrix.unit(d, mode=mode, tau=tau, i=-1, j=-1)
            J = I - Z
            P = Matrix.unit(d, mode=mode, tau=tau, i=0, j=0)
            #self.f+= I.kron(P).dot(e) * 1.E2 + P.kron(I).dot(e) * 1.E2
            #self.f+= I.kron(Z).dot(e) * 1.E2 + Z.kron(I).dot(e) * 1.E2
            print 'test!!!'

    def _gen_matrices(self, d, n, h, dim, mode, tau, verb):
        isperiodic = self.PDE.bc == BC_PR

        I = Matrix.eye(d, mode, tau)

        B = Matrix.volterra(d, mode, tau, h, isperiodic=isperiodic)
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

        if 1 < 0:
            print 'test3'
            I = Matrix.eye(d, mode=mode, tau=tau)
            Z = Matrix.unit(d, mode=mode, tau=tau, i=-1, j=-1)
            J = I - Z
            self.f = self.Hy.dot(Z.kron(I)).dot(self.f)

    def _gen_system(self, d, n, h, dim, mode, tau, verb):
        self.A = self.Hx + self.Hy
        self.rhs = self.Hy.dot(self.f)

    def _gen_solution(self, PDE, LSS, d, n, h, dim, mode, eps, tau, verb):
        sol = LSS.solve(self.A, self.rhs, eps, tau, PDE.sol0, verb)
        self.wx = sol
        self.ux = self.Rx.dot(self.wx)
        self.uy = self.Ry.dot(self.f - self.wx)
        self.u  = self.Hx.dot(self.wx)
