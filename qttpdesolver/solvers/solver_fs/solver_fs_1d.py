# -*- coding: utf-8 -*-
from ..solver import BC_PR, Solver
from ...tensor_wrapper import Vector, Matrix

class SolverFS_1d(Solver):
    '''
    1D Finite Sum (FS) solver for elliptic PDEs of the form
    -div(k grad u) = f. See parent class Solver for more details.
    '''

    def _gen_coefficients(self, PDE, GRD, d, n, h, dim, mode, tau, verb):
        self.iKx = PDE.Kx.build([GRD.xc], verb=verb, inv=True, to_diag=True)
        self.f = PDE.F.build([GRD.xr], verb=verb)

    def _gen_matrices(self, d, n, h, dim, mode, tau, verb):
        isperiodic = self.PDE.bc == BC_PR
        self.Bx = Matrix.volterra(d, mode, tau, h, isperiodic=isperiodic)

    def _gen_system(self, d, n, h, dim, mode, tau, verb):
        pass

    def _gen_solution(self, PDE, LSS, d, n, h, dim, mode, eps, tau, verb):
        ikx, f, Bx = self.iKx.diag(), self.f, self.Bx
        g = Bx.T.dot(f) * ikx
        s = g.sum()/ikx.sum()
        self.ux = g - s*ikx
        self.u  = Bx.dot(self.ux)
