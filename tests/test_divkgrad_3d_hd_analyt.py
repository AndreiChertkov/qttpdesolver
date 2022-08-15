# -*- coding: utf-8 -*-
import unittest
import numpy as np

from qttpdesolver import Pde, auto_solve
from qttpdesolver import MODE_NP, MODE_TT, MODE_SP, SOLVER_FS, SOLVER_FD

class Test_divkgrad_3d_hd_analyt(unittest.TestCase):
    ''' Base class for 3D diffusion problem  '''
    
    def setUp(self):
        self.PDE = Pde()
        self.PDE.set_model('divkgrad_3d_hd_analyt')
        self.PDE.set_params([np.pi, np.pi*2, np.pi*3])
        self.PDE.set_verb(False, False, False)       
        self.PDE.set_tau(tau=1.E-10, eps_lss=1.E-10, tau_lss=1.E-10)
        self.PDE.set_lss_params(nswp=20, kickrank=4, local_prec='n', local_iters=2,
                                local_restart=20, trunc_norm=1, max_full_size=100)
        self.PDE.update_d(3)

class Test_divkgrad_3d_hd_analyt_fs(Test_divkgrad_3d_hd_analyt):

    def test_np(self):
        self.PDE.set_solver_name(SOLVER_FS)
        self.PDE.set_mode(MODE_NP)
        auto_solve(self.PDE, present_res_1s=False)

        self.assertTrue(self.PDE.u_err  < 1.16e-01)
        self.assertTrue(self.PDE.ux_err < 8.66e-02)
        self.assertTrue(self.PDE.uy_err < 5.36e-02)
        self.assertTrue(self.PDE.uz_err < 4.76e-02)

    def test_tt(self):
        self.PDE.set_solver_name(SOLVER_FS)
        self.PDE.set_mode(MODE_TT)
        auto_solve(self.PDE, present_res_1s=False)

        self.assertTrue(self.PDE.u_err  < 1.16e-01)
        self.assertTrue(self.PDE.ux_err < 8.66e-02)
        self.assertTrue(self.PDE.uy_err < 5.36e-02)
        self.assertTrue(self.PDE.uz_err < 4.76e-02)
        
    def test_sp(self):
        self.PDE.set_solver_name(SOLVER_FS)
        self.PDE.set_mode(MODE_SP)
        raised = False
        try:
            auto_solve(self.PDE, present_res_1s=False)
        except:
            raised = True
        self.assertTrue(raised)
        
class Test_divkgrad_3d_hd_analyt_fd(Test_divkgrad_3d_hd_analyt):

    def test_np(self):
        self.PDE.set_solver_name(SOLVER_FD)
        self.PDE.set_mode(MODE_NP)
        auto_solve(self.PDE, present_res_1s=False)
        
        self.assertTrue(self.PDE.u_err  < 1.16e-01)
        self.assertTrue(self.PDE.ux_err < 8.66e-02)
        self.assertTrue(self.PDE.uy_err < 5.36e-02)
        self.assertTrue(self.PDE.uz_err < 4.76e-02)

    def test_tt(self):
        self.PDE.set_solver_name(SOLVER_FD)
        self.PDE.set_mode(MODE_TT)
        auto_solve(self.PDE, present_res_1s=False)
        
        self.assertTrue(self.PDE.u_err  < 1.16e-01)
        self.assertTrue(self.PDE.ux_err < 8.66e-02)
        self.assertTrue(self.PDE.uy_err < 5.36e-02)
        self.assertTrue(self.PDE.uz_err < 4.76e-02)
        
    def test_sp(self):
        self.PDE.set_solver_name(SOLVER_FD)
        self.PDE.set_mode(MODE_SP)
        auto_solve(self.PDE, present_res_1s=False)
        
        self.assertTrue(self.PDE.u_err  < 1.16e-01)
        self.assertTrue(self.PDE.ux_err < 8.66e-02)
        self.assertTrue(self.PDE.uy_err < 5.36e-02)
        self.assertTrue(self.PDE.uz_err < 4.76e-02)
        
if __name__ == '__main__':
    unittest.main()