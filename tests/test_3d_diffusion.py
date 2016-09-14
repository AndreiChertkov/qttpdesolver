# -*- coding: utf-8 -*-
import unittest
import numpy as np

from qttpdesolver import Pde, auto_solve, MODE_NP, MODE_TT, MODE_SP

class TestDiffusion3DBase(unittest.TestCase):
    ''' Base class for 3D diffusion problem  '''
    
    def setUp(self):
        self.PDE = Pde()
        self.PDE.set_model('Simple. Analyt 3D diffusion PDE')
        self.PDE.set_params([np.pi, np.pi*2, np.pi*3])
        
        self.PDE.set_with_en(True)  
                
        self.PDE.set_tau(tau_round=1.E-10, tau_cross=1.E-10, tau_amens=1.E-6)
        self.PDE.set_algss_par(nswp=20, kickrank=4, local_prec='n', local_iters=2,
                               local_restart=20, trunc_norm=1, max_full_size=100,
                               tau_u_calc_from_algss=True)
        self.PDE.update_d(3)

class TestDiffusion3D_fs(TestDiffusion3DBase):
    ''' Test of PDE solution result for Solver-FS.  '''
    
    def setUp(self):
        TestDiffusion3DBase.setUp(self)
        self.PDE.set_solver_txt('fs')
   
    def test_np(self):
        self.PDE.set_mode(MODE_NP)
        auto_solve(self.PDE, present_res_1s=False)

        self.assertTrue(self.PDE.u_err     < 1.16e-01)
        self.assertTrue(self.PDE.ud_err[0] < 7.96e-01)
        self.assertTrue(self.PDE.ud_err[1] < 7.06e-01)
        self.assertTrue(self.PDE.ud_err[2] < 6.56e-01)
        self.assertTrue(self.PDE.en_err    < 6.16e-02)
        
    def test_tt(self):
        self.PDE.set_mode(MODE_TT)
        auto_solve(self.PDE, present_res_1s=False)

        self.assertTrue(self.PDE.u_err     < 1.16e-01)
        self.assertTrue(self.PDE.ud_err[0] < 7.96e-01)
        self.assertTrue(self.PDE.ud_err[1] < 7.06e-01)
        self.assertTrue(self.PDE.ud_err[2] < 6.56e-01)
        self.assertTrue(self.PDE.en_err    < 6.16e-02)
        
    def test_sp(self):
        ''' For Solver FS MODE_SP should be not available.  '''
        self.PDE.set_mode(MODE_SP)
        raised = False
        try:
            auto_solve(self.PDE, present_res_1s=False)
        except:
            raised = True
        self.assertTrue(raised)
        
class TestDiffusion3D_fd(TestDiffusion3DBase):
    ''' Test of PDE solution result for Solver-FD.  '''
    
    def setUp(self):
        TestDiffusion3DBase.setUp(self)
        self.PDE.set_solver_txt('fd')
        
    def test_np(self):
        self.PDE.set_mode(MODE_NP)
        auto_solve(self.PDE, present_res_1s=False)
        
        self.assertTrue(self.PDE.u_err     < 1.16e-01)
        self.assertTrue(self.PDE.ud_err[0] < 7.96e-01)
        self.assertTrue(self.PDE.ud_err[1] < 7.06e-01)
        self.assertTrue(self.PDE.ud_err[2] < 6.56e-01)
        self.assertTrue(self.PDE.en_err    < 6.16e-02)

    def test_tt(self):
        self.PDE.set_mode(MODE_TT)
        auto_solve(self.PDE, present_res_1s=False)
        
        self.assertTrue(self.PDE.u_err     < 1.16e-01)
        self.assertTrue(self.PDE.ud_err[0] < 7.96e-01)
        self.assertTrue(self.PDE.ud_err[1] < 7.06e-01)
        self.assertTrue(self.PDE.ud_err[2] < 6.56e-01)
        self.assertTrue(self.PDE.en_err    < 6.16e-02)
        
    def test_sp(self):
        self.PDE.set_mode(MODE_SP)
        auto_solve(self.PDE, present_res_1s=False)
        
        self.assertTrue(self.PDE.u_err     < 1.16e-01)
        self.assertTrue(self.PDE.ud_err[0] < 7.96e-01)
        self.assertTrue(self.PDE.ud_err[1] < 7.06e-01)
        self.assertTrue(self.PDE.ud_err[2] < 6.56e-01)
        self.assertTrue(self.PDE.en_err    < 6.16e-02)
        
if __name__ == '__main__':
    unittest.main()