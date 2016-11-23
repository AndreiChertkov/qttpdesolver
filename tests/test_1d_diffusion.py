# -*- coding: utf-8 -*-
import unittest
import numpy as np

from qttpdesolver import Pde, auto_solve
from qttpdesolver import MODE_NP, MODE_TT, MODE_SP, SOLVER_FS, SOLVER_FD

class TestDiffusion1DBase(unittest.TestCase):
    ''' Base class for 1D diffusion problem  '''
    
    def setUp(self):          
        self.PDE = Pde()
        self.PDE.set_model('Simple. Analyt 1D diffusion PDE')
        self.PDE.set_params([np.pi*2])
        self.PDE.set_verb(False, False, False)       
        self.PDE.set_tau(tau=1.E-14, eps_lss=1.E-14, tau_lss=1.E-14)
        self.PDE.set_lss_params(nswp=20, kickrank=4, local_prec='n', local_iters=2,
                                local_restart=20, trunc_norm=1, max_full_size=100)
        self.PDE.update_d(8)
    
class TestDiffusion1D_fs(TestDiffusion1DBase):
    ''' Test of PDE solution result for Solver-FS.  '''
    
    def setUp(self):
        TestDiffusion1DBase.setUp(self)
        self.PDE.set_solver_txt(SOLVER_FS)
   
    def test_np(self):
        self.PDE.set_mode(MODE_NP)
        auto_solve(self.PDE, present_res_1s=False)
        
        self.assertTrue(self.PDE.u_err  < 5.16E-05)
        self.assertTrue(self.PDE.ux_err < 2.66E-05)

    def test_tt(self):
        self.PDE.set_mode(MODE_TT)
        auto_solve(self.PDE, present_res_1s=False)
        
        self.assertTrue(self.PDE.u_err  < 5.16E-05)
        self.assertTrue(self.PDE.ux_err < 2.66E-05)

    def test_sp(self):
        ''' For Solver FS MODE_SP should be not available.  '''
        self.PDE.set_mode(MODE_SP)
        raised = False
        try:
            auto_solve(self.PDE, present_res_1s=False)
        except:
            raised = True
        self.assertTrue(raised)
        
class TestDiffusion1D_fd(TestDiffusion1DBase):
    ''' Test of PDE solution result for Solver-FD.  '''
    
    def setUp(self):
        TestDiffusion1DBase.setUp(self)
        self.PDE.set_solver_txt(SOLVER_FD)
        
    def test_np(self):
        self.PDE.set_mode(MODE_NP)
        auto_solve(self.PDE, present_res_1s=False)
        
        self.assertTrue(self.PDE.u_err  < 5.16E-05)
        self.assertTrue(self.PDE.ux_err < 2.66E-05)

    def test_tt(self):
        self.PDE.set_mode(MODE_TT)
        auto_solve(self.PDE, present_res_1s=False)
        
        self.assertTrue(self.PDE.u_err  < 5.16E-05)
        self.assertTrue(self.PDE.ux_err < 2.66E-05)
        
    def test_sp(self):
        self.PDE.set_mode(MODE_SP)
        auto_solve(self.PDE, present_res_1s=False)
        
        self.assertTrue(self.PDE.u_err  < 5.16E-05)
        self.assertTrue(self.PDE.ux_err < 2.66E-05)
        
if __name__ == '__main__':
    unittest.main()