# -*- coding: utf-8 -*-
ZEROp = 1.E-15
ZEROm =-1.E-15

import unittest
import numpy as np

import tt

from qttpdesolver.utils.general import rel_err
from qttpdesolver.utils.transform2mesh import transform2finer
from qttpdesolver.utils.transform2mesh import transform2coarser

class TestUtilsTransform2mesh(unittest.TestCase):
    ''' Tests for functions from utils.transform2mesh module (prototype).  '''
    
    def setUp(self):
        self.d = 6
        self.n = 2**self.d
        self.tau = 1.E-8

        self.x = np.arange(self.n)+1.
        self.x_tt = tt.tensor(self.x.reshape([2]*self.d, order='F'), self.tau)
        
    def tearDown(self):
        pass
    
class TestUtilsTstransform2mesh_transform2finer(TestUtilsTransform2mesh):
    ''' 
    Tests for function transform2finer
    from module utils.tstransform2mesh.
    '''
    
    def test_np_1d_incorrect_size(self):
        ''' The size of vector should be 2^{d}.  '''
        raised = False
        try:
            transform2finer(self.x[:-1], 1, self.tau, 1)
        except:
            raised = True
        self.assertTrue(raised)
        
    def test_np_2d_incorrect_size(self):
        ''' The size of vector should be 2^{2*d}.  '''
        raised = False
        try:
            transform2finer(self.x[:-1], 2, self.tau, 1)
        except:
            raised = True
        self.assertTrue(raised)
        raised = False
        try:
            transform2finer(self.x[:2], 2, self.tau, 1)
        except:
            raised = True
        self.assertTrue(raised)
        
    def test_np_1d(self):
        ''' In 1D result should be correct  '''
        res = transform2finer(self.x, 1, self.tau, 1)
        self.assertTrue(abs(res[0]-self.x[0]/2) < ZEROp)
        self.assertTrue(abs(res[1]-self.x[0]) < ZEROp)
        self.assertTrue(abs(res[2]-(self.x[0]+self.x[1])/2) < ZEROp)
        self.assertTrue(abs(res[-1]-self.x[-1]) < ZEROp)
        
    def test_np_2d(self):
        ''' In 2D result should be correct  '''
        res = transform2finer(self.x, 2, reps=1)
        self.assertTrue(abs(res[0]-self.x[0]/4) < ZEROp)
        self.assertTrue(abs(res[2**(self.d/2)*2]-self.x[0]/2) < ZEROp)
        self.assertTrue(abs(res[2**(self.d/2)*2+1]-self.x[0]) < ZEROp)
        self.assertTrue(abs(res[-1]-self.x[-1]) < ZEROp)
        
    def test_np_1d_mult_vs_reps(self):
        ''' In 1D case 2 cals of function should be equal to reps=2  '''
        res1 = transform2finer(self.x, 1, self.tau, reps=1)
        res1 = transform2finer(res1, 1, self.tau, reps=1)
        res2 = transform2finer(self.x, 1, reps=2)
        eps = rel_err(res1, res2)
        self.assertTrue(eps < ZEROp) 
        
    def test_np_2d_mult_vs_reps(self):
        ''' In 2D case 2 cals of function should be equal to reps=2  '''
        res1 = transform2finer(self.x, 2, self.tau, reps=1)
        res1 = transform2finer(res1, 2, self.tau, reps=1)
        res2 = transform2finer(self.x, 2, reps=2)
        eps = rel_err(res1, res2)
        self.assertTrue(eps < ZEROp) 
        
    def test_np_vs_tt_1d(self):
        ''' In 1D case results for MODE_NP and MODE_TT should be equal.  '''
        res1 = transform2finer(self.x, 1, self.tau, reps=1)
        res2 = transform2finer(self.x_tt, 1, self.tau).full().flatten('F')
        eps = rel_err(res1, res2)
        self.assertTrue(eps < self.tau) 
        
    def test_np_vs_tt_1d_reps(self):
        ''' In 1D case results for MODE_NP and MODE_TT should be equal.  '''
        res1 = transform2finer(self.x, 1, self.tau, reps=3)
        res2 = transform2finer(self.x_tt, 1, self.tau, 3).full().flatten('F')
        eps = rel_err(res1, res2)
        self.assertTrue(eps < self.tau)
        
    def test_np_vs_tt_2d(self):
        ''' In 2D case results for MODE_NP and MODE_TT should be equal.  '''
        res1 = transform2finer(self.x, 2, self.tau, reps=1)
        res2 = transform2finer(self.x_tt, 2, self.tau).full().flatten('F')
        eps = rel_err(res1, res2)
        self.assertTrue(eps < self.tau) 

    def test_np_vs_tt_2d_reps(self):
        ''' In 2D case results for MODE_NP and MODE_TT should be equal.  '''
        res1 = transform2finer(self.x, 2, self.tau, reps=2)
        res2 = transform2finer(self.x_tt, 2, self.tau, 2).full().flatten('F')
        eps = rel_err(res1, res2)
        self.assertTrue(eps < self.tau) 
     
class TestUtilsTstransform2mesh_transform2coarser(TestUtilsTransform2mesh):
    ''' 
    Tests for function transform2coarser
    from module utils.tstransform2mesh.
    '''
    
    def test_np_1d(self):
        ''' In 1D result should be correct  '''
        res = transform2coarser(self.x, 1, self.tau, 1)
        self.assertTrue(abs(res[0]-self.x[1]) < ZEROp)
        self.assertTrue(abs(res[1]-self.x[3]) < ZEROp)
        self.assertTrue(abs(res[-1]-self.x[-1]) < ZEROp)
        
    def test_np_2d(self):
        ''' In 2D result should be correct  '''
        res = transform2coarser(self.x, 2, reps=1)
        self.assertTrue(abs(res[0]-self.x[2**(self.d/2)+1]) < ZEROp)
        self.assertTrue(abs(res[-1]-self.x[-1]) < ZEROp)
        
    def test_np_1d_mult_vs_reps(self):
        ''' In 1D case 2 cals of function should be equal to reps=2  '''
        res1 = transform2coarser(self.x, 1, self.tau, reps=1)
        res1 = transform2coarser(res1, 1, self.tau, reps=1)
        res2 = transform2coarser(self.x, 1, reps=2)
        eps = rel_err(res1, res2)
        self.assertTrue(eps < ZEROp) 
        
    def test_np_2d_mult_vs_reps(self):
        ''' In 2D case 2 cals of function should be equal to reps=2  '''
        res1 = transform2coarser(self.x, 2, self.tau, reps=1)
        res1 = transform2coarser(res1, 2, self.tau, reps=1)
        res2 = transform2coarser(self.x, 2, reps=2)
        eps = rel_err(res1, res2)
        self.assertTrue(eps < ZEROp) 
        
    def test_np_vs_tt_1d(self):
        ''' In 1D case results for MODE_NP and MODE_TT should be equal.  '''
        res1 = transform2coarser(self.x, 1, self.tau, reps=1)
        res2 = transform2coarser(self.x_tt, 1, self.tau).full().flatten('F')
        eps = rel_err(res1, res2)
        self.assertTrue(eps < self.tau) 
        
    def test_np_vs_tt_1d_reps(self):
        ''' In 1D case results for MODE_NP and MODE_TT should be equal.  '''
        res1 = transform2coarser(self.x, 1, self.tau, reps=3)
        res2 = transform2coarser(self.x_tt, 1, self.tau, 3).full().flatten('F')
        eps = rel_err(res1, res2)
        self.assertTrue(eps < self.tau)
        
    def test_np_vs_tt_2d(self):
        ''' In 2D case results for MODE_NP and MODE_TT should be equal.  '''
        res1 = transform2coarser(self.x, 2, self.tau, reps=1)
        res2 = transform2coarser(self.x_tt, 2, self.tau).full().flatten('F')
        eps = rel_err(res1, res2)
        self.assertTrue(eps < self.tau) 

    def test_np_vs_tt_2d_reps(self):
        ''' In 2D case results for MODE_NP and MODE_TT should be equal.  '''
        res1 = transform2coarser(self.x, 2, self.tau, reps=2)
        res2 = transform2coarser(self.x_tt, 2, self.tau, 2).full().flatten('F')
        eps = rel_err(res1, res2)
        self.assertTrue(eps < self.tau) 
        
if __name__ == '__main__':
    unittest.main()