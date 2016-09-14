# -*- coding: utf-8 -*-
ZEROp = 1.E-15
ZEROm =-1.E-15

import unittest
import numpy as np

import tt

from qttpdesolver.utils.general import rel_err
from qttpdesolver.utils.splitting import half_array, eo_sub_array, part_array

class TestUtilsSplitting(unittest.TestCase):
    ''' Tests for functions from utils.splitting module (prototype).  '''
    
    def setUp(self):
        self.d = 6
        self.n = 2**self.d
        self.tau = 1.E-8

        self.x = np.arange(self.n)+1.
        self.x_tt = tt.tensor(self.x.reshape([2]*self.d, order='F'), self.tau)
        
        self.d2 = 5
        self.n2 = 2**self.d2
        self.x2 = np.arange(self.n2)+1.
        self.x2_tt = tt.tensor(self.x2.reshape([2]*self.d2, order='F'), self.tau)
        
    def tearDown(self):
        pass
    
class TestUtilsSplitting_half_array(TestUtilsSplitting):
    ''' 
    Tests for function half_array
    from module utils.splitting.
    '''
    
    def test_np_index_0(self):
        ''' For index 0 it should be the first half of array.  '''
        res = half_array(self.x, 0)
        eps = rel_err(res, self.x[:self.x.shape[0]/2])
        self.assertTrue(eps < ZEROp) 
        
    def test_np_index_1(self):
        ''' For index 1 it should be the second half of array.  '''
        res = half_array(self.x, 1)
        eps = rel_err(res, self.x[self.x.shape[0]/2:])
        self.assertTrue(eps < ZEROp) 
        
    def test_np_incorrect_index(self):
        ''' If index != 0, 1, then in MODE_NP the error should be raised.  '''
        raised = False
        try:
            half_array(self.x, 2)
        except:
            raised = True
        self.assertTrue(raised)
        
    def test_np_vs_tt_index_0(self):
        ''' In the both modes the results should be the same for index=0.  '''
        res1 =  half_array(self.x, 0)
        res2 =  half_array(self.x_tt, 0, self.tau).full().flatten('F')
        eps = rel_err(res1, res2)
        self.assertTrue(eps < self.tau)
        
    def test_np_vs_tt_index_1(self):
        ''' In the both modes the results should be the same for index=1.  '''
        res1 =  half_array(self.x, 1)
        res2 =  half_array(self.x_tt, 1, self.tau).full().flatten('F')
        eps = rel_err(res1, res2)
        self.assertTrue(eps < self.tau)
        
    def test_np_original_is_unchanged(self):
        ''' The function shouldn't change original vector.  '''
        x = self.x.copy()
        half_array(self.x, 1)
        eps = rel_err(x, self.x)
        self.assertTrue(eps < ZEROp)
        
    def test_tt_original_is_unchanged(self):
        ''' The function shouldn't change original tensor.  '''
        x_tt = self.x_tt.copy()
        half_array(self.x, 1, self.tau)
        eps = rel_err(x_tt, self.x_tt)
        self.assertTrue(eps < ZEROp)
        
class TestUtilsSplitting_eo_sub_array(TestUtilsSplitting):
    ''' 
    Tests for function eo_sub_array
    from module utils.splitting.
    '''
    
    def test_np_index_0(self):
        ''' For index 0 it should be even elements of vector.  '''
        res = eo_sub_array(self.x, 0)
        eps = rel_err(res, self.x[0::2])
        self.assertTrue(eps < ZEROp) 
        
    def test_np_index_1(self):
        ''' For index 1 it should be odd elements of vector.  '''
        res = eo_sub_array(self.x, 1)
        eps = rel_err(res, self.x[1::2])
        self.assertTrue(eps < ZEROp) 
        
    def test_np_incorrect_index(self):
        ''' If index != 0, 1, then in MODE_NP the error should be raised.  '''
        raised = False
        try:
            eo_sub_array(self.x, 2)
        except:
            raised = True
        self.assertTrue(raised)
        
    def test_np_vs_tt_index_0(self):
        ''' In the both modes the results should be the same for index=0.  '''
        res1 =  eo_sub_array(self.x, 0)
        res2 =  eo_sub_array(self.x_tt, 0, self.tau).full().flatten('F')
        eps = rel_err(res1, res2)
        self.assertTrue(eps < self.tau)
        
    def test_np_vs_tt_index_1(self):
        ''' In the both modes the results should be the same for index=1.  '''
        res1 =  eo_sub_array(self.x, 1)
        res2 =  eo_sub_array(self.x_tt, 1, self.tau).full().flatten('F')
        eps = rel_err(res1, res2)
        self.assertTrue(eps < self.tau)
        
    def test_np_original_is_unchanged(self):
        ''' The function shouldn't change original vector.  '''
        x = self.x.copy()
        eo_sub_array(self.x, 1)
        eps = rel_err(x, self.x)
        self.assertTrue(eps < ZEROp)
        
    def test_tt_original_is_unchanged(self):
        ''' The function shouldn-t change original tensor.  '''
        x_tt = self.x_tt.copy()
        eo_sub_array(self.x, 1, self.tau)
        eps = rel_err(x_tt, self.x_tt)
        self.assertTrue(eps < ZEROp)
        
class TestUtilsSplitting_part_array(TestUtilsSplitting):
    ''' 
    Tests for function part_array
    from module utils.splitting.
    '''
    
    def test_np(self):
        ''' We construct array from parts, and then check, that them are
            obtained correctly by function.  '''
        d = 3; x_parts = []
        for _ in range(2**d):
            x_parts.append(np.random.randn(2**d))
        x = np.hstack(x_parts)
        for i in range(2**d):
            res = part_array(x, i)
            eps = rel_err(res, x_parts[i])
            self.assertTrue(eps < ZEROp) 

    def test_np_vs_tt(self):
        ''' In the both modes the results should be the same.  '''
        for i in range(2**(self.d/2)):
            res1 = part_array(self.x, i)
            res2 = part_array(self.x_tt, i, self.tau).full().flatten('F')
            eps = rel_err(res1, res2)
            self.assertTrue(eps < self.tau) 
      
if __name__ == '__main__':
    unittest.main()