# -*- coding: utf-8 -*-
ZEROp = 1.E-15
ZEROm =-1.E-15

import unittest
import numpy as np

import tt

from qttpdesolver.utils.general import MODE_NP, MODE_TT, rel_err
from qttpdesolver.utils.grid import _mesh_cc, _mesh_lc, _mesh_rc
from qttpdesolver.utils.grid import _mesh_uxe, _mesh_uye, _mesh_uze
from qttpdesolver.utils.grid import _construct_mesh, quan_on_grid
from qttpdesolver.utils.grid import coord2ind, delta_on_grid, deltas_on_grid

class TestUtilsGrid(unittest.TestCase):
    ''' Tests for functions from utils.grid module (prototype).  '''
    
    def setUp(self):
        self.d = 3
        self.n = 2**self.d
        self.h = 1./self.n
        
        self.tau = 1.E-8
        self.eps = 1.E-8
        
        self.x = np.arange(self.n)+2.; self.x[2] = -4.
        self.y =-2.*self.x
        self.z = self.x*self.y
        
        self.x_tt = tt.tensor(self.x.reshape([2]*self.d, order='F'), self.tau)
        self.y_tt = tt.tensor(self.y.reshape([2]*self.d, order='F'), self.tau)
        self.z_tt = tt.tensor(self.z.reshape([2]*self.d, order='F'), self.tau)
        
        self.A_tt = tt.Toeplitz(self.x_tt, self.d, kind='U')
        self.B_tt = tt.Toeplitz(self.y_tt, self.d, kind='L')
        self.C_tt = tt.Toeplitz(self.z_tt, self.d, kind='L')
        
        self.A = self.A_tt.full()
        self.B = self.B_tt.full()
        self.C = self.C_tt.full()
        
    def tearDown(self):
        pass
    
class TestUtilsGrid__mesh_cc(TestUtilsGrid):
    ''' 
    Tests for function _mesh_cc
    from module utils.grid.
    '''
        
    def test_np_is_correct_1d(self):
        ''' Check for 1D.  '''
        x = _mesh_cc(self.d, 1, self.tau, mode=MODE_NP)[0]
        self.assertTrue(abs(x[0]-self.h/2) < ZEROp)
        self.assertTrue(abs(x[1]-3.*self.h/2) < ZEROp)
        self.assertTrue(abs(x[-1]-(1.-self.h/2)) < ZEROp)
        self.assertTrue(abs(x[self.n/2+1]-(x[self.n/2]+self.h)) < ZEROp)

    def test_np_vs_tt_1d(self):
        ''' In 1D case results for MODE_NP and MODE_TT should be equal.  '''
        res1 = _mesh_cc(self.d, 1, self.tau, mode=MODE_NP)[0]
        res2 = _mesh_cc(self.d, 1, self.tau, mode=MODE_TT)[0].full().flatten('F')
        eps = rel_err(res1, res2)
        self.assertTrue(eps < self.tau) 

    def test_np_is_correct_2d(self):
        ''' Check for 2D.  '''
        x, y = _mesh_cc(self.d, 2, self.tau, mode=MODE_NP)
        self.assertTrue(abs(x[0, 0]-self.h/2) < ZEROp)
        self.assertTrue(abs(x[1, 0]-3.*self.h/2) < ZEROp)
        self.assertTrue(abs(x[0, 1]-self.h/2) < ZEROp)
        self.assertTrue(abs(x[1, 1]-3.*self.h/2) < ZEROp)
        self.assertTrue(abs(x[-1, -1]-(1.-self.h/2)) < ZEROp)

        self.assertTrue(abs(y[0, 0]-self.h/2) < ZEROp)
        self.assertTrue(abs(y[1, 0]-self.h/2) < ZEROp)
        self.assertTrue(abs(y[0, 1]-3.*self.h/2) < ZEROp)
        self.assertTrue(abs(y[1, 1]-3.*self.h/2) < ZEROp)
        self.assertTrue(abs(y[-1, -1]-(1.-self.h/2)) < ZEROp)
        
    def test_np_vs_tt_2d(self):
        ''' In 2D case results for MODE_NP and MODE_TT should be equal.  '''
        res1 = _mesh_cc(self.d, 2, self.tau, mode=MODE_NP)
        res2 = _mesh_cc(self.d, 2, self.tau, mode=MODE_TT)
        eps = rel_err(res1[0].flatten('F'), res2[0].full().flatten('F'))
        self.assertTrue(eps < self.tau) 
        eps = rel_err(res1[1].flatten('F'), res2[1].full().flatten('F'))
        self.assertTrue(eps < self.tau) 

    def test_np_is_correct_3d(self):
        ''' Check for 3D.  '''
        x, y, z = _mesh_cc(self.d, 3, self.tau, mode=MODE_NP)
        self.assertTrue(abs(x[0, 0, 0]-self.h/2) < ZEROp)
        self.assertTrue(abs(x[1, 0, 0]-3.*self.h/2) < ZEROp)
        self.assertTrue(abs(x[0, 1, 0]-self.h/2) < ZEROp)
        self.assertTrue(abs(x[1, 1, 0]-3.*self.h/2) < ZEROp)
        self.assertTrue(abs(x[1, 1, 2]-3.*self.h/2) < ZEROp)
        self.assertTrue(abs(x[-1, -1, -1]-(1.-self.h/2)) < ZEROp)

        self.assertTrue(abs(y[0, 0, 0]-self.h/2) < ZEROp)
        self.assertTrue(abs(y[1, 0, 0]-self.h/2) < ZEROp)
        self.assertTrue(abs(y[0, 1, 0]-3.*self.h/2) < ZEROp)
        self.assertTrue(abs(y[1, 1, 0]-3.*self.h/2) < ZEROp)
        self.assertTrue(abs(y[1, 1, 2]-3.*self.h/2) < ZEROp)
        self.assertTrue(abs(y[-1, -1, -1]-(1.-self.h/2)) < ZEROp)
        
        self.assertTrue(abs(z[0, 0, 0]-self.h/2) < ZEROp)
        self.assertTrue(abs(z[2, 2, 0]-self.h/2) < ZEROp)
        self.assertTrue(abs(z[0, 1, 1]-3.*self.h/2) < ZEROp)
        self.assertTrue(abs(z[0, 1, 2]-5.*self.h/2) < ZEROp)
        self.assertTrue(abs(z[-1, -1, -1]-(1.-self.h/2)) < ZEROp)
        
    def test_np_vs_tt_3d(self):
        ''' In 3D case results for MODE_NP and MODE_TT should be equal.  '''
        res1 = _mesh_cc(self.d, 3, self.tau, mode=MODE_NP)
        res2 = _mesh_cc(self.d, 3, self.tau, mode=MODE_TT)
        eps = rel_err(res1[0].flatten('F'), res2[0].full().flatten('F'))
        self.assertTrue(eps < self.tau) 
        eps = rel_err(res1[1].flatten('F'), res2[1].full().flatten('F'))
        self.assertTrue(eps < self.tau) 
        eps = rel_err(res1[2].flatten('F'), res2[2].full().flatten('F'))
        self.assertTrue(eps < self.tau) 

class TestUtilsGrid__mesh_lc(TestUtilsGrid):
    ''' 
    Tests for function _mesh_lc
    from module utils.grid.
    '''
        
    def test_np_is_correct_1d(self):
        ''' Check for 1D.  '''
        x = _mesh_lc(self.d, 1, self.tau, mode=MODE_NP)[0]
        self.assertTrue(abs(x[0]-0.) < ZEROp)
        self.assertTrue(abs(x[1]-self.h) < ZEROp)
        self.assertTrue(abs(x[-1]-(1.-self.h)) < ZEROp)
        self.assertTrue(abs(x[self.n/2+1]-(x[self.n/2]+self.h)) < ZEROp)

    def test_np_vs_tt_1d(self):
        ''' In 1D case results for MODE_NP and MODE_TT should be equal.  '''
        res1 = _mesh_lc(self.d, 1, self.tau, mode=MODE_NP)[0]
        res2 = _mesh_lc(self.d, 1, self.tau, mode=MODE_TT)[0].full().flatten('F')
        eps = rel_err(res1, res2)
        self.assertTrue(eps < self.tau) 

    def test_np_is_correct_2d(self):
        ''' Check for 2D.  '''
        x, y = _mesh_lc(self.d, 2, self.tau, mode=MODE_NP)
        self.assertTrue(abs(x[0, 0]-0.) < ZEROp)
        self.assertTrue(abs(x[1, 0]-self.h) < ZEROp)
        self.assertTrue(abs(x[0, 1]-0.) < ZEROp)
        self.assertTrue(abs(x[1, 1]-self.h) < ZEROp)
        self.assertTrue(abs(x[-1, -1]-(1.-self.h)) < ZEROp)

        self.assertTrue(abs(y[0, 0]-0.) < ZEROp)
        self.assertTrue(abs(y[1, 0]-0.) < ZEROp)
        self.assertTrue(abs(y[0, 1]-self.h) < ZEROp)
        self.assertTrue(abs(y[1, 1]-self.h) < ZEROp)
        self.assertTrue(abs(y[-1, -1]-(1.-self.h)) < ZEROp)
        
    def test_np_vs_tt_2d(self):
        ''' In 2D case results for MODE_NP and MODE_TT should be equal.  '''
        res1 = _mesh_lc(self.d, 2, self.tau, mode=MODE_NP)
        res2 = _mesh_lc(self.d, 2, self.tau, mode=MODE_TT)
        eps = rel_err(res1[0].flatten('F'), res2[0].full().flatten('F'))
        self.assertTrue(eps < self.tau) 
        eps = rel_err(res1[1].flatten('F'), res2[1].full().flatten('F'))
        self.assertTrue(eps < self.tau) 

    def test_np_is_correct_3d(self):
        ''' Check for 3D.  '''
        x, y, z = _mesh_lc(self.d, 3, self.tau, mode=MODE_NP)
        self.assertTrue(abs(x[0, 0, 0]-0.) < ZEROp)
        self.assertTrue(abs(x[1, 0, 0]-self.h) < ZEROp)
        self.assertTrue(abs(x[0, 1, 0]-0.) < ZEROp)
        self.assertTrue(abs(x[1, 1, 0]-self.h) < ZEROp)
        self.assertTrue(abs(x[1, 1, 2]-self.h) < ZEROp)
        self.assertTrue(abs(x[-1, -1, -1]-(1.-self.h)) < ZEROp)

        self.assertTrue(abs(y[0, 0, 0]-0.) < ZEROp)
        self.assertTrue(abs(y[1, 0, 0]-0.) < ZEROp)
        self.assertTrue(abs(y[0, 1, 0]-self.h) < ZEROp)
        self.assertTrue(abs(y[1, 1, 0]-self.h) < ZEROp)
        self.assertTrue(abs(y[1, 1, 2]-self.h) < ZEROp)
        self.assertTrue(abs(y[-1, -1, -1]-(1.-self.h)) < ZEROp)
        
        self.assertTrue(abs(z[0, 0, 0]-0.) < ZEROp)
        self.assertTrue(abs(z[2, 2, 0]-0.) < ZEROp)
        self.assertTrue(abs(z[0, 1, 1]-self.h) < ZEROp)
        self.assertTrue(abs(z[0, 1, 2]-2.*self.h) < ZEROp)
        self.assertTrue(abs(z[-1, -1, -1]-(1.-self.h)) < ZEROp)
        
    def test_np_vs_tt_3d(self):
        ''' In 3D case results for MODE_NP and MODE_TT should be equal.  '''
        res1 = _mesh_lc(self.d, 3, self.tau, mode=MODE_NP)
        res2 = _mesh_lc(self.d, 3, self.tau, mode=MODE_TT)
        eps = rel_err(res1[0].flatten('F'), res2[0].full().flatten('F'))
        self.assertTrue(eps < self.tau) 
        eps = rel_err(res1[1].flatten('F'), res2[1].full().flatten('F'))
        self.assertTrue(eps < self.tau) 
        eps = rel_err(res1[2].flatten('F'), res2[2].full().flatten('F'))
        self.assertTrue(eps < self.tau) 

class TestUtilsGrid__mesh_rc(TestUtilsGrid):
    ''' 
    Tests for function _mesh_rc
    from module utils.grid.
    '''
        
    def test_np_is_correct_1d(self):
        ''' Check for 1D.  '''
        x = _mesh_rc(self.d, 1, self.tau, mode=MODE_NP)[0]
        self.assertTrue(abs(x[0]-self.h) < ZEROp)
        self.assertTrue(abs(x[1]-2.*self.h) < ZEROp)
        self.assertTrue(abs(x[-1]-1.) < ZEROp)
        self.assertTrue(abs(x[self.n/2+1]-(x[self.n/2]+self.h)) < ZEROp)

    def test_np_vs_tt_1d(self):
        ''' In 1D case results for MODE_NP and MODE_TT should be equal.  '''
        res1 = _mesh_rc(self.d, 1, self.tau, mode=MODE_NP)[0]
        res2 = _mesh_rc(self.d, 1, self.tau, mode=MODE_TT)[0].full().flatten('F')
        eps = rel_err(res1, res2)
        self.assertTrue(eps < self.tau) 

    def test_np_is_correct_2d(self):
        ''' Check for 2D.  '''
        x, y = _mesh_rc(self.d, 2, self.tau, mode=MODE_NP)
        self.assertTrue(abs(x[0, 0]-self.h) < ZEROp)
        self.assertTrue(abs(x[1, 0]-2.*self.h) < ZEROp)
        self.assertTrue(abs(x[0, 1]-self.h) < ZEROp)
        self.assertTrue(abs(x[1, 1]-2.*self.h) < ZEROp)
        self.assertTrue(abs(x[-1, -1]-1.) < ZEROp)

        self.assertTrue(abs(y[0, 0]-self.h) < ZEROp)
        self.assertTrue(abs(y[1, 0]-self.h) < ZEROp)
        self.assertTrue(abs(y[0, 1]-2.*self.h) < ZEROp)
        self.assertTrue(abs(y[1, 1]-2.*self.h) < ZEROp)
        self.assertTrue(abs(y[-1, -1]-1.) < ZEROp)
        
    def test_np_vs_tt_2d(self):
        ''' In 2D case results for MODE_NP and MODE_TT should be equal.  '''
        res1 = _mesh_rc(self.d, 2, self.tau, mode=MODE_NP)
        res2 = _mesh_rc(self.d, 2, self.tau, mode=MODE_TT)
        eps = rel_err(res1[0].flatten('F'), res2[0].full().flatten('F'))
        self.assertTrue(eps < self.tau) 
        eps = rel_err(res1[1].flatten('F'), res2[1].full().flatten('F'))
        self.assertTrue(eps < self.tau) 

    def test_np_is_correct_3d(self):
        ''' Check for 3D.  '''
        x, y, z = _mesh_rc(self.d, 3, self.tau, mode=MODE_NP)
        self.assertTrue(abs(x[0, 0, 0]-self.h) < ZEROp)
        self.assertTrue(abs(x[1, 0, 0]-2.*self.h) < ZEROp)
        self.assertTrue(abs(x[0, 1, 0]-self.h) < ZEROp)
        self.assertTrue(abs(x[1, 1, 0]-2.*self.h) < ZEROp)
        self.assertTrue(abs(x[1, 1, 2]-2.*self.h) < ZEROp)
        self.assertTrue(abs(x[-1, -1, -1]-1.) < ZEROp)

        self.assertTrue(abs(y[0, 0, 0]-self.h) < ZEROp)
        self.assertTrue(abs(y[1, 0, 0]-self.h) < ZEROp)
        self.assertTrue(abs(y[0, 1, 0]-2.*self.h) < ZEROp)
        self.assertTrue(abs(y[1, 1, 0]-2.*self.h) < ZEROp)
        self.assertTrue(abs(y[1, 1, 2]-2.*self.h) < ZEROp)
        self.assertTrue(abs(y[-1, -1, -1]-1.) < ZEROp)
        
        self.assertTrue(abs(z[0, 0, 0]-self.h) < ZEROp)
        self.assertTrue(abs(z[2, 2, 0]-self.h) < ZEROp)
        self.assertTrue(abs(z[0, 1, 1]-2.*self.h) < ZEROp)
        self.assertTrue(abs(z[0, 1, 2]-3.*self.h) < ZEROp)
        self.assertTrue(abs(z[-1, -1, -1]-1.) < ZEROp)
        
    def test_np_vs_tt_3d(self):
        ''' In 3D case results for MODE_NP and MODE_TT should be equal.  '''
        res1 = _mesh_rc(self.d, 3, self.tau, mode=MODE_NP)
        res2 = _mesh_rc(self.d, 3, self.tau, mode=MODE_TT)
        eps = rel_err(res1[0].flatten('F'), res2[0].full().flatten('F'))
        self.assertTrue(eps < self.tau) 
        eps = rel_err(res1[1].flatten('F'), res2[1].full().flatten('F'))
        self.assertTrue(eps < self.tau) 
        eps = rel_err(res1[2].flatten('F'), res2[2].full().flatten('F'))
        self.assertTrue(eps < self.tau)

class TestUtilsGrid__mesh_uxe(TestUtilsGrid):
    ''' 
    Tests for function _mesh_uxe
    from module utils.grid.
    '''
        
    def test_np_is_correct_1d(self):
        ''' Check for 1D.  '''
        x = _mesh_uxe(self.d, 1, self.tau, mode=MODE_NP)[0]
        self.assertTrue(abs(x[0]-self.h/2.) < ZEROp)
        self.assertTrue(abs(x[1]-3.*self.h/2.) < ZEROp)
        self.assertTrue(abs(x[-1]-(1.-self.h/2.)) < ZEROp)
        self.assertTrue(abs(x[self.n/2+1]-(x[self.n/2]+self.h)) < ZEROp)

    def test_np_vs_tt_1d(self):
        ''' In 1D case results for MODE_NP and MODE_TT should be equal.  '''
        res1 = _mesh_uxe(self.d, 1, self.tau, mode=MODE_NP)[0]
        res2 = _mesh_uxe(self.d, 1, self.tau, mode=MODE_TT)[0].full().flatten('F')
        eps = rel_err(res1, res2)
        self.assertTrue(eps < self.tau) 
        
    def test_np_is_correct_2d(self):
        ''' Check for 2D.  '''
        x, y = _mesh_uye(self.d, 2, self.tau, mode=MODE_NP)
        self.assertTrue(abs(x[0, 0]-self.h) < ZEROp)
        self.assertTrue(abs(x[1, 0]-2.*self.h) < ZEROp)
        self.assertTrue(abs(x[0, 1]-self.h) < ZEROp)
        self.assertTrue(abs(x[1, 1]-2.*self.h) < ZEROp)
        self.assertTrue(abs(x[-1, -1]-1.) < ZEROp)

        self.assertTrue(abs(y[0, 0]-self.h/2.) < ZEROp)
        self.assertTrue(abs(y[1, 0]-self.h/2.) < ZEROp)
        self.assertTrue(abs(y[0, 1]-3.*self.h/2) < ZEROp)
        self.assertTrue(abs(y[1, 1]-3.*self.h/2) < ZEROp)
        self.assertTrue(abs(y[-1, -1]-(1.-self.h/2.)) < ZEROp)
        
    def test_np_vs_tt_2d(self):
        ''' In 2D case results for MODE_NP and MODE_TT should be equal.  '''
        res1 = _mesh_uye(self.d, 2, self.tau, mode=MODE_NP)
        res2 = _mesh_uye(self.d, 2, self.tau, mode=MODE_TT)
        eps = rel_err(res1[0].flatten('F'), res2[0].full().flatten('F'))
        self.assertTrue(eps < self.tau) 
        eps = rel_err(res1[1].flatten('F'), res2[1].full().flatten('F'))
        self.assertTrue(eps < self.tau) 
        
    def test_np_incorrect_dim1_2d(self):
        ''' _mesh_uye shouldn't work for 1D.  '''
        raised = False
        try:
            _mesh_uye(self.d, 1, self.tau, mode=MODE_NP)
        except:
            raised = True
        self.assertTrue(raised)
        
    def test_np_is_correct_3d(self):
        ''' Check for 3D.  '''
        x, y, z = _mesh_uze(self.d, 3, self.tau, mode=MODE_NP)
        self.assertTrue(abs(x[0, 0, 0]-self.h) < ZEROp)
        self.assertTrue(abs(x[1, 0, 0]-2.*self.h) < ZEROp)
        self.assertTrue(abs(x[0, 1, 0]-self.h) < ZEROp)
        self.assertTrue(abs(x[1, 1, 0]-2.*self.h) < ZEROp)
        self.assertTrue(abs(x[1, 1, 2]-2.*self.h) < ZEROp)
        self.assertTrue(abs(x[-1, -1, -1]-1.) < ZEROp)

        self.assertTrue(abs(y[0, 0, 0]-self.h) < ZEROp)
        self.assertTrue(abs(y[1, 0, 0]-self.h) < ZEROp)
        self.assertTrue(abs(y[0, 1, 0]-2.*self.h) < ZEROp)
        self.assertTrue(abs(y[1, 1, 0]-2.*self.h) < ZEROp)
        self.assertTrue(abs(y[1, 1, 2]-2.*self.h) < ZEROp)
        self.assertTrue(abs(y[-1, -1, -1]-1.) < ZEROp)
        
        self.assertTrue(abs(z[0, 0, 0]-self.h/2.) < ZEROp)
        self.assertTrue(abs(z[2, 2, 0]-self.h/2.) < ZEROp)
        self.assertTrue(abs(z[0, 1, 1]-3.*self.h/2.) < ZEROp)
        self.assertTrue(abs(z[0, 1, 2]-5.*self.h/2.) < ZEROp)
        self.assertTrue(abs(z[-1, -1, -1]-(1.-self.h/2.)) < ZEROp)
        
    def test_np_incorrect_dim1_3d(self):
        ''' _mesh_uze shouldn't work for 1D.  '''
        raised = False
        try:
            _mesh_uze(self.d, 1, self.tau, mode=MODE_NP)
        except:
            raised = True
        self.assertTrue(raised)
        
    def test_np_incorrect_dim2_3d(self):
        ''' _mesh_uze shouldn't work for 2D.  '''
        raised = False
        try:
            _mesh_uze(self.d, 2, self.tau, mode=MODE_NP)
        except:
            raised = True
        self.assertTrue(raised)
        
    def test_np_vs_tt_3d(self):
        ''' In 3D case results for MODE_NP and MODE_TT should be equal.  '''
        res1 = _mesh_uze(self.d, 3, self.tau, mode=MODE_NP)
        res2 = _mesh_uze(self.d, 3, self.tau, mode=MODE_TT)
        eps = rel_err(res1[0].flatten('F'), res2[0].full().flatten('F'))
        self.assertTrue(eps < self.tau) 
        eps = rel_err(res1[1].flatten('F'), res2[1].full().flatten('F'))
        self.assertTrue(eps < self.tau) 
        eps = rel_err(res1[2].flatten('F'), res2[2].full().flatten('F'))
        self.assertTrue(eps < self.tau)

class TestUtilsGrid__construct_mesh(TestUtilsGrid):
    ''' 
    Tests for function _construct_mesh
    from module utils.grid.
    '''
        
    def test_correct_grid_name(self):
        ''' It should work for correct grid names.  '''
        names = ['cc', 'cell centers', 'lc', 'left corners', 'rc', 'right corners', 'UYe', 'upper-Z EDGE midpoints'] 
        names+= ['uxe', 'upper-x edge midpoints', 'uye', 'upper-y edge midpoints', 'uze', 'upper-z edge midpoints']
        for name in names:
            raised = False
            try:
                _construct_mesh(self.d, 3, tau=None, mode=MODE_NP, grid=name)
            except:
                raised = True
            self.assertFalse(raised)

    def test_incorrect_grid_name(self):
        ''' It shouldn't work for incorrect grid name.  '''
        raised = False
        try:
            _construct_mesh(self.d, 2, tau=None, mode=MODE_NP, grid='xxx')
        except:
            raised = True
        self.assertTrue(raised)
        
class TestUtilsGrid_quan_on_grid(TestUtilsGrid):
    ''' 
    Tests for function quan_on_grid
    from module utils.grid.
    '''
    
    def setUp(self):
        TestUtilsGrid.setUp(self)
        
        def func_1d(x):
            return x*x+1./(x+2.)
        
        def func_1d_tt(x):
            return x*x+1./(x+2.)
        
        def func_2d(x, y):
            return 1.+x*y*y+y/(x+2.)
        
        def func_2d_tt(r):
            x, y = r[:, 0], r[:, 1]
            return 1.+x*y*y+y/(x+2.)
        
        def func_3d(x, y, z):
            return 1.+x*y*z*z+z/(x+y+2.)
        
        def func_3d_tt(r):
            x, y, z = r[:, 0], r[:, 1], r[:, 2]
            return 1.+x*y*z*z+z/(x+y+2.)
            
        self.func_1d = func_1d
        self.func_1d_tt = func_1d_tt
        self.func_2d = func_2d
        self.func_2d_tt = func_2d_tt
        self.func_3d = func_3d
        self.func_3d_tt = func_3d_tt
        
    def test_np_is_correct_1d(self):
        ''' In 1D case it should be correct  '''
        names = ['cc', 'lc', 'rc', 'uxe']
        res2 = [self.func_1d(3.*self.h/2.),
                self.func_1d(self.h), 
                self.func_1d(2*self.h),
                self.func_1d(3.*self.h/2.)]
        for i, name in enumerate(names):
            res1 = quan_on_grid(self.func_1d, self.d, 1, self.tau, self.eps, mode=MODE_NP, grid=name)     
            self.assertTrue(np.abs(res1[1]-res2[i]) < ZEROp)

    def test_np_is_correct_1d_inv(self):
        ''' In 1D case it should be correct  '''
        names = ['cc', 'lc', 'rc', 'uxe']
        res2 = [1./self.func_1d(3.*self.h/2.),
                1./self.func_1d(self.h), 
                1./self.func_1d(2*self.h),
                1./self.func_1d(3.*self.h/2.)]
        for i, name in enumerate(names):
            res1 = quan_on_grid(self.func_1d, self.d, 1, self.tau, self.eps, mode=MODE_NP, grid=name, inv=True)     
            self.assertTrue(np.abs(res1[1]-res2[i]) < ZEROp)
            
    def test_np_vs_tt_1d(self):
        ''' In 1D case for all grid types results for MODE_NP and MODE_TT 
            should be equal.  '''
        names = ['cc', 'lc', 'rc', 'uxe']
        for name in names:
            res1 = quan_on_grid(self.func_1d, self.d, 1, self.tau, self.eps, mode=MODE_NP, grid=name)
            res2 = quan_on_grid(self.func_1d_tt, self.d, 1, self.tau, self.eps, mode=MODE_TT, grid=name)      
            eps = rel_err(res1, res2.full().flatten('F'))
            self.assertTrue(eps < self.eps) 
            
    def test_np_vs_tt_1d_inv(self):
        ''' In 1D case for all grid types results for MODE_NP and MODE_TT 
            should be equal.  '''
        names = ['cc', 'lc', 'rc', 'uxe']
        for name in names:
            res1 = quan_on_grid(self.func_1d, self.d, 1, self.tau, self.eps, mode=MODE_NP, grid=name, inv=True)
            res2 = quan_on_grid(self.func_1d_tt, self.d, 1, self.tau, self.eps, mode=MODE_TT, grid=name, inv=True)      
            eps = rel_err(res1, res2.full().flatten('F'))
            self.assertTrue(eps < self.eps)
            
    def test_np_is_correct_2d(self):
        ''' In 2D case it should be correct  '''
        names = ['cc', 'lc', 'rc', 'uxe', 'uye']
        res2 = [self.func_2d(3.*self.h/2., self.h/2.),
                self.func_2d(self.h, 0.), 
                self.func_2d(2.*self.h, self.h),
                self.func_2d(3.*self.h/2., self.h),
                self.func_2d(2.*self.h, self.h/2.)]
        for i, name in enumerate(names):
            res1 = quan_on_grid(self.func_2d, self.d, 2, self.tau, self.eps, mode=MODE_NP, grid=name)     
            self.assertTrue(np.abs(res1[1]-res2[i]) < ZEROp)
            
    def test_np_is_correct_2d_inv(self):
        ''' In 2D case it should be correct  '''
        names = ['cc', 'lc', 'rc', 'uxe', 'uye']
        res2 = [1./self.func_2d(3.*self.h/2., self.h/2.),
                1./self.func_2d(self.h, 0.), 
                1./self.func_2d(2.*self.h, self.h),
                1./self.func_2d(3.*self.h/2., self.h),
                1./self.func_2d(2.*self.h, self.h/2.)]
        for i, name in enumerate(names):
            res1 = quan_on_grid(self.func_2d, self.d, 2, self.tau, self.eps, mode=MODE_NP, grid=name, inv=True)     
            self.assertTrue(np.abs(res1[1]-res2[i]) < ZEROp)
            
    def test_np_vs_tt_2d(self):
        ''' In 2D case for all grid types results for MODE_NP and MODE_TT 
            should be equal.  '''
        names = ['cc', 'lc', 'rc', 'uxe', 'uye']
        for name in names:
            res1 = quan_on_grid(self.func_2d, self.d, 2, self.tau, self.eps, mode=MODE_NP, grid=name)
            res2 = quan_on_grid(self.func_2d_tt, self.d, 2, self.tau, self.eps, mode=MODE_TT, grid=name)      
            eps = rel_err(res1, res2.full().flatten('F'))
            self.assertTrue(eps < self.eps) 
            
    def test_np_vs_tt_2d_inv(self):
        ''' In 2D case for all grid types results for MODE_NP and MODE_TT 
            should be equal.  '''
        names = ['cc', 'lc', 'rc', 'uxe', 'uye']
        for name in names:
            res1 = quan_on_grid(self.func_2d, self.d, 2, self.tau, self.eps, mode=MODE_NP, grid=name, inv=True)
            res2 = quan_on_grid(self.func_2d_tt, self.d, 2, self.tau, self.eps, mode=MODE_TT, grid=name, inv=True)      
            eps = rel_err(res1, res2.full().flatten('F'))
            self.assertTrue(eps < self.eps)
            
    def test_np_is_correct_3d(self):
        ''' In 3D case it should be correct  '''
        names = ['cc', 'lc', 'rc', 'uxe', 'uye', 'uze']
        res2 = [self.func_3d(3.*self.h/2., self.h/2., self.h/2.),
                self.func_3d(self.h, 0., 0.), 
                self.func_3d(2.*self.h, self.h, self.h),
                self.func_3d(3.*self.h/2., self.h, self.h),
                self.func_3d(2.*self.h, self.h/2., self.h),
                self.func_3d(2.*self.h, self.h, self.h/2.)]
        for i, name in enumerate(names):
            res1 = quan_on_grid(self.func_3d, self.d, 3, self.tau, self.eps, mode=MODE_NP, grid=name)     
            self.assertTrue(np.abs(res1[1]-res2[i]) < ZEROp)

    def test_np_is_correct_3d_inv(self):
        ''' In 3D case it should be correct  '''
        names = ['cc', 'lc', 'rc', 'uxe', 'uye', 'uze']
        res2 = [1./self.func_3d(3.*self.h/2., self.h/2., self.h/2.),
                1./self.func_3d(self.h, 0., 0.), 
                1./self.func_3d(2.*self.h, self.h, self.h),
                1./self.func_3d(3.*self.h/2., self.h, self.h),
                1./self.func_3d(2.*self.h, self.h/2., self.h),
                1./self.func_3d(2.*self.h, self.h, self.h/2.)]
        for i, name in enumerate(names):
            res1 = quan_on_grid(self.func_3d, self.d, 3, self.tau, self.eps, mode=MODE_NP, grid=name, inv=True)     
            self.assertTrue(np.abs(res1[1]-res2[i]) < ZEROp)
            
    def test_np_vs_tt_3d(self):
        ''' In 3D case for all grid types results for MODE_NP and MODE_TT 
            should be equal.  '''
        names = ['cc', 'lc', 'rc', 'uxe', 'uye', 'uze']
        for name in names:
            res1 = quan_on_grid(self.func_3d, self.d, 3, self.tau, self.eps, mode=MODE_NP, grid=name)
            res2 = quan_on_grid(self.func_3d_tt, self.d, 3, self.tau, self.eps, mode=MODE_TT, grid=name)      
            eps = rel_err(res1, res2.full().flatten('F'))
            self.assertTrue(eps < self.eps) 
            
    def test_np_vs_tt_3d_inv(self):
        ''' In 3D case for all grid types results for MODE_NP and MODE_TT 
            should be equal.  '''
        names = ['cc', 'lc', 'rc', 'uxe', 'uye', 'uze']
        for name in names:
            res1 = quan_on_grid(self.func_3d, self.d, 3, self.tau, self.eps, mode=MODE_NP, grid=name, inv=True)
            res2 = quan_on_grid(self.func_3d_tt, self.d, 3, self.tau, self.eps, mode=MODE_TT, grid=name, inv=True)      
            eps = rel_err(res1, res2.full().flatten('F'))
            self.assertTrue(eps < self.eps)
            
class TestUtilsGrid_coord2ind(TestUtilsGrid):
    ''' 
    Tests for function coord2ind
    from module utils.grid.
    '''
        
    def test_1d(self):
        ''' Check for 1D.  '''
        L = 2.
        d = 3
        n = 2**d
        h = L/n
        ind_exp = [0, 0, 0, 0, 1, 1, n-2, n-1, n-1]
        for i, r in enumerate([[-1.], [0.], [h], [h*1.4], [h*1.51], [h*1.99], [h*n-h*0.6], [h*n-h*0.1], [h*n+10]]):
            ind = coord2ind(r, d, L)
            self.assertEqual(ind, ind_exp[i])
    
    def test_2d(self):
        ''' Check for 2D.  '''
        L = 2.
        d = 3
        n = 2**d
        h = L/n
        
        ind_exp = [0, 0, 0, 0, 9, 9, 54, 63, 63]
        for i, r in enumerate([[-1., -1.], [0., 0.], [h, h], [h*1.4, h*1.4], [h*1.51, h*1.51], 
                  [h*1.99, h*1.99], [h*n-h*0.6, h*n-h*0.6], [h*n-h*0.1, h*n-h*0.1], [h*n+10, h*n+10]]):
            ind = coord2ind(r, d, L)
            self.assertEqual(ind, ind_exp[i])
    
    def test_3d(self):
        ''' Check for 3D.  '''
        L = 2.
        d = 2
        n = 2**d
        h = L/n
        
        ind_exp = [0, 0, 0, 0, 21, 21, 42, 63, 63]
        for i, r in enumerate([[-1.]*3, [0.]*3, [h]*3, [h*1.4]*3, [h*1.51]*3, 
                               [h*1.99]*3, [h*n-h*0.6]*3, [h*n-h*0.1]*3, [h*n+10]*3]):
            ind = coord2ind(r, d, L)
            self.assertEqual(ind, ind_exp[i])

class TestUtilsGrid_delta_on_grid(TestUtilsGrid):
    ''' 
    Tests for function delta_on_grid
    from module utils.grid.
    '''
        
    def test_1d_np(self):
        ''' Check for 1D.  '''
        L = 1.
        d = 3
        n = 2**d
        h = L/n
        r, val = [2*h+0.45*h], 3.
        res = delta_on_grid(r, val, d, tau=1.E-8, mode=MODE_NP)
        self.assertEqual(len(res.shape), 1)
        self.assertEqual(res.shape[0], n)
        self.assertTrue(abs(np.sum(res)-val/h) < ZEROp)

    def test_1d_np_vs_tt(self):
        ''' In 1D case results for MODE_NP and MODE_TT should be equal.  '''
        L = 1.
        d = 3
        n = 2**d
        h = L/n
        r, val = [2*h+0.45*h], 3.
        res1 = delta_on_grid(r, val, d, tau=1.E-8, mode=MODE_NP)
        res2 = delta_on_grid(r, val, d, tau=1.E-8, mode=MODE_TT).full().flatten('F')
        self.assertTrue(rel_err(res1, res2) < ZEROp) 
        
    def test_2d_np(self):
        ''' Check for 2D.  '''
        L = 1.
        d = 3
        n = 2**d
        h = L/n
        r, val = [2*h+0.45*h, 2*h+0.45*h], 2.
        res = delta_on_grid(r, val, d, tau=1.E-8, mode=MODE_NP)
        self.assertEqual(len(res.shape), 1)
        self.assertEqual(res.shape[0], n*n)
        self.assertTrue(abs(np.sum(res)-val/h/h) < ZEROp)

    def test_2d_np_vs_tt(self):
        ''' In 2D case results for MODE_NP and MODE_TT should be equal.  '''
        L = 1.
        d = 3
        n = 2**d
        h = L/n
        r, val = [2*h+0.45*h]*2, 3.
        res1 = delta_on_grid(r, val, d, tau=1.E-8, mode=MODE_NP)
        res2 = delta_on_grid(r, val, d, tau=1.E-8, mode=MODE_TT).full().flatten('F')
        self.assertTrue(rel_err(res1, res2) < ZEROp) 
         
    def test_3d_np(self):
        ''' Check for 3D.  '''
        L = 1.
        d = 3
        n = 2**d
        h = L/n
        r, val = [2*h+0.45*h, 2*h+0.45*h, 2*h+0.45*h], 2.
        res = delta_on_grid(r, val, d, tau=1.E-8, mode=MODE_NP)
        self.assertEqual(len(res.shape), 1)
        self.assertEqual(res.shape[0], n*n*n)
        self.assertTrue(abs(np.sum(res)-val/h/h/h) < ZEROp)

    def test_3d_np_vs_tt(self):
        ''' In 3D case results for MODE_NP and MODE_TT should be equal.  '''
        L = 1.
        d = 3
        n = 2**d
        h = L/n
        r, val = [2*h+0.45*h]*3, 4.
        res1 = delta_on_grid(r, val, d, tau=1.E-8, mode=MODE_NP)
        res2 = delta_on_grid(r, val, d, tau=1.E-8, mode=MODE_TT).full().flatten('F')
        self.assertTrue(rel_err(res1, res2) < ZEROp) 
        
class TestUtilsGrid_deltas_on_grid(TestUtilsGrid):
    ''' 
    Tests for function deltas_on_grid
    from module utils.grid.
    '''
        
    def test_1d_np(self):
        ''' Check for 1D.  '''
        L = 1.
        d = 3
        n = 2**d
        h = L/n
        r, val = [[2*h+0.45*h], [3*h+0.45*h]], [3., 4.]
        res = deltas_on_grid(r, val, d, tau=1.E-8, mode=MODE_NP)
        self.assertEqual(len(res.shape), 1)
        self.assertEqual(res.shape[0], n)
        self.assertTrue(abs(np.sum(res)-val[0]/h-val[1]/h) < ZEROp)

    def test_1d_np_vs_tt(self):
        ''' In 1D case results for MODE_NP and MODE_TT should be equal.  '''
        L = 1.
        d = 3
        n = 2**d
        h = L/n
        r, val = [[2*h+0.45*h], [3*h+0.45*h]], [3., 4.]
        res1 = deltas_on_grid(r, val, d, tau=1.E-8, mode=MODE_NP)
        res2 = deltas_on_grid(r, val, d, tau=1.E-8, mode=MODE_TT).full().flatten('F')
        self.assertTrue(rel_err(res1, res2) < ZEROp) 
        
    def test_2d_np(self):
        ''' Check for 2D.  '''
        L = 1.
        d = 3
        n = 2**d
        h = L/n
        r, val = [[2*h+0.45*h, 2*h+0.45*h], [2*h+0.45*h, 2*h+0.45*h]], [2., 5.]
        res = deltas_on_grid(r, val, d, tau=1.E-8, mode=MODE_NP)
        self.assertEqual(len(res.shape), 1)
        self.assertEqual(res.shape[0], n*n)
        self.assertTrue(abs(np.sum(res)-val[0]/h/h-val[1]/h/h) < ZEROp)

    def test_2d_np_vs_tt(self):
        ''' In 2D case results for MODE_NP and MODE_TT should be equal.  '''
        L = 1.
        d = 3
        n = 2**d
        h = L/n
        r, val = [[2*h+0.45*h, 2*h+0.45*h], [2*h+0.45*h, 2*h+0.45*h]], [2., 5.]
        res1 = deltas_on_grid(r, val, d, tau=1.E-8, mode=MODE_NP)
        res2 = deltas_on_grid(r, val, d, tau=1.E-8, mode=MODE_TT).full().flatten('F')
        self.assertTrue(rel_err(res1, res2) < ZEROp) 
         
    def test_3d_np(self):
        ''' Check for 3D.  '''
        L = 1.
        d = 3
        n = 2**d
        h = L/n
        r, val = [[2*h+0.45*h, 2*h+0.45*h, 2*h+0.45*h] for _ in range(3)], [2., 1., 6.]
        res = deltas_on_grid(r, val, d, tau=1.E-8, mode=MODE_NP)
        self.assertEqual(len(res.shape), 1)
        self.assertEqual(res.shape[0], n*n*n)
        self.assertTrue(abs(np.sum(res)-val[0]/h/h/h-val[1]/h/h/h-val[2]/h/h/h) < ZEROp)

    def test_3d_np_vs_tt(self):
        ''' In 3D case results for MODE_NP and MODE_TT should be equal.  '''
        L = 1.
        d = 3
        n = 2**d
        h = L/n
        r, val = [[2*h+0.45*h, 2*h+0.45*h, 2*h+0.45*h] for _ in range(3)], [2., 1., 6.]
        res1 = deltas_on_grid(r, val, d, tau=1.E-8, mode=MODE_NP)
        res2 = deltas_on_grid(r, val, d, tau=1.E-8, mode=MODE_TT).full().flatten('F')
        self.assertTrue(rel_err(res1, res2) < ZEROp) 
        
if __name__ == '__main__':
    unittest.main()