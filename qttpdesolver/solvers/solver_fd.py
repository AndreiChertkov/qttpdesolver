# -*- coding: utf-8 -*-
import time
import numpy as np
from scipy.sparse import coo_matrix

import tt

from ..utils.general import MODE_NP, MODE_TT, MODE_SP, mprod, msum, mvec, vdiag
from ..utils.block_and_space import space_kron
from ..utils.grid import quan_on_grid, deltas_on_grid
from ..utils.spec_matr import findif
from ..utils.alg_syst_solver import alg_syst_solve

class SolverFd(object):
    '''
    Class-based realization of Finite Difference (FD) solver for
    elliptic PDEs of the form -div(k grad u) = f in 1D / 2D / 3D.
    PDE parameters are specified by PDE class instance while __init__.
    
        Function generate_matrices
        * construct all used vectors and matrices as a class members:
    
    K   - is a list of diagonal matrices of k-coeffs on a spatial mesh
    F   - is a vector of f-coeff values on a spatial mesh
    B   - is a list of matrices that are spreading of 
          fin. dif. operator on PDE.dim dimensions
    s   - is a list of matrices that are spreading of 
          diag([1,1,...,1,0]) matrix on PDE.dim dimensions
    z   - is a list of matrices that are spreading of 
          diag([0,0,...,0,1]) matrix on PDE.dim dimensions
    
        Function generate_system
        * construct A and rhs in A x = rhs as a class members:
    
    A   - is a matrix of linear system
    rhs - is a rhs vector
    
        Function solve
        * solve the matrix equation A x = rhs, construct PDE solution and
          its derivatives

        Comments
        * Variables of list type keep values for different space dimensions:
          B[0] - x-dim, B[1] - y-dim (2D, 3D), B[2] - z-dim (3D) and so on.
        * Times, solution and amen results are automatically saved to PDE.
    '''
    
    def __init__(self, PDE):
        self.set_pde(PDE)
           
    def set_pde(self, PDE):
        self.PDE = PDE
        
    def generate_matrices(self):
        PDE = self.PDE; _time = time.time()
        d, n, h, dim, mode, tau_round, tau_cross = PDE.d, PDE.n, PDE.h,\
          PDE.dim, PDE.mode, PDE.tau['mgen']['round'], PDE.tau['mgen']['cross'] 
        
        _t = time.time()
        self.D = [None]*dim
        if dim>=1:
            self.D[0] = quan_on_grid(PDE.k_func, d, dim, tau_round, tau_cross,
                                     mode, 'uxe', 'Dx', PDE.verb_cross)
        if dim>=2:
            self.D[1] = quan_on_grid(PDE.k_func, d, dim, tau_round, tau_cross,
                                     mode, 'uye', 'Dy', PDE.verb_cross)
        if dim>=3:
            self.D[2] = quan_on_grid(PDE.k_func, d, dim, tau_round, tau_cross,
                                     mode, 'uze', 'Dz', PDE.verb_cross)
        self.D = [vdiag(self.D[i], tau_round, to_sp=(mode==MODE_SP))
                  for i in range(dim)]
        PDE.t['D'] = time.time()-_t 
        
        if mode == MODE_TT:
            PDE.d_erank = [self.D[i].erank for i in range(dim)]

        _t = time.time()
        if PDE.f_poi_src_val_list is None: 
            self.F = quan_on_grid(PDE.f_func, d, dim, tau_round, tau_cross,
                                  mode, 'rc', 'F', PDE.verb_cross)
        else: # Point source case          
            self.F = deltas_on_grid(PDE.f_poi_src_xyz_list, 
                                    PDE.f_poi_src_val_list, 
                                    d, tau_round, mode, 'rc')
        PDE.t['F'] = time.time()-_t
        
        if mode == MODE_TT:
            PDE.f_erank = self.F.erank
            
        _t = time.time() 
        Bi = findif(d, n, h, tau_round, mode)
        self.B = [space_kron(Bi, i, d, n, dim, tau_round) for i in range(dim)]
        PDE.t['B'] = time.time()-_t

        if mode != MODE_TT:
            si = coo_matrix((np.ones(n-1), (np.arange(n-1), np.arange(n-1))), 
                            shape=(n, n)).tocsr()
            zi = coo_matrix(([1], ([n-1], [n-1])), 
                            shape=(n, n)).tocsr() 
            if mode == MODE_NP:
                si = si.toarray()
                zi = zi.toarray()
        else:
            e1 = tt.tensor(np.array([0.0, 1.0]))
            e2 = tt.mkron([e1] * d)
            si = vdiag(tt.ones(2, d)-e2, tau_round) 
            zi = vdiag(e2, tau_round) 
        self.s = [space_kron(si, i, d, n, dim, tau_round) for i in range(dim)]
        self.z = [space_kron(zi, i, d, n, dim, tau_round) for i in range(dim)]
        
        PDE.t['mgen'] = time.time()-_time          
        if PDE.verb_gen:
            PDE._present('Time of matrices generation: %-8.4f'%PDE.t['mgen'])
          
    def generate_system(self):
        PDE = self.PDE; _time = time.time()
        dim, mode, tau_round = PDE.dim, PDE.mode, PDE.tau['sgen']['round']
        
        _t = time.time()
        self.A = None
        for i in range(dim):
            Ai = mprod([self.s[i], self.B[i].T, self.D[i], self.B[i], self.s[i]], tau_round)
            Ai = msum([Ai, self.z[i]], tau_round)
            if self.A is None:
                self.A = Ai
            else:
                self.A = msum([self.A, Ai], tau_round)
        PDE.t['A'] = time.time()-_t 

        _t = time.time()
        self.rhs = self.F
        for i in range(dim):
            self.rhs = mvec(self.s[i], self.rhs, tau_round)
        PDE.t['rhs'] = time.time()-_t 
        
        if mode == MODE_TT:
            PDE.a_erank, PDE.rhs_erank = self.A.erank, self.rhs.erank
            
        PDE.t['sgen'] = time.time()-_time
        if PDE.verb_gen:
            PDE._present('Time of system generation  : %-8.4f'%PDE.t['sgen'])
        
    def solve(self):
        PDE = self.PDE; _time = time.time()

        _t = time.time() 
        PDE.u_calc = alg_syst_solve(self.A, self.rhs, PDE.sol0, 
                                    PDE.tau['solve']['amens'], PDE.algss_par, PDE.verb_amen)
        PDE.t['amen'] = time.time() - _t
        if PDE.mode==MODE_TT:
            PDE.u_calc_ranks, PDE.u_calc_erank = PDE.u_calc.r, PDE.u_calc.erank
            
        if PDE.algss_par['tau_u_calc_from_algss']:
            tau_round = PDE.algss_par['max_res']
        else:
            tau_round = PDE.tau['u_calc']['round']
            
        _t = time.time()
        PDE.ud_calc = [mvec(self.B[i], PDE.u_calc, tau_round) for i in range(PDE.dim)]
        PDE.t['ud_calc'] = time.time() - _t
            
        PDE.t['solve'] = time.time()-_time                      
        if PDE.verb_gen:
            PDE._present('Time of system solving     : %-8.4f'%PDE.t['solve'])  