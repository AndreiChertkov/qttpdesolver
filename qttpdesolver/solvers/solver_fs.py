# -*- coding: utf-8 -*-
import time

from ..utils.general import MODE_TT, MODE_SP, msum, mprod, mvec, vdiag, vsum, vinv 
from ..utils.block_and_space import mblock, vblock, space_kron, sum_out, kron_out
from ..utils.spec_matr import eye, volterra
from ..utils.splitting import half_array
from ..utils.grid import quan_on_grid, deltas_on_grid
from ..utils.alg_syst_solver import alg_syst_solve

class SolverFs(object):
    '''
    Class-based realization of Finite Sum (FS) solver for
    elliptic PDEs of the form -div(k grad u) = f in 1D / 2D / 3D.
    PDE parameters are specified by PDE class instance while __init__.
    
        Function generate_matrices
        * construct all used vectors and matrices as a class members:

    iD  - is a list of diagonal inverse matrices of k-coeffs on a spatial mesh
    F   - is a vector of f-coeff values on a spatial mesh
    Bi  - is a matrix discretization of the Volterra integral operator
    R   - is a list of matrices of the form R[i]= iD[i] (I - W[i] iD[i]) B[i]^T
    H   - is a list of matrices of the form H[i]= B[i] R[i] 
          * R and H are constructed only for 2D and 3D cases
    
        Function generate_system
        * works for 2D and 3D (in 1D case it is nothing to do)
        * construct A and rhs in A x = rhs (for 2D, 3D) as a class members:
        
    A   - is a matrix of linear system
    rhs - is a rhs vector

        Function solve
        * in 1D case the explicit formula for solution is used
        * in 2D and 3D cases solve the matrix equation A x = rhs, construct 
          PDE solution and its derivatives, and also save the following 
          vectors and matrices as a class members:

    wx  - temporary variable w_x in a vector form (only for 2D, 3D cases)
    wy  - temporary variable w_y in a vector form (only for 3D case)

        Comments
        * Variables of list type keep values for different space dimensions:
          H[0] - x-dim, H[1] - y-dim (2D, 3D), H[2] - z-dim (3D) and so on.
        * Times, solution and AMEn results are automatically saved to PDE.
        * For 1d case tau['mgen'] is used for all operations.
    '''
    
    def __init__(self, PDE):
        self.set_pde(PDE)

    def set_pde(self, PDE):
        self.PDE = PDE
        if self.PDE.mode == MODE_SP:
           raise ValueError('MODE_SP is not available for Solver-FS.') 
           
    def generate_matrices(self):
        PDE = self.PDE; _time = time.time()
        d, n, h, dim, mode, tau_round, tau_cross = PDE.d, PDE.n, PDE.h,\
          PDE.dim, PDE.mode, PDE.tau['mgen']['round'], PDE.tau['mgen']['cross'] 
        
        _t = time.time()
        self.iD = [None]*dim
        if dim>=1:
            self.iD[0] = quan_on_grid(PDE.k_func, d, dim, tau_round, tau_cross,
                                      mode, 'uxe', 'iDx', PDE.verb_cross, inv=True)
        if dim>=2:
            self.iD[1] = quan_on_grid(PDE.k_func, d, dim, tau_round, tau_cross,
                                      mode, 'uye', 'iDy', PDE.verb_cross, inv=True)
        if dim>=3:
            self.iD[2] = quan_on_grid(PDE.k_func, d, dim, tau_round, tau_cross,
                                      mode, 'uze', 'iDz', PDE.verb_cross, inv=True)
        PDE.t['D'] = time.time()-_t     

        if mode == MODE_TT:
            PDE.id_erank = [self.iD[i].erank for i in range(dim)]
            
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
        self.Bi = volterra(d, n, h, tau_round, mode)
        B = [space_kron(self.Bi, i, d, n, dim, tau_round) for i in range(dim)]
        PDE.t['B'] = time.time()-_t
          
        if dim == 1:
            _t = time.time()   
            self.iD = [vdiag(self.iD[0], tau_round)]
            PDE.t['D']+= time.time()-_t 
            PDE.t['mgen'] = time.time()-_time
            if PDE.verb_gen:
                PDE._present('Time of matrices generation: %-8.4f'%PDE.t['mgen'])
            return
            
        _t = time.time()
        iQ = [sum_out(self.iD[i], i, d, n, dim, tau_round) for i in range(dim)]
        Q = [vinv(iQ[i], tau_cross, tau_round, 
                  name='Q%d'%i, verb=PDE.verb_cross) for i in range(dim)] 
        PDE.t['Q'] = time.time()-_t 

        if mode == MODE_TT:
            PDE.q_erank = [Q[i].erank for i in range(dim)]
            PDE.iq_erank = [iQ[i].erank for i in range(dim)]
            
        _t = time.time()
        W = [kron_out(Q[i], i, d, n, dim, tau_round) for i in range(dim)]
        PDE.t['W'] = time.time()-_t 
        
        _t = time.time()   
        self.iD = [vdiag(self.iD[i], tau_round) for i in range(dim)]
        PDE.t['D']+= time.time()-_t  

        _t = time.time() 
        if mode != MODE_TT:
            I = eye(d*dim, n**dim, mode)
        else:                            # n**dim may be too large for MODE_TT.
            I = eye(d*dim, None, mode) 
        WiD = [mprod([W[i], self.iD[i]], tau_round) for i in range(dim)]
        ImWiD = [msum([I, (-1.)*WiD[i]], tau_round) for i in range(dim)]
        self.R = [mprod([self.iD[i], ImWiD[i], B[i].T], tau_round)
                  for i in range(dim)]
        PDE.t['R'] = time.time()-_t 
        
        _t = time.time()
        self.H = [mprod([B[i], self.R[i]], tau_round) for i in range(dim)]    
        PDE.t['H'] = time.time()-_t 

        if mode == MODE_TT:
            PDE.w_erank = [W[i].erank for i in range(dim)]
            PDE.r_erank = [self.R[i].erank for i in range(dim)]
            PDE.h_erank = [self.H[i].erank for i in range(dim)]
            
        PDE.t['mgen'] = time.time()-_time
        if PDE.verb_gen:
            PDE._present('Time of matrices generation: %-8.4f'%PDE.t['mgen'])

    def generate_system(self):
        PDE = self.PDE; _time = time.time()
        dim, mode, tau_round = PDE.dim, PDE.mode, PDE.tau['sgen']['round']
        
        if dim == 1:
            PDE.t['sgen'] = time.time()-_time
            if PDE.verb_gen:
                PDE._present('Time of system generation  : %-8.4f'%PDE.t['sgen'])
            return
            
        _t = time.time()
        if dim == 2:
            self.A = msum([self.H[0], self.H[1]], tau_round)
        if dim == 3:
            Hx = msum([self.H[2], self.H[0]], tau_round)
            Hy = msum([self.H[2], self.H[1]], tau_round)
            #                 mu1        mu2
            self.A = mblock([[Hx       , self.H[2] ], 
                             [self.H[2], Hy        ]], 
                            tau_round)
        PDE.t['A'] = time.time()-_t 
        
        _t = time.time()
        self.rhs = mvec(self.H[-1], self.F, tau_round)
        if dim == 3:
            self.rhs = vblock([self.rhs, self.rhs], tau_round)
        PDE.t['rhs'] = time.time()-_t 

        if mode == MODE_TT:
            PDE.a_erank, PDE.rhs_erank = self.A.erank, self.rhs.erank
            
        PDE.t['sgen'] = time.time()-_time
        if PDE.verb_gen:
            PDE._present('Time of system generation  : %-8.4f'%PDE.t['sgen'])
        
    def solve(self):
        PDE = self.PDE; _time = time.time()
        
        if PDE.dim == 1:
            tau_round = PDE.tau['mgen']['round']
            iD, B = vdiag(self.iD[0], tau_round), self.Bi
            g = mvec(B.T, self.F, tau_round) * iD
            s = vsum(g, tau_round)
            s/= vsum(iD, tau_round)
            ux = msum([g, -iD*s], tau_round)
            PDE.ud_calc, PDE.u_calc = [ux], mvec(B, ux, tau_round)
            if PDE.mode==MODE_TT:
                PDE.u_calc_ranks, PDE.u_calc_erank = PDE.u_calc.r, PDE.u_calc.erank
            PDE.t['solve'] = time.time()-_time
            if PDE.verb_gen:
                PDE._present('Time of system solving     : %-8.4f'%PDE.t['solve'])
            return

        _t = time.time() 
        sol = alg_syst_solve(self.A, self.rhs, PDE.sol0, 
                             PDE.tau['solve']['amens'], PDE.algss_par, PDE.verb_amen)
        PDE.t['amen'] = time.time() - _t
        
        if PDE.algss_par['tau_u_calc_from_algss']:
            tau_round = PDE.algss_par['max_res']
        else:
            tau_round = PDE.tau['u_calc']['round']
            
        if PDE.dim == 2:
            self.wx = sol
            self.wy = None
            _t = time.time()
            PDE.ud_calc = [mvec(self.R[0], self.wx, tau_round),
                           mvec(self.R[1], msum([self.F, -1.*self.wx], tau_round), tau_round)]
            PDE.t['ud_calc'] = time.time() - _t
        if PDE.dim == 3:
            self.wx = half_array(sol, 0, tau_round)
            self.wy = half_array(sol, 1, tau_round)
            _t = time.time()
            PDE.ud_calc = [mvec(self.R[0], self.wx, tau_round),
                           mvec(self.R[1], self.wy, tau_round),
                           mvec(self.R[2], msum([self.F, -1.*self.wx, -1.*self.wy], tau_round), tau_round)]
            PDE.t['ud_calc'] = time.time() - _t
            
        PDE.u_calc = mvec(self.H[0], self.wx, tau_round)
        if PDE.mode==MODE_TT:
            PDE.u_calc_ranks, PDE.u_calc_erank = PDE.u_calc.r, PDE.u_calc.erank
        
        PDE.t['solve'] = time.time()-_time
        if PDE.verb_gen:
            PDE._present('Time of system solving     : %-8.4f'%PDE.t['solve'])     