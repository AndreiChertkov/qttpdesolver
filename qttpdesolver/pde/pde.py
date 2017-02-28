# -*- coding: utf-8 -*-
from copy import deepcopy

from ..tensor_wrapper import LinSystSolver
from ..utils.grid import Grid
from plots import plot
from txts import compose_res_1s, compose_res, compose_info
from model_pde import ModelPde

from ..solvers.solver import BC_HD

class Pde(ModelPde):
    '''
    A container for PDE of the form -div(k grad u) = f.
    See parent class ModelPDE for more details.
    
    * Function clean should be called before each calculation!
    
                 Global parameters [and default values]
                 
    GRD        - [Grid class instance] spatial grid
                 * is set by set_grd function
    LSS        - [LinSystSolver class instance] solver of linear systems 
                 * is set by set_lss function and then may be configured 
                 by set_lss_params function                    
    d          - [None] scale of grid size
    n          - [None] 1d grid size (=2^d)
    h          - [None] grid step (=L/n)
                 * d, n, h are set by update_d function
                 * Domain size may be only L=1
    solver_txt - [None] name of the used PDE solver
                 * is set by set_solver_txt function
                 * available options: 'fs' (SOLVER_FS) and 'fd' (SOLVER_FD)
    bc         - [BC_HD] is the type of boundary conditions
                 * available options: 
                     'hd' (BC_HD) - homogeneous Dirichlet,
                     'pr' (BC_PR) - periodic
                 * is set by set_bc function
    verb_gen   - [False] is the verbosity of general output (times and so on)
    verb_crs   - [False] is the verbosity of output for cross iterations
    verb_lss   - [False] is the verbosity of output for LSS iterations
                 * verb_* are set by set_verb function
                 * verb_* are for Solver's output verbosity only
    sol0       - [None] initial guess for solution of linear system:
                 for solver_fs - is a guess for mu_x (2d) or [mu_x,mu_y] (3d),
                 for solver_fd - is a guess for PDE solution u_calc
                 * is set by set_sol0 function
    tau        - [None] tt-accuracy for round-operations and cross-function
    eps_lss    - [None] accuracy for linear system solver
    tau_lss    - [None] tt-accuracy for round-operations for result of linear
                 system solution (if is None, then the real accuracy will be used)
    tau_real   - [None] tt-accuracy for construction of real (analytical)
                 solution (if is None, then it will be selected as tau * 1.E-2)
                 * tau_* and eps_lss are set by set_tau function

                 Calculation results [and default values]
                 * are set by solver in auto mode after calculation
                 * function clean set default values for this variables

    u_calc       - [None] calculated solution of the PDE
    u_calc_ranks - [None] ranks or u_calc
    u_real       - [None] exact (analitical) solution of the PDE
    u_real_ranks - [None] ranks or u_real      
    u_err        - [None] is the norm of the u calculation's error  
     
    ux_calc      - [None] calculated derivative du/dx of the solution
    ux_real      - [None] exact (analitical) derivative du/dx of the solution
    ux_err       - [None] is the norm of the du/dx calculation's error
    ... the same for uy and uz ...
    
    uu_calc      - [None] product (u_calc, u_calc)
    uu_real      - [None] product (u_real, u_real)
    uu_err       - [None] error of the calculated value of uu_calc   
    
    uf_calc      - [None] product (u_calc, f)
    uf_real      - [None] product (u_real, f)
    uf_err       - [None] error of the calculated value of uf_calc
    
    t_full       - [None] full time of the Solver work
    r            - [dict of None] is dictionary with eranks of the main arrays
    t            - [dict of None] is dictionary with times of solution steps
    
                 Functions
                 
    set_grd        - set spatial grid class instance
                     * Default grid is set automatically
    set_lss        - set linear system solver class instance
                     * Default linear system solver is set automatically
    set_lss_params - set parameters for linear system solver
                     * Default parameters are set automatically
    set_solver_txt - set solver_txt
    set_verb       - set verb_gen, verb_crs and verb_lss
    set_sol0       - set sol0
    set_tau        - set all tt-accuracies
    update_d       - set a given value of d, recalculate n and h values               
    clean          - remove all calculation results (set None-values)
    present_info   - present parameters of the calculation in full text mode
    present_res_1s - present calculation results in one string mode
    present_res    - present calculation results in full text mode
    plot_res       - plot u_real, u_calc (one or both of them may be None)
                     and its derivatives 
                     * call this function only for moderate d values          
    copy           - copy parameters and calc results to a new Pde instanсe 
                     * functions and arrays are not copied by default
    '''
    
    def __init__(self, GRD=None, LSS=None):
        super(Pde, self).__init__()
        self.set_grd(GRD)
        self.set_lss(LSS)
        self.set_lss_params()
        self.set_solver_txt(None)
        self.set_bc(BC_HD)
        self.set_verb(False, False, False)
        self.set_sol0(None)
        self.set_tau()
        self.update_d()
        self.clean()

    @property
    def t_full(self):
        t = 0
        if self.t['cgen'] is not None:
            t+= self.t['cgen']
        if self.t['mgen'] is not None:
            t+= self.t['mgen']
        if self.t['sgen'] is not None:
            t+= self.t['sgen']
        if self.t['soln'] is not None:
            t+= self.t['soln'] 
        if t==0:
            t = None
        return t
         
    def set_grd(self, GRD=None):
        if GRD is not None:
            self.GRD = GRD
        else:
            self.GRD = Grid()
            
    def set_lss(self, LSS=None):
        if LSS is not None:
            self.LSS = LSS
        else:
            self.LSS = LinSystSolver()
            
    def set_lss_params(self, nswp=20, kickrank=4, local_prec='n', local_iters=2,
                      local_restart=40, trunc_norm=1, max_full_size=50):  
        self.LSS.set_params(nswp, kickrank, local_prec, local_iters,
                            local_restart, trunc_norm, max_full_size)
        
    def set_solver_txt(self, solver_txt):
        self.solver_txt = solver_txt
        
    def set_bc(self, bc):
        self.bc = bc
        
    def set_verb(self, verb_gen, verb_crs, verb_lss):
        self.verb_gen = verb_gen
        self.verb_crs = verb_crs
        self.verb_lss = verb_lss
        
    def set_sol0(self, sol0):
        self.sol0 = sol0
                    
    def set_tau(self, tau=1.E-10, eps_lss=1.E-10, tau_lss=None, tau_real=None):
        self.tau = tau
        self.eps_lss = eps_lss
        self.tau_lss = tau_lss
        if tau_real is not None:
            self.tau_real = tau_real
        else:
            self.tau_real = tau * 1.E-2
        
    def update_d(self, d=None):
        if d is not None:
            self.d = d
            self.n = 2**self.d
            self.h = self.L/self.n
        else:
            self.d, self.n, self.h = None, None, None
        
    def clean(self):
        self.GRD.clean()
        self.LSS.clean()
        self.t = dict.fromkeys(['cgen', 'mgen', 'sgen', 'soln', 'prep'])
        self.r = dict.fromkeys(['f', 'Bx', 'By', 'Bz', 'iBx', 'iBy', 'iBz',
                                'iKx', 'iKy', 'iKz', 'Kx', 'Ky', 'Kz',
                                'iqx', 'iqy', 'iqz', 'qx', 'qy', 'qz',
                                'Wx', 'Wy', 'Wz', 'Rx', 'Ry', 'Rz', 
                                'Hx', 'Hy', 'Hz', 'u_calc', 'u_real',
                                'ux_calc', 'uy_calc', 'uz_calc',
                                'ux_real', 'uy_real', 'uz_real',
                                'wx_calc', 'wy_calc', 'A', 'rhs'])
        
        self.u_real, self.u_real_ranks = None, None
        self.u_calc, self.u_calc_ranks = None, None
        self.u_err = None
        
        self.ux_real, self.uy_real, self.uz_real = None, None, None
        self.ux_calc, self.uy_calc, self.uz_calc = None, None, None
        self.ux_err , self.uy_err , self.uz_err  = None, None, None

        self.uu_real, self.uu_calc, self.uu_err  = None, None, None
        self.uf_real, self.uf_calc, self.uf_err  = None, None, None
   
    def present_info(self, full=False):
        self._present(compose_info(self, full))  
        
    def present_res_1s(self):
        self._present(compose_res_1s(self))             
        
    def present_res(self):
        self._present(compose_res(self))  

    def plot_res(self):
        plot(self)
        
    def copy(self, with_u_real =False, with_u_calc =False,
                   with_ux_real=False, with_ux_calc=False,
                   with_uy_real=False, with_uy_calc=False,
                   with_uz_real=False, with_uz_calc=False,
                   with_grd=False, with_sol0=False, with_funcs=False):     
        PDE = Pde()
        PDE.out_file      = self.out_file
        PDE.print_to_std  = self.print_to_std
        PDE.print_to_file = self.print_to_file
        PDE.set_dim(self.dim)
        PDE.set_mode(self.mode)
        PDE.model_num     = self.model_num
        PDE.params        = deepcopy(self.params)
        PDE.params_txt    = self.params_txt
        PDE.txt           = self.txt
        PDE.L             = self.L
        PDE.k_txt         = self.k_txt
        PDE.f_txt         = self.f_txt
        PDE.u_txt         = self.u_txt
        PDE.ux_txt        = self.ux_txt
        PDE.uy_txt        = self.uy_txt
        PDE.uz_txt        = self.uz_txt
        if with_grd: PDE.GRD = self.GRD.copy()
        PDE.LSS           = self.LSS.copy()
        PDE.set_solver_txt(self.solver_txt)
        PDE.set_bc(self.bc)
        PDE.set_verb(self.verb_gen, self.verb_crs, self.verb_lss)
        if with_sol0 and self.sol0: PDE.set_sol0(self.sol0.copy())
        PDE.set_tau(self.tau, self.eps_lss, self.tau_lss, self.tau_real)
        PDE.update_d(self.d)
        PDE.t = deepcopy(self.t)
        PDE.r = deepcopy(self.r)    
        if with_u_real and self.u_real is not None: PDE.u_real = self.u_real.copy()
        PDE.u_real_ranks = deepcopy(self.u_real_ranks)
        if with_u_calc and self.u_calc is not None: PDE.u_calc = self.u_calc.copy()
        PDE.u_calc_ranks = deepcopy(self.u_calc_ranks)
        PDE.u_err = self.u_err      
        if with_ux_real and self.ux_real is not None: PDE.ux_real = self.ux_real.copy()
        if with_uy_real and self.uy_real is not None: PDE.uy_real = self.uy_real.copy()
        if with_uz_real and self.uz_real is not None: PDE.uz_real = self.uz_real.copy()
        if with_ux_calc and self.ux_calc is not None: PDE.ux_calc = self.ux_calc.copy()
        if with_uy_calc and self.uy_calc is not None: PDE.uy_calc = self.uy_calc.copy()
        if with_uz_calc and self.uz_calc is not None: PDE.uz_calc = self.uz_calc.copy()
        PDE.ux_err  = self.ux_err
        PDE.uy_err  = self.uy_err
        PDE.uz_err  = self.uz_err
        PDE.uu_real = self.uu_real
        PDE.uu_calc = self.uu_calc
        PDE.uu_err  = self.uu_err
        PDE.uf_real = self.uf_real
        PDE.uf_calc = self.uf_calc
        PDE.uf_err  = self.uf_err
        return PDE
          
#        PDE_tmp = deepcopy(self)
#        if not with_u_real:
#            PDE_tmp.u_real = None
#        if not with_u_calc:
#            PDE_tmp.u_calc = None
#        if not with_ux_calc:
#            PDE_tmp.ux_calc = None
#        if not with_ux_real:
#            PDE_tmp.ux_real = None
#        if not with_uy_calc:
#            PDE_tmp.uy_calc = None
#        if not with_uy_real:
#            PDE_tmp.uy_real = None
#        if not with_uz_calc:
#            PDE_tmp.uz_calc = None
#        if not with_uz_real:
#            PDE_tmp.uz_real = None
#        if not with_funcs:
#            PDE_tmp.sol0 = None
#            PDE_tmp.f    = None
#            PDE_tmp.k    = None
#            PDE_tmp.u    = None
#            PDE_tmp.ux   = None
#            PDE_tmp.uy   = None
#            PDE_tmp.uz   = None    
#        return PDE_tmp