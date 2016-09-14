# -*- coding: utf-8 -*-
from .pde_utils.txts import compose_res_1s, compose_res, compose_info
from .pde_utils.algss_pars import AlgssPar
from .pde_utils.times import Time
from .pde_utils.taus import Tau
from .pde_utils.copies import copy
from .pde_utils.plots import plot

from model_pde import ModelPde

class Pde(ModelPde):
    '''
    A container for PDE of the form -div(k grad u) = f
    (see parent class ModelPde for more details).
    
    * Function clean should be called before every calculation!
    
                 Global parameters
                 
    solver_txt - [None] name of the used PDE solver (is set by set_solver_txt)
                 * available options: 'solver_fs', 'solver_fd'                          
    d          - [None] scale of grid size
    n          - [None] 1d grid size (=2^d)
    h          - [None] grid step (=L/n)
                 * d, n, h are set by update_d func
    verb_gen   - [False] is the verbosity of general output (times and so on)
    verb_cross - [False] is the verbosity of output for cross iterations
    verb_amen  - [False] is the verbosity of output for amen iterations
                 * verb_* are set by set_verb function
                 * verb_* are for Solver's output verbosity only
    sol0       - [None] initial guess for solution of linear system:
                 for solver_fs - is a guess for mu_x (2d) or [mu_x,mu_y] (3d),
                 for solver_fd - is a guess for PDE solution u_calc
                 (is set by set_sol0 function)
    with_en    - [False] if True then energy will be calculated 
                 (is set by set_with_en function) 
    tau        - [dict of None] tt-accuracy for round-operation, 
                 cross-function and amen-function (is set by set_tau function)           
    algss_par  - [dict of None]  parameters for algebraic system solver
                 (is set by set_algss_par function)
                 * it also contain some results of system solving

                 Calculation results
                 * are set by solver in auto mode after calculation
                 * function clean set default values for all this variables
                 
    a_erank      - [None] TT-erank of matrix A in A x = rhs
    rhs_erank    - [None] TT-erank of vector rhs in A x = rhs
                   * Some additional eranks are also saved: id_erank, d_erank,
                     iq.erank, q_erank, f_erank, w_erank, r_erank, h_erank
                     (some of them may be None for some kinds of solvers,
                     otherwise, all of them, except f_erank,
                     are lists of dim-length)
    u_calc       - [None] calculated solution of the PDE
    u_calc_ranks - [None] ranks or u_real
    u_calc_erank - [None] TT-erank or u_real 
    u_real       - [None] exact (analitical) solution of the PDE
    u_real_ranks - [None] ranks or u_real 
    u_real_erank - [None] TT-erank or u_real     
    u_err        - [None] is the norm of the u calculation's error       
    ud_calc      - [None] calculated derivatives of the solution (list)
    ud_real      - [None] exact (analitical) derivatives of the solution (list)
    ud_err       - [None] is the norm of the du calculation's error (list)   
    en_calc      - [None] energy h^2 * sum( (K_i ud_calc[i], K_i) )
    en_real      - [None] energy h^2 * sum( (K_i ud_real[i], K_i) )
    en_err       - [None] error of the calculated energy
    uf_calc      - [None] product (u_calc, f)
    uf_real      - [None] product (u_real, f)
    uf_err       - [None] error of the calculated value of uf_calc
    uu_calc      - [None] product (u_calc, u_calc)
    uu_real      - [None] product (u_real, u_real)
    uu_err       - [None] error of the calculated value of uu_calc
    t_full       - [None] full time of the calculation 
    t            - [dict of None] is dictionary with times of calc. operations
    
                 Functions
                 
    set_solver_txt - set solver_txt
    update_d       - set a given value of d and recalculate n and h values
    set_verb       - verb_gen, verb_cross and verb_amen
    set_sol0       - set sol0
    set_with_en    - set with_en mode
    set_tau        - construct all tt-accuracies according current
                     understanding of dependencies by a few given values
                     * taus are saved in Tau class instance
    set_algss_par  - set parameters for algebraic system solver
                     * parameters are saved in AlgssPar class instance
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
    def __init__(self):
        ModelPde.__init__(self)
        self.set_solver_txt(None)
        self.update_d()
        self.set_verb(False, False, False)
        self.set_sol0(None)
        self.set_with_en(False)
        self.tau = Tau()
        self.algss_par = AlgssPar()
        self.t = Time()
        self.clean()

    def set_solver_txt(self, solver_txt):
        self.solver_txt = solver_txt
        
    def update_d(self, d=None):
        if d is not None:
            self.d = d
            self.n = 2**self.d
            self.h = self.L/self.n
        else:
            self.d, self.n, self.h = None, None, None
            
    def set_verb(self, verb_gen, verb_cross, verb_amen):
        self.verb_gen   = verb_gen
        self.verb_cross = verb_cross
        self.verb_amen  = verb_amen
        
    def set_sol0(self, sol0):
        self.sol0 = sol0
            
    def set_with_en(self, with_en):
        self.with_en = with_en
        
    def set_tau(self, tau_round=1.E-10, tau_cross=1.E-10, tau_amens=1.E-5):
        self.tau.set_by_main(tau_round, tau_cross, tau_amens)                      
        self.algss_par.set('tau', self.tau['solve']['round'])
        
    def set_algss_par(self, nswp=20, kickrank=4, local_prec='n', local_iters=2,
                      local_restart=40, trunc_norm=1, max_full_size=50,
                      tau_u_calc_from_algss=True):   
        self.algss_par.set('nswp', nswp)
        self.algss_par.set('kickrank', kickrank)
        self.algss_par.set('local_prec', local_prec)
        self.algss_par.set('local_iters', local_iters)
        self.algss_par.set('local_restart', local_restart)
        self.algss_par.set('trunc_norm', trunc_norm)
        self.algss_par.set('max_full_size', max_full_size)
        self.algss_par.set('tau', self.tau['solve']['round'])
        self.algss_par.set('tau_u_calc_from_algss', tau_u_calc_from_algss)

    def clean(self):
        self.algss_par.clean_out()
        self.t.clean()

        self.id_erank, self.d_erank   = None, None
        self.iq_erank, self.q_erank   = None, None
        self.f_erank,  self.w_erank   = None, None
        self.r_erank,  self.h_erank   = None, None
        self.a_erank,  self.rhs_erank = None, None
        
        self.u_err = None
        self.u_calc, self.u_calc_ranks, self.u_calc_erank = None, None, None
        self.u_real, self.u_real_ranks, self.u_real_erank = None, None, None
        self.ud_calc, self.ud_real, self.ud_err = None, None, None
        self.en_calc, self.en_real, self.en_err = None, None, None
        self.uf_real, self.uf_calc, self.uf_err = None, None, None
        self.uu_real, self.uu_calc, self.uu_err = None, None, None
        
    def present_info(self):
        self._present(compose_info(self))  
        
    def present_res_1s(self):
        self._present(compose_res_1s(self))             
        
    def present_res(self):
        self._present(compose_res(self))  

    def plot_res(self):
        plot(self)
        
    def copy(self, with_u_real=False, with_u_calc=False):
        return copy(self, with_u_real, with_u_calc)
        
    @property
    def t_full(self):
        return self.t.get_full()