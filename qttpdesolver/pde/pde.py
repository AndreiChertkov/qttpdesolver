# -*- coding: utf-8 -*-
from ..utils.grid import Grid
from ..tensor_wrapper import MODE_NP, MODE_TT, LinSystSolver
from ..solvers.solver import BC_HD
from ..solvers.solver import SOLVER_VECTORS, SOLVER_MATRICES, SOLVER_TIMES
from .models import set_model, get_models_txt
from .out import compose_model, compose_info, compose_res_1s, compose_res
from .plots import plot
from .clone import clone

PDE_FUNCS = ['Kx', 'Ky', 'Kz', 'Kxy', 'Kyx', 'F', 'U', 'Ux', 'Uy', 'Uz']

class Pde(object):
    '''
    A container for PDE of the form
        -div(k grad u) = f.
    All PDE-model parameteres may be set in auto mode by model name
    (function set_model). Call present_models to see all available models.
    * Function clean should be called before each calculation!

    -------------------------------------
    Model parameters [and default values]
    * function clean_model set default values for this variables

    model_name       - [None] model (benchmark) unique name
    txt              - [None] representation of equation by a string
    dim              - [None] dimension of the PDE
    L                - [1.  ] size of the spatial domain
    Kx               - [None] PDE parameter k_{x} (Func instance)
    Ky               - [None] PDE parameter k_{y} (Func instance)
    Kz               - [None] PDE parameter k_{z} (Func instance)
    Kxy              - [None] PDE parameter k_{xy} (Func instance)
    Kyx              - [None] PDE parameter k_{yx} (Func instance)
    F                - [None] PDE rhs f (Func instance)
    U                - [None] PDE exact solution (Func instance)
    Ux               - [None] PDE exact x-derivative of solution (Func instance)
    Uy               - [None] PDE exact y-derivative of solution (Func instance)
    Uz               - [None] PDE exact z-derivative of solution (Func instance)
    params           - [None] list of params for k, f, ... functions.
                       * all functions must have the same input parameters
    params_names     - [None] list of parameters names
    params_forms     - [None] list of parameters formats for pretty output

    --------------------------------------
    Solver parameters [and default values]

    mode             - [MODE_NP] = 'np' (or MODE_NP) for numpy format,
                                 = 'tt' (or MODE_TT) for tensor train format,
                                 = 'sp' (or MODE_SP) for scipy sparse format
    out_file         - [None] path to file to print (in 'a'-append mode)
                       * '.txt' will be appended automatically if needed
    print_to_std     - [True]  if True, present_* functions print to std
    print_to_file    - [False] if True, present_* functions print to out_file
    GRD              - [Grid class instance] spatial grid
                       * see utils.grid
                       * is set by set_grd function
    LSS              - [LinSystSolver class instance] solver for linear systems
                       * see tensor_wrapper.lin_syst_solver
                       * is set by set_lss function and then may be configured
                         by set_lss_params function
    d                - [None] scale of grid size
    n                - [None] 1d grid size (=2^d)
    h                - [None] grid step (=L/n)
                       * d, n, h are set by update_d function
                       * Domain size may be only L=1 in current version
    solver_name      - [None] name of the used PDE solver
                       * is set by set_solver_name function
                       * available options:
                            'fs' (SOLVER_FS) - finite sum solver,
                            'fd' (SOLVER_FD) - finite difference solver
    bc               - [BC_HD] is the type of boundary conditions
                       * available options:
                            'hd' (BC_HD) - homogeneous Dirichlet,
                            'pr' (BC_PR) - periodic with zero solution integral
                       * is set by set_bc function
    verb_gen         - [False] is the verbosity of general output (times, etc.)
    verb_crs         - [False] is the verbosity of output for cross iterations
    verb_lss         - [False] is the verbosity of output for LSS iterations
                       * verb_* are set by set_verb function
                       * verb_* are for Solver's output verbosity only
    sol0             - [None] initial guess for solution of linear system:
                            for solver_fs - is a guess for
                                mu_x (2d) or [mu_x, mu_y] (3d),
                            for solver_fd - is a guess for
                                u_calc
                       * is set by set_sol0 function
    tau              - [None] tt-accuracy for rounding and cross-function
    eps_lss          - [None] accuracy for linear system solver
    tau_lss          - [None] tt-accuracy for rounding of result of linear
                       system solution
                       * if is None, then the real accuracy (max_res) is used
    tau_real         - [None] tt-accuracy for construction of real solution
                       * if is None, then it will be selected as tau * 1.E-2
                       * tau_* and eps_lss are set by set_tau function

    ----------------------------------------
    Calculation results [and default values]
    * are set by solver in auto mode after calculation
    * function clean set default values for this variables

    u_calc          - [None] calculated solution of the PDE
    u_calc_ranks    - [None] TT-ranks of u_calc
    u_real          - [None] exact (analitical) solution of the PDE
    u_real_ranks    - [None] TT-ranks of u_real
    u_err           - [None] is the norm of the u calculation's error

    ux_calc         - [None] calculated derivative du/dx of the solution
    ux_real         - [None] exact (analitical) derivative du/dx of the solution
    ux_err          - [None] is the norm of the du/dx calculation's error
    ... the same for uy and uz ...

    uu_calc         - [None] product (u_calc, u_calc)
    uu_real         - [None] product (u_real, u_real)
    uu_err          - [None] error of the calculated value of uu_calc

    uf_calc         - [None] product (u_calc, f)
    uf_real         - [None] product (u_real, f)
    uf_err          - [None] error of the calculated value of uf_calc

    t_full          - [None] full time of the Solver work
    r               - [dict of None] is dict with eranks of the main arrays
    t               - [dict of None] is dict with times of solution steps

    ---------
    Functions

    set_print       - set print options
    set_dim         - set spatial dimension
    set_mode        - set mode of calculations
    set_model       - set PDE model via unique name (models module will be used)
    set_params      - set a list of parameters for kx, ky, f, u, ... functions
    clean_model     - delete current model (set all Model parameters to None)
                      * is called from set_model automatically
    set_grd         - set spatial grid class instance
                      * Default grid is set automatically
    set_lss         - set linear system solver class instance
                      * Default linear system solver is set automatically
    set_lss_params  - set parameters for linear system solver
                      * Default parameters are set automatically
    set_solver_name - set solver_name
    set_bc          - set bc (boundary conditions)
    set_verb        - set verb_gen, verb_crs and verb_lss
    set_sol0        - set sol0
    set_tau         - set all TT-accuracies
    update_d        - set a given value of d, and then recalculate n and h
    clean           - remove all calculation results (set None values)
    present_models  - present all available PDE models
    present_model   - present current (selected) PDE model (kx, ky, f, u, etc.)
    present_info    - present parameters of the calculation in full text mode
    present_res_1s  - present calculation results in one string mode
    present_res     - present calculation results in full text mode
    plot_res        - plot u_real, u_calc and its derivatives
                      * u_real and/or u_calc may be None
                      * if MODE_TT or MODE_SP is used, then transformation to MODE_NP will be performed, so call this function only for moderate d values
    copy            - copy parameters and calc results to a new Pde instanсe
                      * functions and arrays are not copied by default
    '''

    def __init__(self, txt=None, dim=None, bc=BC_HD, mode=MODE_NP, GRD=None, LSS=None):
        self.set_model()
        self.set_print()
        self.set_dim(dim)
        self.set_mode(mode)
        self.set_grd(GRD)
        self.set_lss(LSS)
        if not LSS:
            self.set_lss_params()
        self.set_txt(txt)
        self.set_solver_name(None)
        self.set_bc(bc)
        self.set_verb(False, False, False)
        self.set_sol0(None)
        self.set_tau()
        self.update_d()
        self.clean()

    @property
    def t_full(self):
        ops = ['cgen', 'mgen', 'sgen', 'soln']
        return sum([self.t[op] or 0 for op in ops]) or None

    def set_print(self, to_std=True, to_file=False, out_file=None):
        self.print_to_std = to_std
        self.print_to_file = to_file
        self.out_file = out_file
        if self.out_file and not self.out_file.endswith('.txt'):
            self.out_file+= '.txt'

    def set_dim(self, dim):
        self.dim = dim

    def set_mode(self, mode):
        self.mode = mode

    def set_model(self, model_name=None):
        self.clean_model()
        if not model_name:
            return
        self.model_name = model_name
        set_model(self, model_name)

    def set_params(self, params, params_names=None, params_forms=None):
        if not isinstance(params, list):
            raise ValueError('Input parameters should be in list.')
        if self.params is not None:
            if len(params) != len(self.params):
                raise ValueError('Incorrect number of parameters.')
        self.params = params
        self.params_names = params_names or self.params_names
        self.params_forms = params_forms or self.params_forms
        for func in PDE_FUNCS:
            f = getattr(self, func)
            if not f:
                continue
            f.set_args(**{n: v for n, v in zip(self.params_names, self.params)})

    def set_grd(self, GRD=None):
        self.GRD = GRD or Grid()

    def set_lss(self, LSS=None):
        self.LSS = LSS or LinSystSolver()

    def set_lss_params(self, nswp=20, kickrank=4, local_prec='n', local_iters=2,
                      local_restart=40, trunc_norm=1, max_full_size=50):
        self.LSS.set_params(nswp, kickrank, local_prec, local_iters,
                            local_restart, trunc_norm, max_full_size)

    def set_txt(self, txt):
        self.txt = txt

    def set_solver_name(self, solver_name):
        self.solver_name = solver_name

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
        self.tau_real = tau_real or tau * 1.E-2

    def update_d(self, d=None):
        self.d = d
        self.n = 2**self.d if d else None
        self.h = self.L/self.n if d else None

    def clean_model(self):
        self.model_name, self.txt, self.dim, self.L = None, None, None, 1.
        self.params, self.params_names, self.params_forms = None, None, None
        for func in PDE_FUNCS:
            setattr(self, func, None)

    def clean(self):
        self.GRD.clean()
        self.LSS.clean()
        self.t = dict.fromkeys(SOLVER_TIMES)
        self.r = dict.fromkeys(SOLVER_VECTORS + SOLVER_MATRICES)

        self.u_real, self.u_real_ranks = None, None
        self.u_calc, self.u_calc_ranks = None, None
        self.u_err = None

        self.ux_real, self.uy_real, self.uz_real = None, None, None
        self.ux_calc, self.uy_calc, self.uz_calc = None, None, None
        self.ux_err , self.uy_err , self.uz_err  = None, None, None

        self.uu_real, self.uu_calc, self.uu_err  = None, None, None
        self.uf_real, self.uf_calc, self.uf_err  = None, None, None

    def copy(self, with_sol=False, with_sol0=False, with_der=False):
        return clone(self, with_sol, with_sol0, with_der)

    def present_models(self):
        self._present('The following models are available:\n'+get_models_txt())

    def present_model(self):
        self._present(compose_model(self, PDE_FUNCS))

    def present_info(self, full=False):
        self._present(compose_info(self, full))

    def present_res_1s(self):
        self._present(compose_res_1s(self))

    def present_res(self):
        self._present(compose_res(self))

    def plot_res(self):
        plot(self)

    def _present(self, s):
        ''' Print given string to std and/or to file according to settings.  '''
        if self.print_to_std:
            print s
        if not self.print_to_file or not self.out_file:
            return
        with open(self.out_file, "a+") as f:
            f.writelines(s if s.endswith('\n') else s+'\n')
            f.close()
