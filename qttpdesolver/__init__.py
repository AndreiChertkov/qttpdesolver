__version__ = '0.1'


from .tensor_wrapper import MODE_NP, MODE_TT, MODE_SP, Vector, Matrix, Func
from .pde import Pde
from .solvers import SOLVER_FS, SOLVER_FD
from .solvers import BC_HD, BC_PR
from .solvers import create_solver
from .solve import auto_solve
from .multirun import MultiSolve, MultiRes
