# -*- coding: utf-8 -*-
import time
import datetime
import cPickle

from qttpdesolver import auto_solve

class MultiSolve(object):
    '''
        Solve given PDE for different grid factor d values by a list of given
        solvers and modes.
        
                 Global parameters [and default values]
                 
        PDE       - [None] instance of the Pde class with updated calculation 
                    parameters (except solver, mode, d, out_file)
                    * is set by __init__ function
        name      - ['calculation'] name of the calculations (is used as name
                    for files with results)
                    * is set by __init__ function
        folder    - ['./res/'] folder, where results should be saved
                    * is set by __init__ function 
        verb      - [False] if is True, then calculation details will be printed
                    * is set by __init__ function
        reps      - [1] is the number of repeats of each calculation (for time
                    averaging)
                    * is set by __init__ function
        jobs      - [[]] is a list of tuples (solver, mode) for calculations
                    * is set by add_job function
        d_lists   - [{}] is a dict that contain lists [solver][mode] of d values
                    for the corresponding solver and mode
                    * is set by add_job function
        pde_lists - [{}] is a dict that contain lists [solver][mode] of solved
                    PDEs for the corresponding solver and mode
                    * is set by add_job function     
    '''
    
    def __init__(self, PDE=None, name='calculation', folder='./res/', 
                 reps=1, verb=False, load=False):
        self.verb = verb
        self.reps = reps
        self.set_pde(PDE)
        self.set_name(name)
        self.set_folder(folder)
        self.jobs = []
        self.jobs_todo = []
        self.d_lists = {}
        self.pde_lists = {}
        if load:
            self.load()
            
    def set_pde(self, PDE=None):
        self.PDE = PDE
        
    def set_name(self, name='calculation'):
        self.name = name
        
    def set_folder(self, folder='./res/'):
        self.folder = folder
        
    def add_job(self, solver, mode, d_list):
        if self.d_lists.get(solver) is None:
            self.d_lists[solver] = {}
        self.d_lists[solver][mode] = d_list
        if self.pde_lists.get(solver) is None:
            self.pde_lists[solver] = {}
        self.pde_lists[solver][mode] = []
        self.jobs.append((solver, mode))
        self.jobs_todo.append(len(self.jobs)-1)
        
    def prepare(self):
        self.out_file = self.folder+'log_'+self.name
        self.par_file = self.folder+'par_'+self.name
        self.res_file = self.folder+'res_'+self.name
        if not self.PDE:
            return
        self.PDE.out_file = self.out_file
        if self.PDE.print_to_file:        
            f = open(self.out_file+'.txt', "w")
            f.writelines('---------------------------------------------------------\n')
            f.writelines('Computation. %s \n'%(datetime.datetime.now()))
            f.close()
        if self.verb:
            print '----------------------------------------------------------------'
            print '          Script for auto-computations by qttpdesolver          '
            print
            print '          Used PDE model:'
            self.PDE.present_model()
            print '          Used parameters:'
            self.PDE.present_info(full=True)
            print '----------------------------------------------------------------'
                
    def run(self, only_todo=False):
        _time = time.time()
        for i, [solver, mode] in enumerate(self.jobs):
            if only_todo and not i in self.jobs_todo:
                continue
            self.PDE.set_solver_txt(solver)
            self.PDE.set_mode(mode)
            for d in self.d_lists[solver][mode]:
                self._run_one(solver, mode, d)
                self.pde_lists[solver][mode].append(self.PDE.copy())
            self.jobs_todo.remove(i)    
            self.save()
        if self.verb:
            print '---------------- Total time of script work: %-8.2f'%(time.time()-_time)
          
    def _run_one(self, solver, mode, d):
        self.PDE.update_d(d)          
        t = dict.fromkeys(self.PDE.t.keys(), 0.)
        for i in range(self.reps):
            if self.verb and self.reps>1:
                if i==0:
                    print '-->  / ',
                else:
                    print 'r %2d / '%(i+1),
            auto_solve(self.PDE)
            for key in self.PDE.t.keys():
                t[key]  += self.PDE.t[key]
        for key in self.PDE.t.keys():
            self.PDE.t[key] = t[key]/self.reps
        if self.verb and self.reps>1:
            print 'Average   T=%-6.3f'%self.PDE.t_full

    def save(self):
        self._save_par()
        f_path = self.res_file
        if not f_path.endswith('.p'):
            f_path+= '.p'
        cPickle.dump([self.d_lists, self.pde_lists], open(f_path, "wb"), -1)
        
    def load(self):
        self.prepare()
        self._load_par()
        f_path = self.res_file
        if not f_path.endswith('.p'):
            f_path+= '.p'
        self.d_lists, self.pde_lists = cPickle.load(open(f_path, "rb"))
        
    def _save_par(self):
        f_path = self.par_file
        if not f_path.endswith('.p'):
            f_path+= '.p'
        pars = []
        pars.append(self.reps)
        pars.append(self.jobs)
        pars.append(self.jobs_todo)
        cPickle.dump(pars, open(f_path, "wb"), -1)
        
    def _load_par(self):
        f_path = self.par_file
        if not f_path.endswith('.p'):
            f_path+= '.p'
        pars = cPickle.load(open(f_path, "rb"))
        self.reps      = pars[0]
        self.jobs      = pars[1]
        self.jobs_todo = pars[2]

        
        
        
#from utils.transform2mesh import transform2coarser       
#                if d in usave_dlist:
#                    u = self.PDE.u_calc.x
#                    if usave_dconv_list is None:
#                        usave_dconv_list = []
#                    if not d in usave_dconv_list:
#                        if d > usave_dconv:
# 	                       u = transform2coarser(u, self.PDE.dim, reps=d-usave_dconv)
#                    if mode==MODE_TT:
#                        u = u.full().flatten('F')
#                    u = u.reshape(tuple([int(u.size**(1./self.PDE.dim))]*self.PDE.dim), order='F')
#                    cPickle.dump(u, open(res_file+'_u_calc_d%d_to_d%d_%s_%s.p'%\
#                                         (d, usave_dconv, solver, mode) , "wb"))   