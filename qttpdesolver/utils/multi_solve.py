# -*- coding: utf-8 -*-
import time
import datetime
import cPickle

from qttpdesolver import MODE_TT, auto_solve

class MultiSolve(object):
    '''
        Solve given PDE for different grid factor d values by a list of given
        solvers and modes, and save results to disk.
        
                  Global parameters [and default values]
                 
        PDE       - [None] instance of the Pde class with updated calculation 
                    parameters (except solver, mode, d, out_file)
                    * is set by __init__ function
        name      - ['calculation'] name of the calculations
                    * is used as name for files with results
                    * is set by __init__ function
        folder    - ['./res/'] folder, where results should be saved
                    * is set by __init__ function 
        reps      - [1] is the number of repeats of each calculation
                    * if >16 then time averaging will be performed
                    * is set by __init__ function
        verb      - [False] if is True, then calculation details will be printed
                    * is set by __init__ function
                    
                  Calculation results [and default values]
                    
        jobs      - [[]] is a list of tuples (solver, mode) for calculations
                    * is set by add_job function
        jobs_todo - [[]] is a list of numbers of jobs that are not finished
                    * is set by add_job function
        d_lists   - [{}] is a dict that contain lists [solver][mode] 
                    of d values for the corresponding solver and mode
                    * is set by add_job function
        pde_lists - [{}] is a dict that contain lists [solver][mode] of solved
                    PDEs for the corresponding solver and mode
                    * is set by run function     
    '''
    
    def __init__(self, PDE=None, name='calculation', folder='./res/', 
                 reps=1, verb=False, load=False):
        self.PDE = PDE
        self.name = name
        self.folder = folder
        self.reps = reps
        self.verb = verb
        self.jobs = []
        self.jobs_todo = []
        self.d_lists = {}
        self.pde_lists = {}
        if load:
            self.load()

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
            with open(self.out_file+'.txt', "w") as f:
                f.writelines('---------------------------------------------------------\n')
                f.writelines('Computation. %s \n'%(datetime.datetime.now()))
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
        for i, (solver, mode) in enumerate(self.jobs):
            if only_todo and not i in self.jobs_todo:
                continue
            self.PDE.set_solver_txt(solver)
            self.PDE.set_mode(mode)
            for d in self.d_lists[solver][mode]:
                self.pde_lists[solver][mode].append(self._run_one(d))
            self.jobs_todo.remove(i)    
            self.save()
        if self.verb:
            print '\n---------------- Total time of script work: %-8.2f'%(time.time()-_time)
          
    def _run_one(self, d):
        PDEs = []
        for i in range(self.reps):
            if self.verb and self.reps>1:
                if i==0: print '-->  / ',
                else: print 'r %2d / '%(i+1),
            auto_solve(self.PDE, d=d)
            PDEs.append(self.PDE.copy())
        i0 = 0
        for i in range(1, self.reps):
            if self.PDE.mode == MODE_TT:
                if PDEs[i].LSS.iters > 0:
                    if PDEs[i0].LSS.iters > PDEs[i].LSS.iters:
                        i0 = i
                elif PDEs[i0].r['u_calc'] > PDEs[i].r['u_calc']:
                    i0 = i
        for i in range(self.reps):
            if i == i0: continue
            for key in self.PDE.t.keys():
                PDEs[i0].t[key] += PDEs[i].t[key]
        for key in self.PDE.t.keys():
            PDEs[i0].t[key] /= self.reps
        if self.verb and self.reps>1:
            print 'Average      T=%-6.3f'%PDEs[i0].t_full
        return PDEs[i0]
        
    def save(self):
        self._save_par()
        f_path = self.res_file
        if not f_path.endswith('.p'): f_path+= '.p'
        cPickle.dump([self.d_lists, self.pde_lists], open(f_path, "wb"), -1)
        
    def load(self):
        self.prepare()
        self._load_par()
        f_path = self.res_file
        if not f_path.endswith('.p'): f_path+= '.p'
        self.d_lists, self.pde_lists = cPickle.load(open(f_path, "rb"))
        
    def _save_par(self):
        f_path = self.par_file
        if not f_path.endswith('.p'): f_path+= '.p'
        cPickle.dump([self.reps, self.jobs, self.jobs_todo], open(f_path, "wb"), -1)
        
    def _load_par(self):
        f_path = self.par_file
        if not f_path.endswith('.p'): f_path+= '.p'
        self.reps, self.jobs, self.jobs_todo = cPickle.load(open(f_path, "rb")) 