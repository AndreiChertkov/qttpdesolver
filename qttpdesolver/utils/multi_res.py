# -*- coding: utf-8 -*-
import time
import datetime
import cPickle

from qttpdesolver import auto_solve

class MultiRes(object):
    '''
        Present results of MultiSolve calculations.  
    '''
    
    def __init__(self, MS):
        self.reps      = MS.reps
        self.jobs      = MS.jobs
        self.jobs_todo = MS.jobs_todo
        self.pde_lists = MS.pde_lists
        self.d_lists   = MS.d_lists

        if len(self.jobs_todo)>0:
            print 'Warning: there are unfinished jobs:'
            for i in self.jobs_todo:
                print '-> %s [%s]'%tuple(self.jobs[i])
            print 'You should finish them or remove before using MultiRes.'
            
        self.PDE = self.pde_lists[self.jobs[0][0]][self.jobs[0][1]][0].copy()
        self.PDE.update_d(None)
        self.PDE.solver, self.PDE.mode = None, None

    def present_model(self):
        self.PDE.present_model()
        
    def present_info(self):
        self.PDE.present_info(full=True)
        
    def present_res_1s(self, jobs=None, ds=None):  
        for solver, mode in self.jobs:
            if jobs and not [solver, mode] in jobs:
                continue
            for i, d in enumerate(self.d_lists[solver][mode]):
                if ds and not d in ds:
                    continue
                self.pde_lists[solver][mode][i].present_res_1s()
                
                
