# -*- coding: utf-8 -*-
import os

from qttpdesolver import MODE_NP, MODE_TT, MODE_SP, SOLVER_FS, SOLVER_FD
from table import Table

class MultiRes(object):
    '''
        Present results of MultiSolve calculations.  
    '''
    
    def __init__(self, MS, tfolder=None, pfolder=None, save_res=True, verb=True):
        self.reps      = MS.reps
        self.jobs      = MS.jobs
        self.jobs_todo = MS.jobs_todo
        self.pde_lists = MS.pde_lists
        self.d_lists   = MS.d_lists
        self.tfolder   = tfolder
        self.pfolder   = pfolder
        self.save_res  = save_res
        self.verb      = verb
        
        if len(self.jobs_todo)>0:
            print 'Warning: there are unfinished jobs:'
            for i in self.jobs_todo:
                print '-> %s [%s]'%self.jobs[i]
            print 'You should finish them or remove before using MultiRes.'
            
        self.PDE = self.pde_lists[self.jobs[0][0]][self.jobs[0][1]][0].copy()
        self.PDE.update_d(None); self.PDE.solver, self.PDE.mode = None, None
   
    def present_model(self):
        self.PDE.present_model()
        
    def present_info(self):
        self.PDE.present_info(full=True)
        
    def present_res_1s(self, jobs=None, ds=None):
        if jobs and not isinstance(jobs, list):
            jobs = [jobs]
        for solver, mode in self.jobs:
            if jobs and not (solver, mode) in jobs:
                continue
            for i, d in enumerate(self.d_lists[solver][mode]):
                if ds and not d in ds:
                    continue
                self.pde_lists[solver][mode][i].present_res_1s()
    
    def tbl_params(self, lbl='', cpt='', pos='t', tfile=None, bold_hrow=True):
        hrow = ['Parameter', 'Value', 'Parameter', 'Value']
        rows, fmts = [], []
        
        row, fmt = [], []
        row.extend(['Round/cross acc.', self.PDE.tau])
        fmt.extend(['%-s', '%-8.2e'])
        row.extend(['System solver acc.', self.PDE.eps_lss])
        fmt.extend(['%-s', '%-8.2e'])
        rows.append(row); fmts.append(fmt)
        
        row, fmt = [], []      
        row.extend(['Round/cross acc. for real sol.', self.PDE.tau_real])
        fmt.extend(['%-s', '%-8.2e'])
        if self.PDE.tau_lss is not None:
            row.extend(['Round acc. for system sol.', self.PDE.tau_lss])
            fmt.extend(['%-s', '%-8.2e'])
        else:
            row.extend(['Round acc. for system sol.', '= solver residual'])
            fmt.extend(['%-s', '%-s'])
        rows.append(row); fmts.append(fmt)
        
        row, fmt = [], []      
        row.extend(['AMEn sweeps max', self.PDE.LSS.nswp])
        fmt.extend(['%-s', '%-2d'])
        row.extend(['AMEn kickrank', self.PDE.LSS.kickrank])
        fmt.extend(['%-s', '%-1d'])
        rows.append(row); fmts.append(fmt)
        
        row, fmt = [], []      
        row.extend(['AMEn max full size', self.PDE.LSS.max_full_size])
        fmt.extend(['%-s', '%-3d'])
        row.extend(['AMEn local restart', self.PDE.LSS.local_restart])
        fmt.extend(['%-s', '%-3d'])
        rows.append(row); fmts.append(fmt)
        
        row, fmt = [], []      
        row.extend(['AMEn local iters', self.PDE.LSS.local_iters])
        fmt.extend(['%-s', '%-1d'])
        row.extend(['AMEn initial guess', '= rhs'])
        fmt.extend(['%-s', '%-s'])
        rows.append(row); fmts.append(fmt)
            
        TBL = Table(hrow, rows, fmts, lbl, cpt, bold_hrow, False, pos, elem2tex=False)
        TBL.save(self._tfile(tfile))
        TBL.present(rem=[], verb=self.verb)
    
    def tbl_eranks(self, solver, mode=MODE_TT, ds=None, lbl='', cpt='', pos='t', 
                   tfile=None, hrow_rem=None, bold_hrow=True, bold_hcol=True):
        if ds is None:
            ds = self.d_lists[solver][mode]
        hrow = ['d']
        if solver == SOLVER_FD:
            hrow.extend([ 'Kx',  'Ky',  'Kz'][:self.PDE.dim])
        if solver == SOLVER_FS:
            hrow.extend(['iKx', 'iKy', 'iKz'][:self.PDE.dim])
            hrow.extend([ 'qx',  'qy',  'qz'][:self.PDE.dim])
            hrow.extend([ 'Wx',  'Wy',  'Wz'][:self.PDE.dim])
            hrow.extend([ 'Hx',  'Hy',  'Hz'][:self.PDE.dim])
        hrow.extend(['A', 'f', 'u_calc'])
        hrow.extend(['ux_calc', 'uy_calc', 'uz_calc'][:self.PDE.dim])
        if hrow_rem:
            for el in hrow_rem:
                try: hrow.remove(el)
                except: pass
        fmts = ['%2d']+['%5.1f']*(len(hrow)-1)
        rows = []
        for PDE in self.pde_lists[solver][mode]:
            if not PDE.d in ds:
                continue
            rows.append([PDE.d] + [PDE.r[elem] for elem in hrow[1:]])
        TBL = Table(hrow, rows, fmts, lbl, cpt, bold_hrow, bold_hcol, pos)
        TBL.save(self._tfile(tfile))
        TBL.present(rem=['$', '\qttvect', '\qttmatr'], verb=self.verb)
        
    def tbl_result(self, solver, mode=MODE_TT, ds=None, lbl='', cpt='', pos='t', 
                   tfile=None, hrow_rem=None, bold_hrow=True, bold_hcol=True):
        if ds is None:
            ds = self.d_lists[solver][mode]
        PDE0 = self.pde_lists[solver][mode][0]
        hrow = ['d', 'T (s.)']
        fmts = ['%2d', '%-6.1f']
        if PDE0.LSS.iters>0:
            hrow.append('I')
            fmts.append('%2d')
        if PDE0.LSS.max_res is not None:
            hrow.append('AMEn res')
            fmts.append('%-8.2e')
        if PDE0.uf_err is not None:
            hrow.append('Err ext. (u,f)')
            fmts.append('%-8.2e')
        if PDE0.u_err is not None:
            hrow.append('Err u')
            fmts.append('%-8.2e')
        if PDE0.ux_err is not None:
            hrow.append('Err $u_x$')
            fmts.append('%-8.2e')
        if PDE0.uy_err is not None:
            hrow.append('Err $u_y$')
            fmts.append('%-8.2e')
        if PDE0.uz_err is not None:
            hrow.append('Err $u_z$')
            fmts.append('%-8.2e')
        if PDE0.r['u_calc'] is not None:
            hrow.append('Erank u')
            fmts.append('%-5.1f')
                
        rows = []
        for PDE in self.pde_lists[solver][mode]:
            if not PDE.d in ds:
                continue
            row = [PDE.d, PDE.t_full]
            if PDE0.LSS.iters>0:
                row.append(PDE.LSS.iters)
            if PDE0.LSS.max_res is not None:
                row.append(PDE.LSS.max_res)
            if PDE0.uf_err is not None:
                row.append(PDE.uf_err)
            if PDE0.u_err is not None:
                row.append(PDE.u_err)
            if PDE0.ux_err is not None:
                row.append(PDE.ux_err)
            if PDE0.uy_err is not None:
                row.append(PDE.uy_err)
            if PDE0.uz_err is not None:
                row.append(PDE.uz_err)
            if PDE0.r['u_calc'] is not None:
                row.append(PDE.r['u_calc'])
            rows.append(row)
            
        if hrow_rem:
            for el in hrow_rem:
                try:
                    i = hrow.index(el)
                    del hrow[i]
                    del fmts[i]
                    for j in range(len(rows)):
                        del rows[j][i]
                except:
                    pass
        TBL = Table(hrow, rows, fmts, lbl, cpt, bold_hrow, bold_hcol, pos, elem2tex=False)
        TBL.save(self._tfile(tfile))
        TBL.present(rem=['$'], verb=self.verb)

    def _tfile(self, fname):
        if fname and not fname.endswith('.tex'):
            fname+= '.tex'
        if self.save_res and self.tfolder and fname:
            return os.path.join(self.tfolder, fname)
            
    def _pfile(self, fname):
        if self.save_res and self.pfolder and fname:
            return os.path.join(self.pfolder, fname)