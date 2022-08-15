# -*- coding: utf-8 -*-
import os
import sys
import tempfile

class Capture(object):
    ''' Example for class Capture usage:
            c = Capture()
            c.start_capture()
            os.system('echo 10')
            print('20')
            c.stop_capture()
            print c.out
            print len(c.out)
            # >>> ['10\n', '20\n']
            # >>> 2
    '''

    def __init__(self):
        super(Capture, self).__init__()
        self._org = None    # Original stdout stream
        self._dup = None    # Original system stdout descriptor
        self._tmp = None    # Temporary file to write stdout to
        self.clean()

    def clean(self):
        self._out = []

    def start_capture(self):
        self._org = sys.stdout
        sys.stdout = sys.__stdout__
        fdout = sys.stdout.fileno()
        self._tmp = tempfile.TemporaryFile()
        self._dup = None
        if fdout >= 0:
            self._dup = os.dup(fdout)
            os.dup2(self._tmp.fileno(), fdout)

    def stop_capture(self):
        sys.stdout.flush()
        if self._dup is not None:
            os.dup2(self._dup, sys.stdout.fileno())
            os.close(self._dup)
        sys.stdout = self._org
        self._tmp.seek(0)
        self._out = self._tmp.readlines()
        self._tmp.close()

    def present(self, verb):
        if not verb:
            return
        for s in self._out:
            print s,

class CaptureAmen(Capture):
    ''' Example for class CaptureAmen usage:
            import tt
            from tt.amen import amen_solve
            c = CaptureAmen()
            c.start_capture()
            d = 5; n = 2**d
            A = tt.eye(2, d)
            rhs = tt.ones(2, d)
            u0 = tt.rand(rhs.n, r=3)
            tau = 1.E-8
            sol = amen_solve(A, rhs, u0, tau).round(tau)
            c.stop_capture()
            print c.iters
            # >>> 2
    '''

    def __init__(self):
        super(CaptureAmen, self).__init__()

    @property
    def iters(self):
        count = 0
        for x in self._out:
            if 'amen_solve: swp' in x:
                count+= 1
        return count or None

    @property
    def max_dx(self):
        try:
            return float(self._out[-1].split('max_dx= ')[1].split(',')[0])
        except:
            return None

    @property
    def max_res(self):
        try:
            return float(self._out[-1].split('max_res= ')[1].split(',')[0])
        except:
            return None

    @property
    def max_rank(self):
        try:
            return int(self._out[-1].split('max_rank=')[1])
        except:
            return None

class CaptureCross(Capture):

    def __init__(self):
        super(CaptureCross, self).__init__()

    @property
    def sweep(self):
        if not len(self._out) > 0:
            return None
        s = self._out[-1]
        if s.find('sweep ') >= 0: # multifuncrs
            return int(s.split('sweep ')[1].split('{')[0])
        if s.find('swp: ') >= 0: # rect cross
            return int(s.split('swp: ')[1].split('/')[0]) + 1

    @property
    def sweep_sub(self):
        if not len(self._out) > 0:
            return None
        s = self._out[-1]
        if s.find('sweep ') >= 0: # multifuncrs
            return int(s.split('}')[0].split('{')[-1])

    @property
    def err_rel(self):
        if not len(self._out) > 0:
            return None
        s = self._out[-1]
        if s.find('er_rel = ') >= 0: # rect cross
            return float(s.split('er_rel = ')[1].split(' ')[0])

    @property
    def err_abs(self):
        if not len(self._out) > 0:
            return None
        s = self._out[-1]
        if s.find('er_abs = ') >= 0: # rect cross
            return float(s.split('er_abs = ')[1].split(' ')[0])

    @property
    def err_dy(self):
        if not len(self._out) > 0:
            return None
        s = self._out[-1]
        if s.find('max_dy: ') >= 0:
            return float(s.split('max_dy: ')[1].split(',')[0])

    @property
    def erank(self):
        if not len(self._out) > 0:
            return None
        s = self._out[-1]
        if s.find('erank: ') >= 0: # multifuncrs
            return float(s.split('erank: ')[1])
        if s.find('erank = ') >= 0: # rect cross
            return float(s.split('erank = ')[1].split(' ')[0])

    @property
    def evals(self):
        if not len(self._out) > 0:
            return None
        s = self._out[-1]
        if s.find('fun_eval: ') >= 0: # rect cross
            return int(s.split('fun_eval: ')[1])
