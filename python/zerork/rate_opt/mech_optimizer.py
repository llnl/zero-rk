
import os
import time
import shutil
import tempfile
import functools
import numpy as np
import scipy.optimize as optimize
from multiprocessing import Pool
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import OrderedDict

from .bounds_check import BoundsCheck
from ..config import ZERORK_DEBUG
from ..util.rmtree_retry import rmtree_retry

def parallel_opt_fn(args):
    app, mf, thf, trf = args
    val = app.opt_fn(mf, thf, trf)
    return val, app

class mech_optimizer(object):
    def __init__(self, mech_file, therm_file, trans_file=None, nprocs=1):
        self.verbose = False
        self.prelines = []
        self.postlines = []
        self.reactions = []
        self.reaction_eqns = []
        self.opt_apps = []
        self.nprocs = nprocs
        self.duplicates = dict()
        self.opt_bounds = []
        self.param_fns = []
        self.x0 = []
        self.next_output = np.inf
        self.check_interval = np.inf
        self.params = []
        self.values = []
        self.param_log = None
        self.opt_rxn_idxs = set()
        self.mech_file = mech_file #N.B. we don't run with original mech at all
        self.therm_file = therm_file
        self.trans_file = trans_file
        self.output_directory = None
        self.prev_calls = 0
        self.parse(mech_file)
    def parse(self, mech_file):
        with open(mech_file,'r', encoding='utf8') as mf:
            self.lines = mf.readlines()
        reaction_section=False
        post_reaction_section=False
        for line in self.lines:
            comment = "!".join(line.split("!")[1:])
            linex = line.split("!")[0].strip()
            if 'REAC' == line.strip().upper()[0:4]:
                reaction_section=True
                self.prelines.append(line)
#                for token in line.strip().upper().split():
#                    if token in self.e_unit_conversion_map:
#                        self.e_unit_conversion = self.e_unit_conversion_map(token)
#                        if self.e_unit_conversion is None:
#                            raise ValueError("Unsupported energy units.")
                continue
            if reaction_section:
                if 'END' == line.strip().upper()[0:3]:
                    reaction_section=False
                    post_reaction_section=True
                    self.postlines.append(line)
                    continue
                if "=" in linex:
                    tokens=linex.split()
                    A, n, Ea = map(float,tokens[-3:])
                    eqn = ''.join(tokens[:-3])
                    eqn = eqn.replace('<=>','=')
                    eqn_orig = ' '.join(tokens[:-3])
                    rxn = {'eqn': eqn, 'A': A, 'n': n, 'Ea': Ea, 'aux': []}
                    rxn.update({'Aorig':A, 'norig': n, 'Eaorig':Ea, 'eqn_orig': eqn_orig})
                    rxn['comment'] = comment
                    rxn['aux_comments'] = []
                    rxn['line'] = line
                    if eqn in self.reaction_eqns:
                        idxs = [i for i, x in enumerate(self.reaction_eqns) if x == eqn]
                        self.duplicates[eqn] = idxs + [len(self.reactions)]
                    self.reaction_eqns.append(rxn['eqn'])
                    self.reactions.append(rxn)
                else:
                    self.reactions[-1]['aux'].append(line)
                    self.reactions[-1]['aux_comments'].append(comment)
                    if 'PLOG' == linex.upper()[0:4]:
                        plog_vals = list(map(float,linex.split("/")[1].split()))
                        if 'plog_orig' in self.reactions[-1]:
                            self.reactions[-1]['plog_orig'].append(list(plog_vals))
                            self.reactions[-1]['plog_mod'].append(list(plog_vals))
                        else:
                            self.reactions[-1]['plog_orig'] = [list(plog_vals)]
                            self.reactions[-1]['plog_mod'] = [list(plog_vals)]
            elif post_reaction_section:
                self.postlines.append(line)
            else:
                self.prelines.append(line)
    def mult_A_fn(self, rxn_idx_str):
        def fn(A_mult):
            for ridx in map(int, rxn_idx_str.split("-")):
                self.reactions[ridx]['A'] = self.reactions[ridx]['Aorig']*A_mult
                if 'plog_mod' in self.reactions[ridx]:
                    for lidx in range(len(self.reactions[ridx]['plog_mod'])):
                        self.reactions[ridx]['plog_mod'][lidx][1] = \
                              self.reactions[ridx]['plog_orig'][lidx][1]*A_mult
        return fn
    def add_Ea_fn(self, rxn_idx_str):
        def fn(Ea_delta):
            for ridx in map(int, rxn_idx_str.split("-")):
                self.reactions[ridx]['Ea'] = self.reactions[ridx]['Eaorig']+Ea_delta
                if 'plog_mod' in self.reactions[ridx]:
                    for lidx in range(len(self.reactions[ridx]['plog_mod'])):
                        self.reactions[ridx]['plog_mod'][lidx][3] = \
                              self.reactions[ridx]['plog_orig'][lidx][3]+Ea_delta
        return fn
    def add_n_fn(self, rxn_idx_str):
        def fn(n_delta):
            for ridx in map(int, rxn_idx_str.split("-")):
                self.reactions[ridx]['n'] = self.reactions[ridx]['norig']+n_delta
                if 'plog_mod' in self.reactions[ridx]:
                    for lidx in range(len(self.reactions[ridx]['plog_mod'])):
                        self.reactions[ridx]['plog_mod'][lidx][2] = \
                              self.reactions[ridx]['plog_orig'][lidx][2]+n_delta
        return fn
    def mult_n_fn(self, rxn_idx_str):
        def fn(n_mult):
            for ridx in map(int, rxn_idx_str.split("-")):
                self.reactions[ridx]['n'] = self.reactions[ridx]['norig']*n_mult
                if 'plog_mod' in self.reactions[ridx]:
                    for lidx in range(len(self.reactions[ridx]['plog_mod'])):
                        self.reactions[ridx]['plog_mod'][lidx][2] = \
                              self.reactions[ridx]['plog_orig'][lidx][2]*n_mult
        return fn
    def set_rxn_opt(self, rxn_eqn, A_mult=2, n_delta=None, n_mult=None, Ea_delta=None):
        rxn_eqn = ''.join(rxn_eqn.strip().split()) #Remove whitespace
        rxn_eqn = rxn_eqn.replace('<=>', '=')
        if(rxn_eqn not in self.reaction_eqns):
            raise ValueError(f"Unrecognized reaction: {rxn_eqn}")
        elif(A_mult is None and Ea_delta is None):
            raise ValueError(f"Please set a value for either A_mult or Ea_delta: {rxn_eqn}")
        elif(n_mult and n_delta):
            raise ValueError(f"Can't simultaneously set n_mult and n_delta: {rxn_eqn}")
        else:
            if(rxn_eqn in self.duplicates):
                idxs = self.duplicates[rxn_eqn]
                rxn_idx_str = "-".join(map(str,idxs))
            else:
                rxn_idx_str = str(self.reaction_eqns.index(rxn_eqn))
            for ridx in map(int, rxn_idx_str.split("-")):
                if ridx in self.opt_rxn_idxs:
                    print(f"Reaction already included ({rxn_eqn}). Ignoring.")
                    return
                self.opt_rxn_idxs.add(ridx)
            if A_mult is not None:
                self.opt_bounds.append((1/A_mult, A_mult))
                self.x0.append(1.0)
                self.param_fns.append( self.mult_A_fn(rxn_idx_str) )
            if n_delta is not None:
                self.opt_bounds.append((-n_delta, n_delta))
                self.x0.append(0.0)
                self.param_fns.append( self.add_n_fn(rxn_idx_str) )
            if n_mult is not None:
                self.opt_bounds.append((1/n_mult, n_mult))
                self.x0.append(1.0)
                self.param_fns.append( self.mult_n_fn(rxn_idx_str) )
            if Ea_delta is not None:
                self.opt_bounds.append((-Ea_delta, Ea_delta))
                self.x0.append(0.0)
                self.param_fns.append( self.add_Ea_fn(rxn_idx_str) )
    def get_opt_bounds(self):
        return self.opt_bounds
    def get_num_opt_vars(self):
        return len(self.opt_bounds)
    def add_opt_app(self, app):
        self.opt_apps.append(app)
    def set_verbose(self, v):
        self.verbose = bool(v)
    def write_mech(self, params, mech_file):
        for p, fn in zip(params, self.param_fns):
            fn(p)
        with open(mech_file,'w') as mfile:
           for line in self.prelines:
               mfile.write(line)
           for idx, rxn in enumerate(self.reactions):
               if idx in self.opt_rxn_idxs:
                   mfile.write(f"{rxn['eqn_orig']} {rxn['A']:0.4e} {rxn['n']:0.6g} {rxn['Ea']:0.6g}")
                   mfile.write(f"   ! Zero-RK optimized from: {rxn['Aorig']:0.4e} {rxn['norig']:0.6g} {rxn['Eaorig']:0.6g}")
                   if len(rxn['comment']) > 0:
                       mfile.write(" !" + rxn['comment'])
                   else:
                       mfile.write("\n")
               else:
                   mfile.write(rxn['line'])
               lidx = 0
               for l,c in zip(rxn['aux'],rxn['aux_comments']):
                   if 'PLOG' == l.strip().upper()[0:4] and idx in self.opt_rxn_idxs:
                       p, A, n, Ea = self.reactions[idx]['plog_mod'][lidx]
                       ll = f"PLOG / {p:4.2e} {A:0.4e} {n:0.6g} {Ea:0.6g} /"
                       p, A, n, Ea = self.reactions[idx]['plog_orig'][lidx]
                       ll += f" ! Zero-RK optimized from {A:0.4e} {n:0.6g} {Ea:0.6g}"
                       if(len(c)>0):
                           ll += " !" + c
                       else:
                           ll += "\n"
                       mfile.write(ll)
                       lidx += 1
                   else:
                       mfile.write(l)
           for line in self.postlines:
               mfile.write(line)

    def plot_values(self, values_in, filename):
        values = np.array(values_in)
        x = np.array(range(len(values)))+1
        ten_best_idxs = values.argsort()[:10]
        x_best = x[ten_best_idxs]
        best_vals = values[ten_best_idxs]

        plt.plot(x, values, 'b-', label="All")
        plt.plot(x_best, best_vals, 'ro', label="Ten best")
        plt.plot(x_best[0], best_vals[0], 'gs', label="Best")
        plt.xlabel("Iteration")
        plt.ylabel("Objective Function")
        plt.ylim([0, 5*best_vals[0]])
        plt.title(f"Minimum score: {best_vals[0]} (iteration {ten_best_idxs[0]} / {x[-1]})")
        plt.legend()
        plt.savefig(filename)
        plt.clf()

    def opt_fn(self, params):
        if(len(params.shape) == 1):
            params = np.atleast_2d(params)
        else:
            params = params.T

        tmpdirs = []
        mech_files = []
        idx_map = {}
        params_new = []
        num_params = params.shape[0]
        values = np.zeros( (num_params,) )

        self.curr_calls += num_params

        for i,p in enumerate(params):
            try:
                #Assumes we are called in the same order to avoid lookup
                idx = self.curr_calls - num_params + i
                if(idx >= len(self.params) or tuple(p) != self.params[idx]):
                    #If not in same order still see if we have computed this
                    #value before
                    idx = self.params.index(tuple(p))
                values[i] = self.values[idx]
            except ValueError:
                idx_map[len(params_new)] = i
                params_new.append(p)
                tmpdirs.append(tempfile.mkdtemp(dir=os.getcwd()))
                mech_files.append(os.path.join(tmpdirs[-1],'chem.inp'))
                self.write_mech(p,mech_files[-1])

        if(mech_files):
            inputs = [(app, mf, self.therm_file, self.trans_file) for mf in mech_files
                                                                  for app in self.opt_apps]

            try:
                if self.nprocs > 1:
                    with Pool(self.nprocs) as p:
                        outputs = p.map(parallel_opt_fn, inputs)
                        for j, a in enumerate(self.opt_apps):
                            for i, mf in enumerate(mech_files):
                                o = outputs[i*len(self.opt_apps)+j]
                                idx = idx_map[i]
                                values[idx] += o[0]
                                self.opt_apps[j] = o[1]
                else:
                    for i,mf in enumerate(mech_files):
                        for app in self.opt_apps:
                            idx = idx_map[i]
                            values[idx] += app.opt_fn(mech_files[i], self.therm_file, self.trans_file)
            finally:
                for tmpdir in tmpdirs:
                    if(not ZERORK_DEBUG):
                        rmtree_retry(tmpdir)

            for i,p in enumerate(params_new):
                self.params.append(tuple(p))
                idx = idx_map[i]
                values[idx] /= len(self.opt_apps)
                self.values.append(values[idx])

        #Output
        num_calls_total = len(self.values)
        num_calls = num_calls_total - self.prev_calls
        if(self.output_directory is not None):
            for p, val in zip(params, values):
                if(val < self.best_val):
                    self.best_val = val
                    mech_file_out = os.path.join(self.output_directory, os.path.basename(self.mech_file)+"_optimized")
                    self.write_mech(p, mech_file_out)

            if(num_calls > self.next_output):
                plot_file_out = os.path.join(self.output_directory, "history.png")
                self.plot_values(self.values, plot_file_out)

        if(self.param_log is not None):
            for p,val in zip(params, values):
                self.param_log.write(",".join([f"{pi}" for pi in p])+ f",{val}\n")
            self.param_log.flush()

        if(num_calls > self.next_output):
            self.next_output += self.check_interval
            progress = float(num_calls_total)/self.estimated_num_fcalls
            current_time = time.perf_counter()
            elapsed_time = current_time - self.start_time
            interval_time = current_time - self.output_time
            self.output_time = time.perf_counter()
            rate = self.check_interval / interval_time
            estimated_time_left = (self.estimated_num_fcalls-num_calls_total)/rate
            print(f"Completed {num_calls_total} / {self.estimated_num_fcalls}"
                  f" ({progress*100:.04g}%) in {elapsed_time:.03g} s. Expect to"
                  f" finish in {estimated_time_left:.03g} s.")

        if(values.shape[0] == 1):
            return values[0]
        else:
            return values

    def optimize(self, method="basinhopping", check_interval=10,
                 output_directory=None, x0=None, opt_kwargs={}):

        self.params = []
        self.values = []
        self.best_val = np.inf
        if(output_directory is not None):
            self.output_directory = output_directory
            if(not os.path.isdir(self.output_directory)):
                os.mkdir(self.output_directory)
        else:
            self.output_directory = "OPT_" + os.path.basename(self.mech_file)
            i=1
            while(os.path.isdir(self.output_directory)):
                self.output_directory = f"OPT{i}_"+os.path.basename(self.mech_file)
                i+=1
            if(i>1):
                print(f"WARNING: found existing OPT directory for {self.mech_file}.  Using {self.output_directory} for current output.")
            os.mkdir(self.output_directory)

        self.check_interval = check_interval
        self.next_output = check_interval
        if(x0 is None):
            x0 = self.x0

        self.curr_calls = 0
        param_filename = os.path.join(self.output_directory, "params.log")
        if(os.path.exists(param_filename)):
           with open(param_filename,'r') as restart_log:
               for line in restart_log:
                   if(line[0] == "#"):
                       continue
                   try:
                       params_value = list(map(float,line.split(",")))
                   except:
                       break
                   if(len(params_value) != self.get_num_opt_vars() + 1):
                       break
                   self.params.append(tuple(params_value[:-1]))
                   self.values.append(params_value[-1])
        if(self.values):
            self.best_val = np.min(self.values)
            self.prev_calls = len(self.values)
            i=1
            backup_file = os.path.join(self.output_directory, f"params.log.{i}")
            while(os.path.exists(backup_file)):
                i+=1
                backup_file = os.path.join(self.output_directory, f"params.log.{i}")
            shutil.copy(param_filename, backup_file)
            print(f"Found existing params.log file.  Reusing existing results.",
                  f"Copying existing file to {backup_file}.")

        self.start_time = time.perf_counter()
        self.output_time = time.perf_counter()
        with open(param_filename,'w') as self.param_log:
            self.param_log.write("#"+",".join([f"p{i}" for i in range(self.get_num_opt_vars())])+f",objective\n")
            self.param_log.flush()

            if(method=="basinhopping"):
                niters = 5
                minimizer_kwargs = opt_kwargs
                minimizer_kwargs.setdefault('method','Nelder-Mead')
                minimizer_kwargs.setdefault('bounds',self.opt_bounds)
                minimizer_kwargs.setdefault('options',{})
                minimizer_kwargs['options'].setdefault('maxfev', 1500)
                minimizer_kwargs['options'].setdefault('fatol', 1.0e-5)
                minimizer_kwargs['options'].setdefault('adaptive', True)

                self.estimated_num_fcalls=minimizer_kwargs['options']['maxfev']*(niters+1)
                bounds_check = BoundsCheck(self.opt_bounds)
                opt_result = optimize.basinhopping(self.opt_fn, x0,
                                                   minimizer_kwargs=minimizer_kwargs,
                                                   accept_test=bounds_check,
                                                   niter=niters,
                                                   seed=42)
            else:
                popsize = 12
                maxiter_de = 20
                maxiter_nm = 1000

                de_options = { "popsize":popsize, "seed":42, "maxiter":maxiter_de, "polish": False}
                de_options.update(opt_kwargs)
                if(self.nprocs > len(self.opt_apps)):
                    de_options["vectorized"] = True
                    de_options["updating"] = "deferred"

                fcalls_de = (de_options['maxiter'] + 1)*de_options['popsize']*len(x0)
                self.estimated_num_fcalls=fcalls_de + maxiter_nm

                opt_result = optimize.differential_evolution(self.opt_fn,  self.opt_bounds, **de_options)

                nm_options = {'maxiter': maxiter_nm}
                nm_options['fatol'] = 1.0e-5
                nm_options['adaptive'] = True
                opt_result = optimize.minimize(self.opt_fn, opt_result.x, method="Nelder-Mead",
                                             options=nm_options)

        print(f"Completed optimization.")
        self.param_log = None

