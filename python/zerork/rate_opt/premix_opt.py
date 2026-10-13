
import os
import io
import sys
import time
import shutil
import pathlib
import tempfile
import contextlib
import subprocess
import importlib_resources

import numpy as np
from ruamel.yaml import YAML


from .opt_app import opt_app
from ..config import ZERORK_ROOT, ZERORK_DEBUG
from ..util.rmtree_retry import rmtree_retry

sys.path.append(os.path.join(ZERORK_ROOT , "lib" , "python" , "site-packages"))
from flamesolver import compute_unsteady_flame_speeds, compute_steady_flame_speeds

yaml=YAML(typ="safe")
yaml.default_flow_style=False

class premix_opt(opt_app):
    def __init__(self, full_mechanism=None, full_therm=None, full_trans=None, save_full=None,
                 comparison_file=None, input_file=None, exe=None, procs=1,
                 sim_time_weight=None):
        assert (full_mechanism is not None or comparison_file is not None), "Must supply either full_mechanism or comparison_file"
        self.curr_data = None
        self.n_data = 0
        self.save_full = save_full
        self.prev_solution = None
        self.comparison_file = comparison_file
        self.comp_data = None
        if comparison_file is None:
            assert(full_therm is not None), "Must supply full_therm with full_mechanism"
            assert(full_trans is not None), "Must supply full_trans with full_mechanism"
            self.full_mechanism = full_mechanism
            self.full_therm = full_therm
            self.full_trans = full_trans

            #Defer running full mechanism until all options set and we are "run"
            # in the optimization loop
        if input_file is None:
            base_yaml_ref = importlib_resources.files('zerork.rate_opt') / 'data' / 'premix_base.yml'
            with importlib_resources.as_file(base_yaml_ref) as base_yaml_file:
                with open(base_yaml_file,'r') as yfile:
                    self.yaml = yaml.load(yfile)
        else:
            with open(input_file,'r') as yfile:
                self.yaml = yaml.load(yfile)
        #TODO: Do we allow for custom exe in this app?
        #if exe is None:
        #    self.exe = ZERORK_EXE
        #else:
        #    self.exe = exe
        self.temperatures = [500]
        self.pressures = [101325]
        self.phis = [1.0]
        self.egrs = [0.0]
        self.procs = procs
        self.zerork_timeout = 300
        self.sim_time_weight = sim_time_weight
        self.error_fn = self.mean_absolute_relative_error
        self.error_fn_map = {
             'mean_square_log_error': self.mean_square_log_error,
             'mean_absolute_log_error': self.mean_absolute_log_error,
             'mean_absolute_relative_error': self.mean_absolute_relative_error,
        }

    def set(self, key, value):
        if(key == "pressures"):
            self.pressures=value
        elif(key == "temperatures"):
            self.temperatures=value
        elif(key == "phis"):
            self.phis=value
        elif(key == "egrs"):
            self.egrs=value
        else:
            self.yaml[key] = value

    def set_error_fn(self, error_fn_name):
        assert (error_fn_name in self.error_fn_map), f"Unrecognized error_fn_name: {error_fn_name}"
        self.error_fn = self.error_fn_map[error_fn_name]

    def write_yaml(self, file_name):
        with open(file_name,'w') as yfile:
            yaml.dump(self.yaml, yfile)

    def mean_square_log_error(self):
        return np.sum(np.power(np.log(self.curr_speeds) - np.log(self.comp_speeds),2))/self.n_data

    def mean_absolute_log_error(self):
        return np.sum(np.abs(np.log(self.curr_speeds) - np.log(self.comp_speeds)))/self.n_data

    def mean_absolute_relative_error(self):
        return np.sum(np.abs((self.curr_speeds - self.comp_speeds)/self.comp_speeds))/self.n_data

    def opt_fn(self, mech_file, therm_file, trans_file):
        if(self.comp_data is None):
            if(self.comparison_file):
                self.comp_data, self.comp_speeds = self.read_flame_speed_file(self.comparison_file)
            else:
                self.comp_data, self.comp_speeds, _ = self.run(self.full_mechanism,
                                                               self.full_therm,
                                                               self.full_trans,
                                                               self.save_full)
            self.n_data = self.comp_speeds.size
        self.curr_data, self.curr_speeds, curr_solution = self.run(mech_file, therm_file, trans_file)
        if(self.prev_solution is None):
            self.prev_solution = curr_solution
        assert(self.curr_speeds.size == self.n_data)
        self.err = self.error_fn()
        assert(self.sim_time_weight is None) #Not implemented for premix
        return self.err

    def read_flame_speed_file(self, filename):
        data = np.atleast_2d(np.genfromtxt(filename, comments='#'))
        steady=False
        if(data.shape[1] == 5):
            steady=True

        speeds = []
        for t in self.temperatures:
            for f in self.phis:
                for p in self.pressures:
                    for e in self.egrs:
                        speed = None
                        if(steady):
                            speed = data[(data[:,1]==t) &
                                         (data[:,0]==f) &
                                         (data[:,2]==p)][0,3]
                        else:
                            speed = data[(data[:,1]==t) &
                                          (data[:,0]==f) &
                                          (data[:,2]==p) &
                                          (data[:,3]==e)][0,4]
                        if(not speed):
                            print(f"Didn't find flame speed for {t}, {p}, {f} in {filename}")
                            sys.exit()
                        speeds.append(speed)
        speeds = np.array(speeds)
        return data, speeds


    def run(self, mech_file, therm_file, trans_file, save_file=None):
        tmpdir = tempfile.mkdtemp(dir='.') #TODO (where to put?)
        self.yaml['mech_file'] = mech_file
        self.yaml['therm_file'] = therm_file 
        self.yaml['trans_file'] = trans_file 
        self.yaml['log_file'] = os.devnull
        self.yaml["pseudo_unsteady"] = "n"
        zerork_infile_name = 'zerork.yml'
        curr_data = None
        curr_speeds = None
        curr_solutions = None
        cwd = os.getcwd()
        os.chdir(tmpdir)
        steady=False
        try:
            nretries=0
            while True:
                try:
                    f=self.phis[0]
                    nodes = 1
                    min_pts_per_lF=8
                    zerork_out = io.StringIO()
                    solved = False
                    if(self.prev_solution is not None):
                        try:
                            with open('datainput','wb') as dfile:
                                dfile.write(self.prev_solution[0])
                            with open('gridinput','wb') as gfile:
                                gfile.write(self.prev_solution[1])

                            self.write_yaml(zerork_infile_name)
                            with contextlib.redirect_stdout(zerork_out), contextlib.redirect_stderr(zerork_out):
                                compute_steady_flame_speeds('datainput','gridinput',
                                                            zerork_infile_name, self.phis, self.temperatures,
                                                            self.pressures, self.procs, nodes,
                                                            min_pts_per_lF)
                            steady = True
                            solved = True
                        except SystemExit:
                            shutil.copy('log','logsteady')
                            shutil.copy('zerork.yml','input_steady.yml')
                            print(f"Steady solve failed [{f}]. Trying pseudo_unsteady")
                            self.yaml["pseudo_unsteady"] = "y"
                            self.yaml["pseudo_unsteady_time"] = 0.2
                            self.write_yaml(zerork_infile_name)
                            try:
                                with contextlib.redirect_stdout(zerork_out), contextlib.redirect_stderr(zerork_out):
                                    compute_steady_flame_speeds('datainput','gridinput',
                                                                zerork_infile_name, self.phis, self.temperatures,
                                                                self.pressures, self.procs, nodes,
                                                                min_pts_per_lF)
                                steady = True
                                solved = True
                            except SystemExit:
                                print(zerork_out.getvalue())
                                shutil.copy('log','logpseudosteady')
                                shutil.copy('zerork.yml','input_pseudosteady.yml')
                                print(f"Psuedo-steady solve failed [{f}]. Using unsteady.")
                    if(not solved):
                        self.write_yaml(zerork_infile_name)
                        do_steady_solve=True
                        min_pts_per_lF=12
                        with contextlib.redirect_stdout(zerork_out), contextlib.redirect_stderr(zerork_out):
                            compute_unsteady_flame_speeds(zerork_infile_name, self.phis, self.temperatures,
                                                          self.pressures, self.egrs, self.procs, nodes,
                                                          min_pts_per_lF, do_steady_solve,
                                                          nprocs_steady=min(self.procs,2))
                            solved = True
                    break
                except subprocess.TimeoutExpired as e:
                    nretries += 1
                    if nretries >= 3:
                        print("ERROR: Zero-RK premix app timed out multiple times.")
                        raise e
                except subprocess.CalledProcessError as e:
                    print("ERROR: Zero-RK premix returned error. Output was:")
                    print(e.output)
                    raise e

            try:
                curr_data, curr_speeds = self.read_flame_speed_file('flame_speeds')
                curr_solution = None
                if(not steady):
                    t, f, p, e = self.temperatures[0], self.phis[0], self.pressures[0], self.egrs[0]
                    dfilename = f'datap{p}phi{f}Tu{t}egr{e}'
                    d = None
                    with open(dfilename,'rb') as dfile:
                        d = dfile.read()
                    gfilename = f'gridp{p}phi{f}Tu{t}egr{e}'
                    g = None
                    with open(gfilename,'rb') as gfile:
                        g = gfile.read()
                    curr_solution = (d, g)
                else:
                    t, f, p = self.temperatures[0], self.phis[0], self.pressures[0]
                    dfilename = f'datasteadyp{p}phi{f}Tu{t}'
                    d = None
                    with open(dfilename,'rb') as dfile:
                        d = dfile.read()
                    gfilename = f'gridsteadyp{p}phi{f}Tu{t}'
                    g = None
                    with open(gfilename,'rb') as gfile:
                        g = gfile.read()
                    curr_solution = (d, g)
            except (IOError, IndexError) as e:
                print("No data from ZeroRK")
                raise e
            os.chdir(cwd)
            if save_file is not None:
                shutil.copy(f'{tmpdir}/flame_speeds', save_file)
            if(solved):
                if(not ZERORK_DEBUG):
                    rmtree_retry(tmpdir)
        finally:
            os.chdir(cwd)

        return curr_data, curr_speeds, curr_solution


