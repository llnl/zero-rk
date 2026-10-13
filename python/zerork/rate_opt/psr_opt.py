
import os
import time
import shutil
import tempfile
import subprocess
import importlib_resources

import numpy as np
from ruamel.yaml import YAML

from .opt_app import opt_app
from ..config import ZERORK_ROOT
from ..util.rmtree_retry import rmtree_retry

yaml=YAML(typ="safe")
yaml.default_flow_style=False #TODO: Yes/no?

ZERORK_EXE=os.getenv("ZERORK_EXE", default=os.path.join(ZERORK_ROOT,'bin','constVolumePSR.x'))
ZERORK_MPI_EXE=os.getenv("ZERORK_MPI_EXE", default=os.path.join(ZERORK_ROOT,'bin','constVolumePSR_mpi.x'))


class psr_opt(opt_app):
    def __init__(self, full_mechanism=None, full_therm=None, full_trans=None, save_full=None,
                 comparison_file=None, input_file=None, exe=None, procs=1,
                 sim_time_weight=None):
        assert (full_mechanism is not None or comparison_file is not None), "Must supply either full_mechanism or comparison_file"
        self.curr_data = None
        self.curr_mass_fracs = None
        self.n_data = 0
        self.save_full = save_full
        if comparison_file is not None:
            self.comp_data = np.genfromtxt(comparison_file, comments='#', skip_footer=1)
            num_cols = self.comp_data.shape[1]
            self.num_tracked_species = int((num_cols - 10)/2)
            self.comp_mass_fracs = self.comp_data[:,-self.num_tracked_species:]
            self.n_data = self.comp_mass_fracs.size
        else:
            assert(full_therm is not None), "Must supply full_therm with full_mechanism"
            self.full_mechanism = full_mechanism
            self.full_therm = full_therm
            self.full_trans = full_trans

            self.comp_data = None
            self.comp_mass_fracs = None
            #Defer running full mechanism until all options set and we are "run"
            # in the optimization loop
        if input_file is None:
            base_yaml_ref = importlib_resources.files('zerork.rate_opt') / 'data' / 'psr_base.yml'
            with importlib_resources.as_file(base_yaml_ref) as base_yaml_file:
                with open(base_yaml_file,'r') as yfile:
                    self.yaml = yaml.load(yfile)
        else:
            with open(input_file,'r') as yfile:
                self.yaml = yaml.load(yfile)
        if exe is None:
            self.exe = ZERORK_EXE
        else:
            self.exe = exe
        self.procs = procs
        self.zerork_timeout = 300
        self.sim_time_weight = sim_time_weight
        self.error_fn = self.mean_square_log_error
        self.error_fn_map = {
             'mean_square_log_error': self.mean_square_log_error,
             'mean_absolute_log_error': self.mean_absolute_log_error,
             'mean_absolute_relative_error': self.mean_absolute_relative_error,
        }
        self.yaml['min_mass_frac'] = 1.0e-10

    def set(self, key, value):
        self.yaml[key] = value

    def set_error_fn(self, error_fn_name):
        assert (error_fn_name in self.error_fn_map), f"Unrecognized error_fn_name: {error_fn_name}"
        self.error_fn = self.error_fn_map[error_fn_name]

    def write_yaml(self, file_name):
        with open(file_name,'w') as yfile:
            yaml.dump(self.yaml, yfile)

    def mean_square_log_error(self):
        return np.sum(np.power(np.log(self.curr_mass_fracs) - np.log(self.comp_mass_fracs),2))/(self.n_data*self.num_tracked_species)

    def mean_absolute_log_error(self):
        return np.sum(np.abs(np.log(self.curr_mass_fracs) - np.log(self.comp_mass_fracs)))/(self.n_data*self.num_tracked_species)

    def mean_absolute_relative_error(self):
        return np.sum(np.abs((self.curr_mass_fracs - self.comp_mass_fracs)/self.comp_mass_fracs))/(self.n_data*self.num_tracked_species)

    def opt_fn(self, mech_file, therm_file, trans_file):
        if(self.comp_data is None):
            self.comp_data, self.comp_mass_fracs = self.run(self.full_mechanism, self.full_therm, self.full_trans, self.save_full)
            self.n_data = self.comp_mass_fracs.size
            self.comp_mass_fracs = np.maximum(self.comp_mass_fracs, self.yaml['min_mass_frac']) 
        self.curr_data, self.curr_mass_fracs = self.run(mech_file, therm_file, trans_file)
        assert(self.curr_mass_fracs.size == self.n_data)
        self.err = self.error_fn()
        if self.sim_time_weight is not None:
            print(f"#psr_opt err without sim time: {self.err}")
            self.err += self.sim_time_weight*np.sum(self.curr_data[:,8])/self.n_data
        return self.err

    def run(self, mech_file, therm_file, trans_file=None, save_file=None):
        tmpdir = tempfile.mkdtemp(dir='.') #TODO (where to put?)
        self.yaml['mechFile'] = mech_file
        self.yaml['thermFile'] = therm_file 
        idt_file = os.path.join(tmpdir,'zerork.dat')
        self.yaml['idtFile'] = idt_file
        self.yaml['logFile'] = os.path.join(tmpdir,'zerork.cklog')
        self.yaml['thistFile'] = os.path.join(tmpdir,'zerork.thist')
        zerork_infile_name = os.path.join(tmpdir,'zerork.yml')
        self.write_yaml(zerork_infile_name)
        curr_data = None
        curr_mass_fracs = None
        try:
            nretries=0
            while True:
                try:
                    cmd_list = [self.exe,zerork_infile_name]
                    if(self.procs > 1):
                        #TODO: More robust please (selection of exe, mpiexec, can we default to all procs?)
                        numprocs=str(self.procs)
                        cmd_list = ['srun', '-n', numprocs, ZERORK_MPI_EXE, zerork_infile_name]
                    zerork_out=subprocess.check_output(cmd_list, stderr=subprocess.STDOUT,
                                                       timeout=self.zerork_timeout,
                                                       universal_newlines=True).split('\n')
                    break
                except subprocess.TimeoutExpired as e:
                    nretries += 1
                    if nretries >= 3:
                        print("ERROR: Zero-RK PSR app timed out multiple times.")
                        raise e
                except subprocess.CalledProcessError as e:
                    print("ERROR: Zero-RK PSR returned error. Output was:")
                    print(e.output)
                    raise e

            try:
                curr_data = np.genfromtxt(idt_file, comments='#', skip_footer=1)
                num_cols = curr_data.shape[1]
                self.num_tracked_species = int((num_cols - 10)/2)
                curr_mass_fracs = curr_data[:,-self.num_tracked_species:]
                curr_mass_fracs = np.maximum(curr_mass_fracs, self.yaml['min_mass_frac']) 
            except (IOError, IndexError) as e:
                print("No data from ZeroRK, ZeroRK output was:")
                for line in zerork_out:
                    print("\t", line)
                raise e
            if save_file is not None:
                shutil.move(idt_file, save_file)
        finally:
            #Clean up
            rmtree_retry(tmpdir)

        return curr_data, curr_mass_fracs


