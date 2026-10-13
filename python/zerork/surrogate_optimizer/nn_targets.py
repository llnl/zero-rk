
from __future__ import print_function

import importlib_resources
import numpy
from .surrogate_mixture import TargetBase, RelativeErrorFn

class RON_NN_Target(TargetBase):
    def __init__(self, target, weight, target_fn=RelativeErrorFn()):

        self.P0 = 20.0*1.0e5
        self.phi0 = 1.0
        self.egr0 = 0.0
        self.T0 = 750.0
        self.dT0 = 850.0
        self.dT_perturb = self.dT0*(1.0 + 1.0e-4)
        self.eps_T = self.dT_perturb - self.dT0

        wb_ref = importlib_resources.files('zerork.surrogate_optimizer') / 'data' / 'RON_nn_data.npz'
        with importlib_resources.as_file(wb_ref) as wb_file:
            nn_data = numpy.load(wb_file)
        self.l1_weights = nn_data['l1_weights']
        self.l1_biases  = nn_data['l1_biases']
        self.l2_weights = nn_data['l2_weights']
        self.l2_biases  = nn_data['l2_biases']
        self.norms_range = nn_data['norms_range']
        self.norms_min   = nn_data['norms_min']

        TargetBase.__init__(self,"RON (NN 2.0)",target,weight,target_fn)
    def __call__(self, mixture):
        computed_RON = self._compute_RON(mixture)
        error = TargetBase.__call__(self,computed_RON)
        return error
    def _compute_RON(self, mixture):
        for temp,p,phi,egr,idt in zip(mixture.temps,mixture.press,mixture.phis,mixture.egrs,mixture.idts):
            if(numpy.abs(temp-self.T0) < 1.0e-8
                and numpy.abs(p-self.P0) < 1.0e-8
                and numpy.abs(phi-self.phi0) < 0.001
                and numpy.abs(egr-self.egr0) < 0.001):
                idt_T0 = idt
        for temp,p,phi,egr,idt in zip(mixture.temps,mixture.press,mixture.phis,mixture.egrs,mixture.idts):
            if(numpy.abs(temp-self.dT_perturb) < 1.0e-8
                and numpy.abs(p-self.P0) < 1.0e-8
                and numpy.abs(phi-self.phi0) < 0.001
                and numpy.abs(egr-self.egr0) < 0.001):
                idt_dT_perturb = idt
        for temp,p,phi,egr,idt in zip(mixture.temps,mixture.press,mixture.phis,mixture.egrs,mixture.idts):
            if(numpy.abs(temp-self.dT0) < 1.0e-8
                and numpy.abs(p-self.P0) < 1.0e-8
                and numpy.abs(phi-self.phi0) < 0.001
                and numpy.abs(egr-self.egr0) < 0.001):
                idt_dT= idt

        didt_dt = (idt_dT_perturb - idt_dT)/self.eps_T

        atom_types  = mixture.calcAtomType(normalize=False)
        (H, C, O) = mixture.calcAtomFractions()

        feature_labels =  ['IDT', 'dIDT/dT', 'H',  'AT0', 'AT2', 'AT4', 'AT6', 'AT8', 'AT9', 'AT17']
        X = numpy.zeros( (1,len(feature_labels)) )

        X[0,0] = 1.0/idt_T0
        X[0,1] = didt_dt/idt_dT
        X[0,2] = H
        X[0,3] = atom_types[0]
        X[0,4] = atom_types[2]
        X[0,5] = atom_types[4]
        X[0,6] = atom_types[6]
        X[0,7] = atom_types[8]
        X[0,8] = atom_types[9]
        X[0,9] = atom_types[17]

        X = (X - self.norms_min)/self.norms_range

        l1 = 1/(1+numpy.exp(-(numpy.dot(X, self.l1_weights) + self.l1_biases)))
        l2 = numpy.dot(l1,self.l2_weights) + self.l2_biases
        out = l2
        RON = out[0][0]
        return RON 

    def get_idt_sweeps(self):
        sw_orig = {}
        sw_orig['temps'] = [self.T0, self.dT0, self.dT_perturb]
        sw_orig['press'] = [self.P0]
        sw_orig['phis']  = [self.phi0]
        sw_orig['egrs']  = [self.egr0]

        return [sw_orig]


class MON_NN_Target(TargetBase):
    def __init__(self, target, weight, target_fn=RelativeErrorFn()):

        self.P0 = 20.0*1.0e5
        self.phi0 = 1.0
        self.egr0 = 0.0
        self.T0 = 825.0
        self.dT0 = 850.0
        self.dT_perturb = self.dT0*(1.0 + 1.0e-4)
        self.eps_T = self.dT_perturb - self.dT0

        wb_ref = importlib_resources.files('zerork.surrogate_optimizer') / 'data' / 'MON_nn_data.npz'
        with importlib_resources.as_file(wb_ref) as wb_file:
            nn_data = numpy.load(wb_file)
        self.l1_weights = nn_data['l1_weights']
        self.l1_biases  = nn_data['l1_biases']
        self.l2_weights = nn_data['l2_weights']
        self.l2_biases  = nn_data['l2_biases']
        self.norms_range = nn_data['norms_range']
        self.norms_min   = nn_data['norms_min']

        TargetBase.__init__(self,"MON (NN 2.0)",target,weight,target_fn)
    def __call__(self, mixture):
        computed_MON = self._compute_MON(mixture)
        error = TargetBase.__call__(self,computed_MON)
        return error
    def _compute_MON(self, mixture):
        for temp,p,phi,egr,idt in zip(mixture.temps,mixture.press,mixture.phis,mixture.egrs,mixture.idts):
            if(numpy.abs(temp-self.T0) < 1.0e-8
                and numpy.abs(p-self.P0) < 1.0e-8
                and numpy.abs(phi-self.phi0) < 0.001
                and numpy.abs(egr-self.egr0) < 0.001):
                idt_T0 = idt
        for temp,p,phi,egr,idt in zip(mixture.temps,mixture.press,mixture.phis,mixture.egrs,mixture.idts):
            if(numpy.abs(temp-self.dT_perturb) < 1.0e-8
                and numpy.abs(p-self.P0) < 1.0e-8
                and numpy.abs(phi-self.phi0) < 0.001
                and numpy.abs(egr-self.egr0) < 0.001):
                idt_dT_perturb = idt
        for temp,p,phi,egr,idt in zip(mixture.temps,mixture.press,mixture.phis,mixture.egrs,mixture.idts):
            if(numpy.abs(temp-self.dT0) < 1.0e-8
                and numpy.abs(p-self.P0) < 1.0e-8
                and numpy.abs(phi-self.phi0) < 0.001
                and numpy.abs(egr-self.egr0) < 0.001):
                idt_dT = idt

        didt_dt = (idt_dT_perturb - idt_dT)/self.eps_T

        atom_types  = mixture.calcAtomType(normalize=False)
        (H, C, O) = mixture.calcAtomFractions()

        feature_labels =  ['IDT', 'dIDT/dT', 'H',  'AT0', 'AT2', 'AT4', 'AT6', 'AT8', 'AT9', 'AT17']
        X = numpy.zeros( (1,len(feature_labels)) )

        X[0,0] = 1.0/idt_T0
        X[0,1] = didt_dt/idt_dT
        X[0,2] = H
        X[0,3] = atom_types[0]
        X[0,4] = atom_types[2]
        X[0,5] = atom_types[4]
        X[0,6] = atom_types[6]
        X[0,7] = atom_types[8]
        X[0,8] = atom_types[9]
        X[0,9] = atom_types[17]

        X = (X - self.norms_min)/self.norms_range

        l1 = 1/(1+numpy.exp(-(numpy.dot(X, self.l1_weights) + self.l1_biases)))
        l2 = numpy.dot(l1,self.l2_weights) + self.l2_biases
        out = l2
        MON = out[0][0]
        return MON 

    def get_idt_sweeps(self):
        sw_orig = {}
        sw_orig['temps'] = [self.T0, self.dT0, self.dT_perturb]
        sw_orig['press'] = [self.P0]
        sw_orig['phis']  = [self.phi0]
        sw_orig['egrs']  = [self.egr0]

        return [sw_orig]


