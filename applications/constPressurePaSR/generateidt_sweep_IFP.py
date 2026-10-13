#!/usr/bin/env python

import os,sys

#Root of spify src directory
SPIFY_SRC_DIR = '${HOME}/src/spify'

#Name your parser
spify_parser_name = "idt_sweep_IFP"

spify_parser_params = []

#Specify parameters

spify_parser_params.append(
{
    'name':'mechFile',
    'type':'string',
    'shortDesc' : "Chemkin Format Mechansim File"
}
)

spify_parser_params.append(
{
    'name':'thermFile',
    'type':'string',
    'shortDesc' : "Chemkin Format Thermodynamics File"
}
)

spify_parser_params.append(
{
    'name':'transport_file',
    'type':'string',
    'shortDesc' : "Chemkin Format Transport File",
    'defaultValue': ''
}
)

spify_parser_params.append(
{
    'name':'transport_model',
    'type':'string',
    'shortDesc' : "Chemkin Format Transport File",
    'defaultValue': 'ConstantLewis'
}
)

spify_parser_params.append(
{
    'name':'heat_transfer_model',
    'type':'string',
    'shortDesc' : "Wall heat transfer model",
    'defaultValue': 'none',
    'discreteValues': ['none', 'Lignola-Reverchon' , 'David-Houzelot-Villermaux'],
}
)

spify_parser_params.append(
{
    'name':'logFile',
    'type':'string',
    'shortDesc' : "Mechanism Parser Log File",
    'defaultValue' : '/dev/null'
}
)

spify_parser_params.append(
{
    'name':'idtFile',
    'type':'string',
    'shortDesc' : "Ignition Delay Output File",
}
)

spify_parser_params.append(
{
    'name':'thistFile',
    'type':'string',
    'shortDesc' : "Temperature History Output File",
}
)

spify_parser_params.append(
{
    'name':'fuel_mole_fracs',
    'type':'m_string_double',
    'shortDesc' : "Fuel mole fractions"
}
)

spify_parser_params.append(
{
    'name':'oxidizer_mole_fracs',
    'type':'m_string_double',
    'shortDesc' : "Oxidizer mole fractions"
}
)

spify_parser_params.append(
{
    'name':'trace_mole_fracs',
    'type':'m_string_double',
    'shortDesc' : "Trace mole fractions",
    'defaultValue' : {}
}
)

spify_parser_params.append(
{
    'name':'delta_T_ignition',
    'type':'double',
    'defaultValue': 400.0,
    'boundMin': 50.0,
    'boundMax': 100000.0
}
)

spify_parser_params.append(
{
    'name':'max_internal_dt',
    'type':'double',
    'shortDesc' : "Maximum Internal Integrator Step Size",
    'defaultValue' : 0.05
}
)

spify_parser_params.append(
{
    'name' :'max_internal_steps',
    'type':'int',
    'shortDesc' : "Maximum Number of Integrator Steps",
    'defaultValue':1000000
}
)

spify_parser_params.append(
{
    'name':'relative_tolerance',
    'type':'double',
    'shortDesc':"Relative Integrator Tolerance",
    'defaultValue':1.0e-8
}
)

spify_parser_params.append(
{
    'name':"absolute_tolerance",
    'type':'double',
    'shortDesc':"Absolute Integrator Tolerance",
    'defaultValue':1.0e-20
}
)


spify_parser_params.append(
{
    'name':"reference_temperature",
    'type':'double',
    'shortDesc' : "Ignition Delat Metric: Maximum species concentration",
    'defaultValue' : 1000.0
}
)


spify_parser_params.append(
{
    'name':"max_cvode_fails1",
    'type':'int',
    'shortDesc' : "Max allowed CVODE failures at original preconditioner threshold",
    'defaultValue' : 5
}
)

spify_parser_params.append(
{
    'name':"max_cvode_fails2",
    'type':'int',
    'shortDesc' : "Max allowed CVODE failures at safety preconditioner threshold",
    'defaultValue' : 2
}
)

spify_parser_params.append(
{
    'name':"safety_threshold",
    'type':'double',
    'shortDesc' : "Safety preconditionary threshold (used after max_cvode_fails1 failures)",
    'defaultValue' : 1.0e-6
}
)

spify_parser_params.append(
{
    'name':"continue_after_ignition",
    'type':'int',
    'shortDesc' : "Continue integration to stop time after ignition delay criteria met",
    'defaultValue' : 1
}
)


spify_parser_params.append(
{
    'name':"initial_temperatures",
    'type':'v_double',
    'shortDesc' : "Vector of initial temperatures to use in sweep.",
}
)

spify_parser_params.append(
{
    'name':"initial_pressures",
    'type':'v_double',
    'shortDesc' : "Vector of initial pressures to use in sweep.",
}
)

spify_parser_params.append(
{
    'name':"initial_phis",
    'type':'v_double',
    'shortDesc' : "Vector of initial equivalence ratios to use in sweep.",
}
)

spify_parser_params.append(
{
    'name':"initial_egrs",
    'type':'v_double',
    'shortDesc' : "Vector of initial EGR ratios to use in sweep.",
}
)

spify_parser_params.append(
{
    'name':"preconditioner_thresholds",
    'type':'v_double',
    'shortDesc' : "Vector of preconditioner thresholds to use in sweep.",
    'defaultValue' : [1.0e-3]
}
)

spify_parser_params.append(
{
    'name':"max_krylov_dimensions",
    'type':'v_int',
    'shortDesc' : "Vector of Krylov dimensions to use in sweep.",
    'defaultValue' : [5]
}
)



spify_parser_params.append(
{
    'name':"one_step_mode",
    'type':'int',
    'shortDesc' : "Switch: integrate in one step mode.",
    'defaultValue' : 0
}
)

spify_parser_params.append(
{
    'name':"long_output",
    'type':'int',
    'shortDesc' : "Switch: longer floating point output data.",
    'defaultValue' : 0
}
)

spify_parser_params.append(
{
    'name':"fake_update",
    'type':'int',
    'shortDesc' : "Switch: fakeupdate.",
    'defaultValue' : 0
}
)

spify_parser_params.append(
{
    'name':"incomplete_LU",
    'type':'int',
    'shortDesc' : "Switch: ILU.",
    'defaultValue' : 0
}
)

spify_parser_params.append(
{
    'name':"threshold_type",
    'type':'int',
    'shortDesc' : "Switch: Preconditioner Threshold Type.",
    'defaultValue' : 1
}
)

spify_parser_params.append(
{
    'name':"partial_pivot_threshold",
    'type':'double',
    'shortDesc' : "partial pivot threshold.",
    'defaultValue' : 0.0
}
)

spify_parser_params.append(
{
    'name':"permutation_type",
    'type':'int',
    'shortDesc' : "perm type.",
    'defaultValue' : 1
}
)

spify_parser_params.append(
{
    'name':"eps_lin",
    'type':'double',
    'shortDesc' : "??",
    'defaultValue' : 0.1
}
)

spify_parser_params.append(
{
    'name':"nonlinear_convergence_coefficient",
    'type':'double',
    'shortDesc' : "??",
    'defaultValue' : 0.05
}
)

spify_parser_params.append(
{
    'name':"steplimiter",
    'type':'double',
    'shortDesc' : "reaction rate limiter",
    'defaultValue' : 1e300
}
)

spify_parser_params.append(
{
    'name':"num_particles",
    'type':'int',
    'shortDesc' : "Number of particles",
    'defaultValue' : 100
}
)

spify_parser_params.append(
{
    'name':"residence_time",
    'type':'double',
    'shortDesc' : "Residence time",
    'defaultValue' : 1.0e-3
}
)

spify_parser_params.append(
{
    'name':"num_residence_time",
    'type':'int',
    'shortDesc' : "Number of residence time to simulate",
    'defaultValue' : 10
}
)

spify_parser_params.append(
{
    'name':"mixing_time",
    'type':'double',
    'shortDesc' : "Mixing time",
    'defaultValue' : 1.0e-3
}
)

spify_parser_params.append(
{
    'name':"pairing_time",
    'type':'double',
    'shortDesc' : "Pairing time",
    'defaultValue' : 1.0e-3
}
)

spify_parser_params.append(
{
    'name':"exhaust_temperature",
    'type':'double',
    'shortDesc' : "Exhaust temperature",
    'defaultValue' : 2400.0
}
)

spify_parser_params.append(
{
    'name':"combustionRegime",
    'type':'int',
    'shortDesc' : "Combustion regime: 1 for premixed, 2 for diffusion",
    'defaultValue' : 1
}
)

spify_parser_params.append(
{
    'name':"writeState",
    'type':'int',
    'shortDesc' : "Write state vector to file at each time step",
    'defaultValue' : 0
}
)

spify_parser_params.append(
{
    'name':"inlet_ratios",
    'type':'v_double',
    'shortDesc' : "Vector of mass flow ratios for inlet streams (must be 2 (fuel+ox and pilot) for premixed and 3 for non-premixed (oxidizer, fuel, pilot)).",
    'defaultValue' : [1.0],
}
)

spify_parser_params.append(
{
    'name':"pilot_duration",
    'type':'double',
    'shortDesc' : "Time in seconds where pilot is active (negative values will make pilot permanent).",
    'defaultValue': -1,
}
)

spify_parser_params.append(
{
    'name':"reactor_radius",
    'type':'double',
    'shortDesc' : "Reactor radius in meters",
    'defaultValue': 1.0e-1,
}
)

spify_parser_params.append(
{
    'name':"njets",
    'type':'int',
    'shortDesc' : "Number of Jets",
    'defaultValue': 4,
}
)

spify_parser_params.append(
{
    'name':"djet",
    'type':'double',
    'shortDesc' : "Jet Diameter in meters",
    'defaultValue': 1.0e-3,
}
)

spify_parser_params.append(
{
    'name':"wall_temperature",
    'type':'double',
    'shortDesc' : "Temperature of the wall in Kelvin",
    'defaultValue': 573,
}
)

spify_parser_params.append(
{
    'name':"random_seed",
    'type':'int',
    'shortDesc' : "Seed for pseudo-random number generator. If 0 will not be set",
    'defaultValue': 0,
}
)


spify_parser_params.append(
{
    'name':"load_balance",
    'type':'bool',
    'shortDesc' : "Active load balancing for chemistry",
    'defaultValue' : 0
}
)

spify_parser_params.append(
{
    'name':"fully_coupled",
    'type':'bool',
    'shortDesc' : "Coupled reacting and mixing steps",
    'defaultValue' : 1
}
)


#Make sure we can import SpifyParserGenerator
sys.path.append(os.path.join(os.path.expandvars(SPIFY_SRC_DIR),'src'))

#Import
from SpifyParserGenerator import SpifyParserGenerator as spg

#Generate parser code
spg().generate(spify_parser_name,spify_parser_params)
spg().make_master_file(spify_parser_name,spify_parser_params)


#Done.
