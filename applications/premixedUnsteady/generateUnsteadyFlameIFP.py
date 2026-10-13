#!/usr/bin/env python

import os

from SpifyParserGenerator import SpifyParserGenerator as spg

#Name your parser
spify_parser_name = "UnsteadyFlameIFP"

spify_parser_params = []

#Specify parameters
spify_parser_params.append(
{
    'name':'mech_file',
    'type':'string',
    'shortDesc' : "Chemkin Format Mechansim File [input]"
}
)

spify_parser_params.append(
{
    'name':'therm_file',
    'type':'string',
    'shortDesc' : "Chemkin Format Thermodynamics File [input]"
}
)

spify_parser_params.append(
{
    'name':'trans_file',
    'type':'string',
    'shortDesc' : "Transport File [input]"
}
)

spify_parser_params.append(
{
    'name':'transport_model',
    'type':'string',
    'shortDesc' : "Transport model [input]"
}
)

spify_parser_params.append(
{
    'name':'grid_file',
    'type':'string',
    'longDesc' : "Grid file [input] column 1 = position [m]",
    'defaultValue': os.devnull
}
)

spify_parser_params.append(
{
    'name':'wall_temperature_file',
    'type':'string',
    'longDesc' : "Wall temperature file [input] column 1 = position [m], column 2 = temperature [K].  Inlet position is at zero. If set to os.devnull then the Nusselt number is set to zero for an adiabatic case.",
    'defaultValue': os.devnull
}
)

spify_parser_params.append(
{
    'name':'initial_profile_file',
    'type':'string',
    'longDesc' : "Initial species and temperature file [input] column 1 = position [m], column 2 = mass flow, column 3=temperature [K] column 4->num_species+3.  Inlet position is at zero. If set to os.devnull then the inlet composition is used.",
    'defaultValue': os.devnull
}
)

spify_parser_params.append(
{
    'name':'restart_file',
    'type':'string',
    'longDesc' : "Binary data file for restarts. Contains number of grid points, number of variables, time, and all species, relative volume, and temperature data.",
    'defaultValue': os.devnull
}
)


spify_parser_params.append(
{
    'name':'log_file',
    'type':'string',
    'shortDesc' : "Mechanism Parser Log File [output]",
    'defaultValue' : os.devnull
}
)

spify_parser_params.append(
{
    'name':'field_file',
    'type':'string',
    'shortDesc' : "File name to record the time history of the field data [output]",
    'defaultValue' : os.devnull
}
)

spify_parser_params.append(
{
    'name':'pressure',
    'type':'double',
    'shortDesc' : "Pressure of the flame system [Pa]",
    'defaultValue' : 1.01325e5
}
)

spify_parser_params.append(
{
    'name':'mass_flow',
    'type':'double',
    'shortDesc' : "Mass flow of the system [kg/s]",
    'defaultValue' : 1.0e-6
}
)

spify_parser_params.append(
{
    'name':'diameter',
    'type':'double',
    'shortDesc' : "diameter of the flame tube [m]",
    'defaultValue' : 1.0e-3
}
)

spify_parser_params.append(
{
    'name':'inlet_velocity',
    'type':'double',
    'shortDesc' : "Inlet velocity of the flame tube [m/s]",
    'defaultValue' : 1.0
}
)

spify_parser_params.append(
{
    'name':'length',
    'type':'double',
    'shortDesc' : "length of the flame tube [m]",
    'defaultValue' : 0.01
}
)

spify_parser_params.append(
{
    'name':'nusselt',
    'type':'double',
    'shortDesc' : "Nusselt number for the heat transfer to the wall [-]",
    'defaultValue': 0.0
}
)

spify_parser_params.append(
{
    'name':'inlet_fuel_comp',
    'type':'m_string_double',
    'shortDesc' : "Fuel composition mole fraction map for inlet"
}
)

spify_parser_params.append(
{
    'name':'inlet_oxidizer_comp',
    'type':'m_string_double',
    'shortDesc' : "Oxidizer composition mole fraction map for inlet"
}
)

spify_parser_params.append(
{
    'name':'initial_comp',
    'type':'m_string_double',
    'shortDesc' : "Mixture composition mole fraction map for the entire domain"
}
)

spify_parser_params.append(
{
    'name':'inlet_full_comp',
    'type':'m_string_double',
    'shortDesc' : "Full composition mole fraction map for inlet",
    'defaultValue': {}
}
)

spify_parser_params.append(
    {
        'name':'egr_comp',
	'type':'m_string_double',
        'shortDesc' : "EGR composition mole fraction map for the inlet",
        'defaultValue': {}
    }
)


spify_parser_params.append(
{
    'name':'initial_inlet_extent',
    'type':'double',
    'longDesc' : "Extent [m] to which the inlet composition is initialized within the domain interior. That is, every grid point less than or equal to this value is set to the inlet composition (from inlet_oxidizer_comp, inlet_fuel_comp and inlet_phi).  Every grid point greater than this value is set to initial_comp.",
    'defaultValue' : 0.005
}
)


spify_parser_params.append(
{
    'name':'thickness',
    'type':'double',
    'longDesc' : "Thickness of mixing region between inlet and initial compositions [m].",
    'defaultValue' : 1e-5
}
)


spify_parser_params.append(
{
    'name':'max_time',
    'type':'double',
    'shortDesc' : "Maximum integration time [s]"
}
)

spify_parser_params.append(
{
    'name':'track_max',
    'type':'v_string',
    'longDesc' : "Vector of species names (case sensitive) to track the position of their maximum value in the domain. In the case of multiple maxima at the same value, only the most upstream position is reported. Note that temperature is tracked automatically.",
    'defaultValue' : ['not_a_species']
}
)

spify_parser_params.append(
{
    'name':'print_dt',
    'type':'double',
    'shortDesc' : "Print the basic stats every interval equal to print_dt [s]",
    'defaultValue' : 1.0e-3
}
)

spify_parser_params.append(
{
    'name':'stats_dt',
    'type':'double',
    'longDesc' : "Print the basic stats every interval equal to stats_dt [s]",
    'defaultValue' : 1
}
)

spify_parser_params.append(
{
    'name':'field_dt_multiplier',
    'type':'int',
    'longDesc' : "Print all the field variables at an interval equal to this multiplier times stats_dt",
    'defaultValue' : 100
}
)

spify_parser_params.append(
{
    'name':'convective_scheme_type',
    'type':'int',
    'longDesc' : "Convective scheme type: (0 = First order upwind, 1 = Second order upwind, 2 = Second order centered)",
    'defaultValue' : 2
}
)

spify_parser_params.append(
{
    'name':'simulation_type',
    'type':'int',
    'longDesc' : "Simulation type: (0 = FREI, 1 = Steady flame)",
    'defaultValue' : 1
}
)

spify_parser_params.append(
{
    'name':'integrator_type',
    'type':'int',
    'longDesc' : "Integrator type: (0 = CVode direct dense solve using J from divided differences, 1 = CVode direct banded solve using J from divided differences, 2 = CVode direct banded solve using user supplied J, 3 = CVode preconditioner)",
    'defaultValue' : 3
}
)

spify_parser_params.append(
{
    'name':'implicit_transport',
    'type':'bool',
    'longDesc' : "Flag that when set to true [y] treats the transport terms implicitly",
    'defaultValue' : 0
}
)

spify_parser_params.append(
{
    'name':'store_jacobian',
    'type':'bool',
    'longDesc' : "Flag that when set to true [y] stores the jacobian data for each grid point when computed by user (integrator_type == 3)",
    'defaultValue' : 0
}
)

spify_parser_params.append(
{
    'name':'one_step',
    'type':'bool',
    'longDesc' : "Flag that when set to true [y] computes the source terms from a one-step reaction using the parameters from the input file",
    'defaultValue' : 0
}
)

spify_parser_params.append(
{
    'name':'prefactor',
    'type':'double',
    'shortDesc' : "One-step A factor",
    'defaultValue' : 1.0e-13
}
)

spify_parser_params.append(
{
    'name':'activation_energy',
    'type':'double',
    'shortDesc' : "One-step activation energy",
    'defaultValue' : 4e4
}
)

spify_parser_params.append(
{
    'name':'fuel_exponent',
    'type':'double',
    'shortDesc' : "One-step fuel concentration exponent",
    'defaultValue' : 0.3
}
)

spify_parser_params.append(
{
    'name':'oxidizer_exponent',
    'type':'double',
    'shortDesc' : "One-step oxidizer concentration exponent",
    'defaultValue' : 1.3
}
)

spify_parser_params.append(
{
    'name':'max_internal_dt',
    'type':'double',
    'shortDesc' : "Maximum Internal Integrator Step Size [s]",
    'defaultValue' : 1.0e-6
}
)

spify_parser_params.append(
{
    'name':'max_num_steps',
    'type':'double',
    'shortDesc' : "Maximum Number of Integrator Steps",
    'defaultValue' : 1000000
}
)

spify_parser_params.append(
{
    'name' :'num_points',
    'type':'int',
    'shortDesc' : "Number of grid points",
    'defaultValue':100
}
)

spify_parser_params.append(
{
    'name':'rel_tol',
    'type':'double',
    'shortDesc':"Relative Integrator Tolerance",
    'defaultValue':1.0e-8
}
)

spify_parser_params.append(
{
    'name':"abs_tol",
    'type':'double',
    'shortDesc':"Absolute Integrator Tolerance",
    'defaultValue':1.0e-20
}
)

spify_parser_params.append(
{
    'name':"ref_temperature",
    'type':'double',
    'shortDesc' : "Reference temperature to normalize the ODE dT/dt equation",
    'defaultValue' : 1000.0
}
)

spify_parser_params.append(
{
    'name':"ignition_temperature",
    'type':'double',
    'shortDesc' : "Initial temperature of mixture for steady flame cases",
    'defaultValue' : 1500.0
}
)

spify_parser_params.append(
{
    'name':"inlet_temperature",
    'type':'double',
    'shortDesc' : "Inlet temperature [K]",
    'defaultValue' : 300.0
}
)

spify_parser_params.append(
{
    'name':"inlet_phi",
    'type':'double',
    'shortDesc' : "Inlet equivalence ratio (F/A)/(F/A)_{st} [-]",
    'defaultValue': 1.0
}
)

spify_parser_params.append(
{
    'name':"egr",
    'type':'double',
    'shortDesc' : "Inlet egr value [-]",
    'defaultValue': 0.0
}
)

spify_parser_params.append(
{
    'name':"step_limiter",
    'type':'double',
    'shortDesc' : "Reaction rate limiter",
    'defaultValue': 1.0e+300
}
)

spify_parser_params.append(
{
    'name':"preconditioner_threshold",
    'type':'double',
    'shortDesc' : "preconditioner threshold",
    'defaultValue': 1.0e-3
}
)

spify_parser_params.append(
{
    'name':'use_equilibrium_for_init',
    'type':'bool',
    'longDesc' : "Flag that when set to true [y] uses equilibrium composition for initialization",
    'defaultValue' : 0
}
)

spify_parser_params.append(
{
    'name':'write_mole_fractions_to_field_files',
    'type':'bool',
    'longDesc' : "Flag that when set to true [y] includes mole fractions in field file output",
    'defaultValue' : 0
}
)

spify_parser_params.append(
{
    'name':'write_step_rates_to_field_files',
    'type':'bool',
    'longDesc' : "Flag that when set to true [y] includes step rates in field file output",
    'defaultValue' : 0
}
)

#Generate parser code
spg().generate(spify_parser_name,spify_parser_params)


#Done.
