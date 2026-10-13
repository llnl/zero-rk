#!/usr/bin/env python

import os

from SpifyParserGenerator import SpifyParserGenerator as spg

#Name your parser
spify_parser_name = "SteadyFlameIFP"

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
    'name':'pressure',
    'type':'double',
    'shortDesc' : "Pressure of the flame system [Pa]",
    'defaultValue' : 1.01325e5
}
)

spify_parser_params.append(
{
    'name':'mass_flux_fuel',
    'type':'double',
    'shortDesc' : "Mass flux of the fuel stream [kg/s]",
    'defaultValue' : 1.0
}
)

spify_parser_params.append(
{
    'name':'mass_flux_oxidizer',
    'type':'double',
    'shortDesc' : "Mass flux of the oxidizer stream [kg/s]",
    'defaultValue' : 1.0
}
)

spify_parser_params.append(
{
    'name':'length',
    'type':'double',
    'shortDesc' : "length of the grid/separation distance [m]",
    'defaultValue' : 0.1
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
    'name':'inlet_full_comp',
    'type':'m_string_double',
    'shortDesc' : "Full composition mole fraction map for inlet",
    'defaultValue': {}
}
)

spify_parser_params.append(
{
    'name':'thickness',
    'type':'double',
    'longDesc' : "Thickness of mixing region between fuel and oxidizer compositions [m].",
    'defaultValue' : 1e-3
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
    'longDesc' : "Simulation type: (0 = Planar, 1 = Axisymmetric)",
    'defaultValue' : 1
}
)

spify_parser_params.append(
{
    'name':'flame_type',
    'type':'int',
    'longDesc' : "Simulation type: (0 = diffusion, 1 = premixed twin (R-to-R), 2 = premixed single (R-to-P))",
    'defaultValue' : 0
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
    'name':'superlu_serial',
    'type':'bool',
    'longDesc' : "Flag that when set to true [y] uses the serial version of SuperLU",
    'defaultValue' : 0
}
)

spify_parser_params.append(
{
    'name':'pseudo_unsteady',
    'type':'bool',
    'longDesc' : "Flag that when set to true [y] adds a pseudo unsteady term",
    'defaultValue' : 0
}
)


spify_parser_params.append(
{
    'name':'pseudo_unsteady_dt',
    'type':'double',
    'longDesc' : "Initial pseudo-unsteady time step",
    'defaultValue' : 1.0e-3
}
)

spify_parser_params.append(
{
    'name':'pseudo_unsteady_time',
    'type':'double',
    'longDesc' : "End time for pseudo-unsteady time-stepping ",
    'defaultValue' : 0.08
}
)



spify_parser_params.append(
{
    'name':'diffusion_correction',
    'type':'bool',
    'longDesc' : "Flag that when set to true [y] includes the diffusion mass flux correction",
    'defaultValue' : 0
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
    'name':'max_subiter',
    'type':'int',
    'shortDesc' : "Maximum Number of iterations between jacobian updates",
    'defaultValue' : 10
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
    'name':"ref_momentum",
    'type':'double',
    'shortDesc' : "Reference value to normalize the momentum ODE equation",
    'defaultValue' : 1000.0
}
)

spify_parser_params.append(
{
    'name':"ignition_temperature",
    'type':'double',
    'shortDesc' : "Initial temperature of mixture for steady flame cases",
    'defaultValue' : 2200.0
}
)

spify_parser_params.append(
{
    'name':"fuel_temperature",
    'type':'double',
    'shortDesc' : "Fuel temperature [K]",
    'defaultValue' : 300.0
}
)

spify_parser_params.append(
{
    'name':"oxidizer_temperature",
    'type':'double',
    'shortDesc' : "Oxidizer temperature [K]",
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
    'name':"strain_rate",
    'type':'double',
    'shortDesc' : "Strain rate [1/s]",
    'defaultValue': 100.0
}
)

spify_parser_params.append(
{
    'name':'finite_separation',
    'type':'bool',
    'longDesc' : "Flag that when set to true [y] solves the finite separation equations",
    'defaultValue' : 1
}
)

spify_parser_params.append(
{
    'name':'fixed_temperature',
    'type':'bool',
    'longDesc' : "Flag that when set to true [y] disables the temperature equation",
    'defaultValue' : 0
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
    'name':'sensitivity_analysis',
    'type':'bool',
    'longDesc' : "Flag that when set to true [y] computes sensitivity analysis of the soot volume fraction",
    'defaultValue' : 0
}
)

spify_parser_params.append(
{
    'name':'sensitivity_multiplier',
    'type':'double',
    'shortDesc' : "A-factor multiplier used for sensitivity analysis",
    'defaultValue' : 1.5
}
)

spify_parser_params.append(
{
    'name':"sensitive_reactions",
    'type':'v_int',
    'shortDesc' : "Vector of sensitive reactions for sensitivity analysis",
    'defaultValue' : []
}
)

spify_parser_params.append(
{
    'name':"sensitivity_processors_per_solution",
    'type':'int',
    'shortDesc' : "Number of processors to use on each flame solution. Total MPI procs should divide evenly by this number",
    'defaultValue' : 0
}
)

spify_parser_params.append(
{
    'name':"sensitivity_load_balance",
    'type':'int',
    'shortDesc' : "Method to distribute sensitivity calcs across ranks [0 - static, 1 - dynamic]",
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
