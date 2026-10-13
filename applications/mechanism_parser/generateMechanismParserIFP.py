#!/usr/bin/env python

import os

from SpifyParserGenerator import SpifyParserGenerator as spg

#Name your parser
spify_parser_name = "MechanismParserIFP"

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
    'name':'log_file',
    'type':'string',
    'shortDesc' : "Mechanism Parser Log File [output]",
    'defaultValue' : os.devnull
}
)

#Generate parser code
spg().generate(spify_parser_name,spify_parser_params)


#Done.
