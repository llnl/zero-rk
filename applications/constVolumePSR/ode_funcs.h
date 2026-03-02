#ifndef ODE_FUNCS_H
#define ODE_FUNCS_H

#include <cvode/cvode.h>            // prototypes for CVODE fcts. and consts.
#include <nvector/nvector_serial.h> // serial N_Vector types, fcts., and macros

int const_vol_wsr(realtype t, N_Vector y, N_Vector ydot,
		  void *user_data);

int tempRootFunc(realtype t, N_Vector y, realtype *rootFunc,
		 void *user_data);

#endif
