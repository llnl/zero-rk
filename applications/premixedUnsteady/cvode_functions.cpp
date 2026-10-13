#include "sparse_matrix.h"
#include "cvode_functions.h"
#include "flame_params.h"

// Main RHS function
int ConstPressureFlame(realtype t,
		       N_Vector y,
		       N_Vector ydot,
		       void *user_data)
{
  FlameParams *params = (FlameParams *)user_data;
  const int num_local_points = params->num_local_points_;
  const int num_states  = params->reactor_->GetNumStates();
  int Nlocal = num_local_points*num_states;

  // MPI calls are in Local, no need for Comm function
  ConstPressureFlameLocal(Nlocal, t, y, ydot, user_data);

  return 0;
}

// RHS function
int ConstPressureFlameLocal(int nlocal,
			    realtype t,
			    N_Vector y,
			    N_Vector ydot,
			    void *user_data)
{
  FlameParams *params = (FlameParams *)user_data;
#ifdef ZERORK_MPI
  double *y_ptr    = NV_DATA_P(y);   // caution: assumes realtype == double
  double *ydot_ptr = NV_DATA_P(ydot); // caution: assumes realtype == double
#else
  double *y_ptr    = NV_DATA_S(y);
  double *ydot_ptr = NV_DATA_S(ydot);
#endif

  const int num_local_points = params->num_local_points_;
  const int num_states  = params->reactor_->GetNumStates();
  const int num_species = params->inlet_mass_fractions_.size();
  const int num_local_states = num_local_points*num_states;
  const int convective_scheme_type = params->convective_scheme_type_;
  int my_pe = params->my_pe_;
  int npes  = params->npes_;

  const double ref_temperature = params->ref_temperature_;

  std::vector<double> enthalpies;
  enthalpies.assign(num_species,0.0);

  std::vector<double> mass_flux;
  mass_flux.assign(num_local_points,0.0);
  for(int j=0; j<num_local_points; ++j) {
    mass_flux[j] = params->mass_flux_[j];
  }

  // Splitting RHS into chemistry, convection, and diffusion terms
  // for readability/future use
  std::vector<double> rhs_chem, rhs_conv, rhs_diff;
  rhs_chem.assign(num_local_points*num_states,0.0);
  rhs_conv.assign(num_local_points*num_states,0.0);
  rhs_diff.assign(num_local_points*num_states,0.0);

  const double wall_term = 4.0*params->nusselt_/
    (params->diameter_*params->diameter_);

  double cp_flux_sum, mass_fraction_sum;
  double relative_volume_j;
  int transport_error;

  double local_sum;
  double local_max;

  double velocity, thermal_diffusivity;

  // set the derivative to zero
  for(int j=0; j<num_local_states; ++j) {
    ydot_ptr[j] = 0.0;

    rhs_chem[j] = 0.0;
    rhs_conv[j] = 0.0;
    rhs_diff[j] = 0.0;
  }

  // compute the constant pressure reactor source term
  // using Zero-RK
  for(int j=0; j<num_local_points; ++j) {
    params->reactor_->GetTimeDerivativeLimiter(t,
                                               &y_ptr[j*num_states],
                                               &params->step_limiter_[0],
                                               &rhs_chem[j*num_states]);
  }

  //--------------------------------------------------------------------------
  // Perform parallel communications
#ifdef ZERORK_MPI
  MPI_Comm comm = params->comm_;
  MPI_Status status;
#endif
  int nover = 2;
  long int dsize = num_states*nover;

  std::vector<double> dz, dzm, inv_dz, inv_dzm;
  dz.assign( num_local_points+(2*nover), 0.0);
  dzm.assign( num_local_points+(2*nover), 0.0);
  inv_dz.assign( num_local_points+(2*nover), 0.0);
  inv_dzm.assign( num_local_points+(2*nover), 0.0);

  // Copy y_ptr data into larger arrays
  for (int j=0; j<num_states*num_local_points; ++j) {
    params->y_ext_[num_states*nover + j] = y_ptr[j];
  }
  for (int j=0; j<num_local_points; ++j) {
    params->mass_flux_ext_[nover+j] = mass_flux[j];
  }

  for (int j=0; j<num_local_points+2*nover; ++j) {
    dz[j] = params->dz_local_[j];
    dzm[j] = params->dzm_local_[j];
    inv_dz[j] = params->inv_dz_local_[j];
    inv_dzm[j] = params->inv_dzm_local_[j];
  }

  // Update ghost cells with send/receive
#ifdef ZERORK_MPI
  int nodeDest = my_pe-1;
  if (nodeDest < 0) nodeDest = npes-1;
  int nodeFrom = my_pe+1;
  if (nodeFrom > npes-1) nodeFrom = 0;
  MPI_Sendrecv(&params->y_ext_[nover*num_states], dsize, PVEC_REAL_MPI_TYPE, nodeDest, 0,
	   &params->y_ext_[num_states*(num_local_points+nover)], dsize, PVEC_REAL_MPI_TYPE, nodeFrom, 0, comm, &status);

  nodeDest = my_pe+1;
  if (nodeDest > npes-1) nodeDest = 0;
  nodeFrom = my_pe-1;
  if (nodeFrom < 0) nodeFrom = npes-1;
  MPI_Sendrecv(&params->y_ext_[num_states*num_local_points], dsize, PVEC_REAL_MPI_TYPE, nodeDest, 0,
	   &params->y_ext_[0], dsize, PVEC_REAL_MPI_TYPE, nodeFrom, 0, comm, &status);
#endif

  // Apply boundary conditions
  // First proc: inlet conditions in ghost cells
  if (my_pe ==0) {
    for(int j=0; j<nover; ++j) {
      for(int k=0; k<num_species; ++k) {
	params->y_ext_[j*num_states + k] = params->inlet_mass_fractions_[k];
      }
      params->y_ext_[(j+1)*num_states-1] = params->inlet_temperature_;
      params->y_ext_[(j+1)*num_states-2] = params->inlet_relative_volume_;
      params->mass_flux_ext_[j] = params->mass_flux_inlet_;
    }
  }

  // Last proc: zero gradient. Copy last domain cell in ghost cells
  if (my_pe == npes-1) {
    for(int j=num_local_points+nover; j<num_local_points+2*nover; ++j) {
      for(int k=0; k<num_species; ++k) {
	params->y_ext_[j*num_states + k] = y_ptr[(num_local_points-1)*num_states+k];
      }
      params->y_ext_[(j+1)*num_states-1] = y_ptr[num_local_points*num_states-1];
      params->y_ext_[(j+1)*num_states-2] = y_ptr[num_local_points*num_states-2];
    }
  }

  //--------------------------------------------------------------------------
  // Compute the interior heat capacity, conductivity, and species mass fluxes.
  for(int j=0; j<num_local_points+1; ++j) {
    int jext = j + nover;

    // compute the upstream mid point state for the transport calculations
    for(int k=0; k<num_species; ++k) {

      // mid point mass fractions
      params->transport_input_.mass_fraction_[k] =
        0.5*(params->y_ext_[jext*num_states+k] + params->y_ext_[(jext-1)*num_states+k]);

      // mid point mass fraction gradient
      params->transport_input_.grad_mass_fraction_[k] = inv_dz[jext]*
	(params->y_ext_[jext*num_states+k] - params->y_ext_[(jext-1)*num_states+k]);
    }

    // mid point temperature
    params->transport_input_.temperature_ = 0.5*ref_temperature*
      (params->y_ext_[(jext+1)*num_states-1] + params->y_ext_[jext*num_states-1]);

    // mid point temperature gradient
    params->transport_input_.grad_temperature_[0] = inv_dz[jext]*ref_temperature*
      (params->y_ext_[(jext+1)*num_states-1] - params->y_ext_[jext*num_states-1]);

    // mixture specific heat at mid point. Species cp will be overwritten
    // for diffusion jacobian only
    params->mixture_specific_heat_mid_[j] =
      params->reactor_->GetMixtureSpecificHeat_Cp(
						  params->transport_input_.temperature_,
						  &params->transport_input_.mass_fraction_[0],
						  &params->species_specific_heats_[0]);

    // Reset species cp
    for(int k=0; k<num_species; k++) {
      params->species_specific_heats_[num_species*j+k] = 0.0;
    }

    // specific heat at grid point j
    if (j != num_local_points) { //not used in derivatives
      params->mixture_specific_heat_[j] =
	params->reactor_->GetMixtureSpecificHeat_Cp(
	    ref_temperature*params->y_ext_[(jext+1)*num_states-1],
	    &params->y_ext_[jext*num_states],
	    &params->species_specific_heats_[num_species*j]);
    }

    // compute the conductivity at the upstream mid point (j-1/2)
    transport_error = params->trans_->GetMixtureConductivity(
      params->transport_input_,
      &params->thermal_conductivity_[j]);
    if(transport_error != transport::NO_ERROR) {
      return transport_error;
    }

    // compute the species mass flux at the upstream mid point
    transport_error = params->trans_->GetSpeciesMassFlux(
      params->transport_input_,
      num_species,
      &params->thermal_conductivity_[j],
      &params->mixture_specific_heat_mid_[j],
      &params->species_mass_flux_[j*num_species],
      &params->species_lewis_numbers_[j*num_species]);
    if(transport_error != transport::NO_ERROR) {
      return transport_error;
    }

  } // for j<num_local_points+1

  //--------------------------------------------------------------------------
  // Compute convective and diffusive terms for species and temperature
  for(int j=0; j<num_local_points; ++j) {
    int jext = j + nover;

    double a=0,b=0,c=0,d=0,e=0;; //coefficients of j+2, j+1, j, j-1, j-2 terms
    if(convective_scheme_type == 0) {
      // First order upwind
      a = 0;
      b = 0;
      c =  inv_dz[jext];
      d = -inv_dz[jext];
      e = 0;
    } else if(convective_scheme_type == 1) {
      // Second order upwind
      if (mass_flux[j] >= 0) {
	// Use points upstream
	a = 0;
	b = 0;
	c = inv_dz[jext] + 1.0/(dz[jext]+dz[jext-1]);
	d = -(dz[jext]+dz[jext-1])/(dz[jext]*dz[jext-1]);
	e = dz[jext]/dz[jext-1]/(dz[jext]+dz[jext-1]);
      } else {
	// Use points downstream
	a = -dz[jext+1]/dz[jext+2]/(dz[jext+1]+dz[jext+2]);
	b = inv_dz[jext+1] + inv_dz[jext+2];
	c = -inv_dz[jext+1] - 1.0/(dz[jext+1]+dz[jext+2]);
	d = 0;
	e = 0;
      }
    } else if(convective_scheme_type == 2) {
      // Centered
      a = 0;
      b = dz[jext]/dz[jext+1]/(dz[jext]+dz[jext+1]);
      c = (dz[jext+1]-dz[jext])/dz[jext+1]/dz[jext];
      d = -dz[jext+1]/dz[jext]/(dz[jext]+dz[jext+1]);
      e = 0;
    } else {
      cerr << "Undefined convective scheme \n";
#ifdef ZERORK_MPI
      MPI_Finalize();
#endif
      exit(0);
    }

    relative_volume_j   = params->y_ext_[jext*num_states+num_species];

    // compute the species mass fraction advection and diffusion
    for(int k=0; k<num_species; ++k) {

      rhs_conv[j*num_states+k] -= relative_volume_j*
	(a*params->y_ext_[(jext+2)*num_states+k] +
	 b*params->y_ext_[(jext+1)*num_states+k] +
	 c*params->y_ext_[jext*num_states+k] +
	 d*params->y_ext_[(jext-1)*num_states+k] +
	 e*params->y_ext_[(jext-2)*num_states+k]);

      rhs_diff[j*num_states+k] -= relative_volume_j*inv_dzm[jext]*
	( params->species_mass_flux_[num_species*(j+1)+k]
	  -params->species_mass_flux_[num_species*j+k]);
    }

    // compute the species specific heat diffusive flux sum
    cp_flux_sum = 0.0;
    for(int k=0; k<num_species; ++k) {
      cp_flux_sum += params->species_specific_heats_[num_species*j+k]*
	0.5*(params->species_mass_flux_[num_species*j+k]+
	     params->species_mass_flux_[num_species*(j+1)+k]);
    }

    // Compute the temperature advection
    rhs_conv[(j+1)*num_states-1] -= relative_volume_j*
      (a*params->y_ext_[(jext+3)*num_states-1] +
       b*params->y_ext_[(jext+2)*num_states-1] +
       c*params->y_ext_[(jext+1)*num_states-1] +
       d*params->y_ext_[jext*num_states-1] +
       e*params->y_ext_[(jext-1)*num_states-1]);

    rhs_diff[(j+1)*num_states-1] -= relative_volume_j*
      cp_flux_sum/params->mixture_specific_heat_[j]*
      (a*params->y_ext_[(jext+3)*num_states-1] +
       b*params->y_ext_[(jext+2)*num_states-1] +
       c*params->y_ext_[(jext+1)*num_states-1] +
       d*params->y_ext_[jext*num_states-1] +
       e*params->y_ext_[(jext-1)*num_states-1]);

    // Add the thermal conductivity contribution to dT[j]/dt
    rhs_diff[(j+1)*num_states-1] +=
      (relative_volume_j*inv_dzm[jext]/params->mixture_specific_heat_[j])*
      (params->thermal_conductivity_[j+1]*inv_dz[jext+1]*
       (params->y_ext_[(jext+2)*num_states-1] - params->y_ext_[(jext+1)*num_states-1])
       -params->thermal_conductivity_[j]*inv_dz[jext]*
       (params->y_ext_[(jext+1)*num_states-1] - params->y_ext_[jext*num_states-1]));

  } // for(int j=1; j<num_local_points; ++j) // loop computing rhs

  // -------------------------------------------------------------------------
  // Compute the wall heat transfer contribution
  for(int j=0; j<num_local_points; ++j) {
    int jglobal = j + my_pe*num_local_points;
    relative_volume_j = y_ptr[j*num_states+num_species];
    // wall_term = 4*Nu/d^2
    // grid point thermal conductivity is the average of the conductivity at
    // the upstream and downstream interfaces
    rhs_diff[(j+1)*num_states-1] -= 0.5*wall_term*
      (params->thermal_conductivity_[j]+params->thermal_conductivity_[j+1])*
      (relative_volume_j/params->mixture_specific_heat_[j])*
      (y_ptr[(j+1)*num_states-1] - params->wall_temperature_[jglobal]);
  }

  //--------------------------------------------------------------------------
  // Compute mass flux -- Integrate continuity equation
  // dmdot/dx = -drho/dt -> mdot = int(-drho/dt, dx)
  const double RuTref_p = params->reactor_->GetGasConstant()*
    params->ref_temperature_/params->parser_->pressure();
  // Duplicate the continuity integration for parallel simulations
  for(int l=0; l<npes; ++l) {

#ifdef ZERORK_MPI
    // Send/receive only if necessary
    if (my_pe != npes-1 && my_pe >= l-1) {
      MPI_Send(&params->mass_flux_ext_[num_local_points], nover, PVEC_REAL_MPI_TYPE, my_pe+1, 0, comm);
    }
    if (my_pe !=0 && my_pe >=l) {
      MPI_Recv(&params->mass_flux_ext_[0], nover, PVEC_REAL_MPI_TYPE, my_pe-1, 0, comm, &status);
    }
#endif

    // Compute mass flux if necessary
    if(my_pe >= l) {
      for(int j=0; j<num_local_points; ++j) {
	int jext = j + nover;

	int rvol_id = j*num_states+num_species; // relative volume index of pt j
	int temp_id = rvol_id+1;                // temperature index of pt j

	double alpha = 0.0;
	double beta = 0.0;

	for(int k=0; k<num_species; ++k) {
	  alpha += params->inv_molecular_mass_[k]*rhs_conv[j*num_states+k];
	  beta += params->inv_molecular_mass_[k]*(rhs_chem[j*num_states+k]+rhs_diff[j*num_states+k]);
	}

	alpha *= RuTref_p*y_ptr[temp_id]/y_ptr[rvol_id]/y_ptr[rvol_id];
	beta  *= RuTref_p*y_ptr[temp_id]/y_ptr[rvol_id]/y_ptr[rvol_id];

	alpha += rhs_conv[temp_id]/y_ptr[temp_id]/y_ptr[rvol_id];
	beta += (rhs_chem[temp_id]+rhs_diff[temp_id])/y_ptr[temp_id]/y_ptr[rvol_id];

	double a,b,c; //coefficients of j, j-1, j-2 terms
	if(convective_scheme_type == 0) {
	  // First order upwind
	  a =  inv_dz[jext];
	  b = -inv_dz[jext];
	  c = 0;
	} else {
	  // Second order upwind
	  a = inv_dz[jext] + 1.0/(dz[jext]+dz[jext-1]);
	  b = -(dz[jext]+dz[jext-1])/(dz[jext]*dz[jext-1]);
	  c = dz[jext]/dz[jext-1]/(dz[jext]+dz[jext-1]);
	}

	params->mass_flux_ext_[jext] = (beta - b*params->mass_flux_ext_[jext-1] - c*params->mass_flux_ext_[jext-2])/(a - alpha);

      } // for j<num_local_points
    }	//if my_pe >= l
  } // for l<npes

  //  Copy into params
  for(int j=0; j<num_local_points; ++j) {
    int jext = j + nover;
    mass_flux[j] = params->mass_flux_ext_[jext];
    params->mass_flux_[j] = params->mass_flux_ext_[jext];
  }

  // Compute final convective rhs (including mass flux)
  for(int j=0; j<num_local_points; ++j) {
    // Species
    for(int k=0; k<num_species; ++k) {
      rhs_conv[j*num_states+k] = rhs_conv[j*num_states+k]*mass_flux[j];
    }
    // T
    rhs_conv[j*num_states+num_species+1] = rhs_conv[j*num_states+num_species+1]*mass_flux[j];
  }

  // -------------------------------------------------------------------------
  // Compute the rate of change of the relative volume using the ideal
  // gas equation of state, and the FINAL derivatives of temperature and
  // mass fractions.
  //
  // dv/dt = v/T * dT/dt + RuT/p * \sum_i (1/mw[i] * dy[i]/dt)
  // dv/dt   current units [m^3/kg/s]
  for(int j=0; j<num_local_points; ++j) {
    int rvol_id = j*num_states+num_species; // relative volume index of pt j
    int temp_id = rvol_id+1;                // temperature index of pt j

    mass_fraction_sum = 0.0;
    for(int k=0; k<num_species; ++k) {
      ydot_ptr[j*num_states+k] = rhs_conv[j*num_states+k] +
                                 rhs_chem[j*num_states+k] +
                                 rhs_diff[j*num_states+k];

      mass_fraction_sum += params->inv_molecular_mass_[k]*ydot_ptr[j*num_states+k];
    }
    ydot_ptr[temp_id] = rhs_conv[temp_id] + rhs_chem[temp_id] + rhs_diff[temp_id];

    ydot_ptr[rvol_id] = y_ptr[rvol_id]*ydot_ptr[temp_id]/y_ptr[temp_id] +
      RuTref_p*y_ptr[temp_id]*mass_fraction_sum;
  }

  // -------------------------------------------------------------------------
  // For output to the screen/logfile
  // Compute global continuity error = m_in - m_out - int(drhodt)
  double sum_drhodt = 0.0;
  local_sum = 0.0;
  for(int j=0; j<num_local_points; ++j) {
    int jext = j + nover;
    local_sum -= ydot_ptr[j*num_states+num_species]*dzm[jext]/
      y_ptr[j*num_states + num_species]/y_ptr[j*num_states + num_species];
  }
#ifdef ZERORK_MPI
  MPI_Allreduce(&local_sum,&sum_drhodt,1,PVEC_REAL_MPI_TYPE,MPI_SUM,comm);
#else
  sum_drhodt = local_sum;
#endif

  double mass_flux_outlet;
  double tmp = 0;
  if (my_pe == npes-1) {
    tmp = mass_flux[num_local_points-1];
  }
#ifdef ZERORK_MPI
  MPI_Allreduce(&tmp,&mass_flux_outlet,1,PVEC_REAL_MPI_TYPE,MPI_SUM,comm);
#else
  mass_flux_outlet = tmp;
#endif
  params->mass_change_ = (params->mass_flux_inlet_ - mass_flux_outlet - sum_drhodt)/params->mass_flux_inlet_;

  // Compute laminar flame speed = int(omega_F)/rho_u/YF_u
  double sum_omega_F = 0.0;
  int num_fuel_species = params->fuel_species_id_.size();
  local_sum = 0.0;
  for(int j=0; j<num_local_points; ++j) {
    int jext = j + nover;
    for(int k=0; k<num_fuel_species; ++k) {
      local_sum -= rhs_chem[j*num_states + params->fuel_species_id_[k]]*
	dzm[jext]/y_ptr[j*num_states + num_species];
    }
  }
#ifdef ZERORK_MPI
  MPI_Allreduce(&local_sum,&sum_omega_F,1,PVEC_REAL_MPI_TYPE,MPI_SUM,comm);
#else
  sum_omega_F = local_sum;
#endif

  double sum_inlet_fuel_mass_fractions = 0.0;
  for(int k=0; k<num_fuel_species; ++k) {
    sum_inlet_fuel_mass_fractions += params->inlet_mass_fractions_[params->fuel_species_id_[k]];
  }
  sum_omega_F /= sum_inlet_fuel_mass_fractions/params->inlet_relative_volume_;
  params->flame_speed_ = sum_omega_F;

  // initialize the max variables used for explicit time step information
  // compute the max velocity from the mass flux and relative volume stored
  // in the state vector
  local_max = 0.0;
  for(int j=0; j<num_local_points; ++j) {
    velocity = fabs(mass_flux[j]*y_ptr[(j+1)*num_states-2]);
    if(velocity > local_max) {
      local_max = velocity;
    }
  }
#ifdef ZERORKM_MPI
  MPI_Allreduce(&local_max,&params->max_velocity_,1,PVEC_REAL_MPI_TYPE,MPI_MAX,comm);
#else
  params->max_velocity_ = local_max;
#endif

  double local_temperature;
  local_max = 0.0;
  for(int j=0; j<num_local_points; ++j) {
    int jext = j + nover;
    local_temperature = ref_temperature*params->y_ext_[(jext+1)*num_states-1];
    if(local_temperature > local_max) {
      local_max = local_temperature;
    }
  }
#ifdef ZERORK_MPI
  MPI_Allreduce(&local_max,&params->max_temperature_,1,PVEC_REAL_MPI_TYPE,MPI_MAX,comm);
#else
  params->max_temperature_ = local_max;
#endif

  double gradT;
  local_max = 0.0;
  for(int j=0; j<num_local_points; ++j) {
    int jext = j + nover;
    gradT = fabs(inv_dz[jext]*ref_temperature*
		 (params->y_ext_[(jext+1)*num_states-1] - params->y_ext_[jext*num_states-1]));
    if (gradT > local_max) {
      local_max = gradT;
    }
  }
#ifdef ZERORK_MPI
  MPI_Allreduce(&local_max,&params->flame_thickness_,1,PVEC_REAL_MPI_TYPE,MPI_MAX,comm);
#else
  params->flame_thickness_ = local_max;
#endif

  params->flame_thickness_ = (params->max_temperature_-params->inlet_temperature_)/
    params->flame_thickness_;

  // Compute alternate flame thickness definition lF = K/rho/cp/SL (alpha = K/rho/cp)
  // using inlet values
  if(my_pe == 0) {
    params->flame_thickness_alpha_ = params->thermal_conductivity_[0]/
      params->inlet_relative_volume_/params->mixture_specific_heat_[0]/params->flame_speed_;
  }
#ifdef ZERORK_MPI
  MPI_Bcast(&params->flame_thickness_alpha_, 1, MPI_DOUBLE, 0, comm);
#endif

  // compute the max thermal diffusivity using the average value of the
  // conductivity and the up and downstream interfaces
  local_max = 0.0;
  for(int j=0; j<num_local_points; ++j) {
    thermal_diffusivity =
      fabs(0.5*(params->thermal_conductivity_[j]+
                params->thermal_conductivity_[j+1])*
           y_ptr[(j+1)*num_states-2]/params->mixture_specific_heat_[j]);
    if(thermal_diffusivity > local_max) {
      local_max = thermal_diffusivity;
    }
  }
#ifdef ZERORK_MPI
  MPI_Allreduce(&local_max,&params->max_thermal_diffusivity_,1,PVEC_REAL_MPI_TYPE,MPI_MAX,comm);
#else
  params->max_thermal_diffusivity_ = local_max;
#endif
  return 0;
}

#if defined SUNDIALS2
int ReactorPreconditionerChemistrySetup(realtype t,      // [in] ODE system time
                                        N_Vector y,      // [in] ODE state vector
                                        N_Vector ydot,   // [in] ODE state derivative
                                        booleantype jok,
                                        booleantype *new_j,
                                        realtype gamma,
                                        void *user_data, // [in/out]
                                        N_Vector tmp1,
                                        N_Vector tmp2,
                                        N_Vector tmp3)
{
#elif defined SUNDIALS3 || defined SUNDIALS4
int ReactorPreconditionerChemistrySetup(realtype t,      // [in] ODE system time
                               N_Vector y,      // [in] ODE state vector
                               N_Vector ydot,   // [in] ODE state derivative
                               booleantype jok,
			       booleantype *new_j,
			       realtype gamma,
		               void *user_data) // [in/out]
{
#endif
  FlameParams *params    = (FlameParams *)user_data;
  const int num_local_points   = params->num_local_points_;
  const int num_states   = params->reactor_->GetNumStates();
  const int num_nonzeros = params->reactor_->GetJacobianSize();
#ifdef ZERORK_MPI
  double *y_ptr          = NV_DATA_P(y); //_S // caution: assumes realtype == double
#else
  double *y_ptr          = NV_DATA_S(y);
#endif
  int error_flag = 0;

  if(params->store_jacobian_) {

    if(!jok) {
      // The Jacobian is not okay, need to recompute
      params->saved_jacobian_.assign(num_nonzeros*num_local_points, 0.0);
      for(int j=0; j<num_local_points; ++j) {
        params->reactor_->GetJacobianLimiter(t,
					     &y_ptr[j*num_states],
					     &params->step_limiter_[0],
					     &params->saved_jacobian_[j*num_nonzeros]);
      } // for j<num_local_points
      (*new_j) = true;
    } else {
      (*new_j) = false;
    } // if/else jok


    for(int j=0; j<num_local_points; ++j) {
      // compute I - gamma*J
      // this could be updated with blas routines
      for(int k=0; k<num_nonzeros; ++k) {
        params->reactor_jacobian_[k] =
          -gamma*params->saved_jacobian_[j*num_nonzeros+k];
      }
      for(int k=0; k<num_states; ++k) {
        params->reactor_jacobian_[params->diagonal_id_[k]] += 1.0;
      }
      // factor the numerical jacobian
      if(params->sparse_matrix_[j]->IsFirstFactor()) {
        error_flag =
          params->sparse_matrix_[j]->FactorNewPatternCCS(num_nonzeros,
						 &params->row_id_[0],
						 &params->column_sum_[0],
						 &params->reactor_jacobian_[0]);
      } else {
        error_flag =
          params->sparse_matrix_[j]->FactorSamePattern(
                                                 &params->reactor_jacobian_[0]);
      }
      if(error_flag != 0) {
        params->logger_->PrintF(
          "# DEBUG: At t = %.18g [s],\n"
          "#        grid point %d (z = %.18g [m]) reactor produced a\n"
          "#        sparse matrix error flag = %d\n", t, j, params->z_[j],  error_flag);
        return error_flag;
      }

     } // for(int j=0; j<num_local_points; ++j)

  } else {
    // recompute and factor the Jacobian, there is no saved data
    // TODO: offer option for the fake update
    for(int j=0; j<num_local_points; ++j) {
      params->reactor_->GetJacobianLimiter(t,
					   &y_ptr[j*num_states],
					   &params->step_limiter_[0],
					   &params->reactor_jacobian_[0]);

      // compute I - gamma*J
      // this could be updated with blas routines
      for(int k=0; k<num_nonzeros; ++k) {
        params->reactor_jacobian_[k] *= -gamma;
      }
      for(int k=0; k<num_states; ++k) {
        params->reactor_jacobian_[params->diagonal_id_[k]] += 1.0;
      }
      // factor the numerical jacobian
      if(params->sparse_matrix_[j]->IsFirstFactor()) {
        error_flag =
          params->sparse_matrix_[j]->FactorNewPatternCCS(num_nonzeros,
						 &params->row_id_[0],
						 &params->column_sum_[0],
						 &params->reactor_jacobian_[0]);
      } else {
        error_flag =
          params->sparse_matrix_[j]->FactorSamePattern(
                                                 &params->reactor_jacobian_[0]);
      }
      if(error_flag != 0) {
        params->logger_->PrintF(
          "# DEBUG: At t = %.18g [s],\n"
          "#        grid point %d (z = %.18g [m]) reactor produced a\n"
          "#        sparse matrix error flag = %d\n", t, j, params->z_[j], error_flag);
        return error_flag;
      }

     } // for(int j=0; j<num_points; ++j)

    (*new_j) = true; // without saving, it is always a new Jacobian

  } // if(params->store_jacobian_) else

  return 0;
}

#if defined SUNDIALS2
int ReactorPreconditionerChemistrySolve(realtype t,      // [in] ODE system time
                                        N_Vector y,      // [in] ODE state vector
                                        N_Vector ydot,   // [in] ODE state derivative
                                        N_Vector r,      // [in] jacobian rhs
                                        N_Vector z,      // [out]
                                        realtype gamma,
                                        realtype delta,
                                        int lr,
                                        void *user_data, // [in/out]
                                        N_Vector tmp)
{
#elif defined SUNDIALS3 || defined SUNDIALS4
int ReactorPreconditionerChemistrySolve(realtype t,      // [in] ODE system time
                               N_Vector y,      // [in] ODE state vector
                               N_Vector ydot,   // [in] ODE state derivative
                               N_Vector r,      // [in] jacobian rhs
                               N_Vector z,      // [out]
			       realtype gamma,
			       realtype delta,
			       int lr,
		               void *user_data)    // [in/out]
{
#endif
  FlameParams *params = (FlameParams *)user_data;
  const int num_local_points  = params->num_local_points_;
  const int num_states  = params->reactor_->GetNumStates();
#ifdef ZERORK_MPI
  double *rhs         = NV_DATA_P(r);  // pointers to data array for N_Vector
  double *solution    = NV_DATA_P(z);  // pointers to data array for N_Vector
#else
  double *rhs         = NV_DATA_S(r);  // pointers to data array for N_Vector
  double *solution    = NV_DATA_S(z);  // pointers to data array for N_Vector
#endif
  int error_flag = 0;
  int start_id=0;

  for(int j=0; j<num_local_points; ++j) {
    error_flag = params->sparse_matrix_[j]->Solve(&rhs[start_id],
                                                  &solution[start_id]);
    start_id += num_states;
    if(error_flag != 0) {
      return error_flag;
    }
  }

  return error_flag;
}
