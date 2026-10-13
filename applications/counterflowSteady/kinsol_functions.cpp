#include "kinsol_functions.h"
#include "flame_params.h"

extern "C" void dgbtrf_(int* dim1, int* dim2, int* nu, int* nl, double* a, int* lda, int* ipiv, int* info);
extern "C" void dgbtrs_(char *TRANS, int *N, int *NRHS, int* nu, int* nl, double *A, int *LDA, int *IPIV, double *B, int *LDB, int *INFO);

static double FindMaximumParallel(const int num_points,
                                  const double f[],
                                  int *j_at_max,
                                  MPI_Comm comm);

static double FindMinimumParallel(const int num_points,
                                  const double f[],
                                  int *j_at_min,
                                  MPI_Comm comm);


// Main RHS function
int ConstPressureFlame(N_Vector y,
		       N_Vector ydot,
		       void *user_data)
{
  FlameParams *params = (FlameParams *)user_data;
  const int num_local_points = params->num_local_points_;
  const int num_states  = params->reactor_->GetNumStates();
  int Nlocal = num_local_points*num_states;

  // Parallel communications
  ConstPressureFlameComm(Nlocal, y, user_data);
  // RHS calculations
  ConstPressureFlameLocal(Nlocal, y, ydot, user_data);

  return 0;
}

int ConstPressureFlameComm(int nlocal,
                           N_Vector y,
                           void *user_data)
{
  FlameParams *params = (FlameParams *)user_data;
  double *y_ptr    = NV_DATA_P(y);
  const int num_local_points = params->num_local_points_;
  const int num_states  = params->reactor_->GetNumStates();
  const int num_species = params->reactor_->GetNumSpecies();
  int my_pe = params->my_pe_;
  int npes  = params->npes_;

  MPI_Comm comm = params->comm_;
  MPI_Status status;
  const int nover = params->nover_;
  long int dsize = num_states*nover;
  long int dsize_rel = nover;

  // Compute relative volume
  const double RuTref_p = params->reactor_->GetGasConstant()*
    params->ref_temperature_/params->parser_->pressure();

  for(int j=0; j<num_local_points; ++j) {
    int temp_id = j*num_states+num_species + 1;
    double mass_fraction_sum = 0.0;
    for(int k=0; k<num_species; ++k)
      mass_fraction_sum += params->inv_molecular_mass_[k]*y_ptr[j*num_states+k];
    params->rel_vol_[j] = RuTref_p*y_ptr[temp_id]*mass_fraction_sum;
  }

  // Copy y_ptr data into larger arrays
  for (int j=0; j<num_states*num_local_points; ++j)
    params->y_ext_[num_states*nover + j] = y_ptr[j];

  for (int j=0; j<num_local_points; ++j)
    params->rel_vol_ext_[nover+j] = params->rel_vol_[j];

  // MPI sendrecv
  int nodeDest = my_pe-1;
  if (nodeDest < 0) nodeDest = npes-1;
  int nodeFrom = my_pe+1;
  if (nodeFrom > npes-1) nodeFrom = 0;
  MPI_Sendrecv(&params->y_ext_[nover*num_states], dsize, PVEC_REAL_MPI_TYPE, nodeDest, 0,
               &params->y_ext_[num_states*(num_local_points+nover)], dsize,
               PVEC_REAL_MPI_TYPE, nodeFrom, 0, comm, &status);
  MPI_Sendrecv(&params->rel_vol_ext_[nover], dsize_rel, PVEC_REAL_MPI_TYPE, nodeDest, 0,
               &params->rel_vol_ext_[num_local_points+nover], dsize_rel,
               PVEC_REAL_MPI_TYPE, nodeFrom, 0, comm, &status);

  nodeDest = my_pe+1;
  if (nodeDest > npes-1) nodeDest = 0;
  nodeFrom = my_pe-1;
  if (nodeFrom < 0) nodeFrom = npes-1;
  MPI_Sendrecv(&params->y_ext_[num_states*num_local_points], dsize,
               PVEC_REAL_MPI_TYPE, nodeDest,0, &params->y_ext_[0], dsize,
               PVEC_REAL_MPI_TYPE, nodeFrom, 0, comm, &status);
  MPI_Sendrecv(&params->rel_vol_ext_[num_local_points], dsize_rel,
               PVEC_REAL_MPI_TYPE, nodeDest, 0, &params->rel_vol_ext_[0],
               dsize_rel, PVEC_REAL_MPI_TYPE, nodeFrom, 0, comm, &status);

  // Copy mass flux variable into mass_flux_ext
  for (int j=0; j<num_local_points+2*nover; ++j)
    params->mass_flux_ext_[j] = params->y_ext_[j*num_states + num_species];

  return 0;
}

// RHS function
int ConstPressureFlameLocal(int nlocal,
			    N_Vector y,
			    N_Vector ydot,
			    void *user_data)
{
  FlameParams *params = (FlameParams *)user_data;
  double *y_ptr    = NV_DATA_P(y);   // caution: assumes realtype == double
  double *ydot_ptr = NV_DATA_P(ydot); // caution: assumes realtype == double

  const int num_local_points = params->num_local_points_;
  const int num_total_points = params->z_.size();
  const int num_states  = params->reactor_->GetNumStates();
  const int num_species = params->reactor_->GetNumSpecies();
  const int num_local_states = num_local_points*num_states;
  const int convective_scheme_type = params->convective_scheme_type_;
  MPI_Comm comm = params->comm_;
  int my_pe = params->my_pe_;
  int npes  = params->npes_;
  int nover = params->nover_;
  long int dsize;

  const double ref_temperature = params->ref_temperature_;
  const double ref_momentum = params->ref_momentum_;
  const bool finite_separation = params->parser_->finite_separation();
  const bool fixed_temperature = params->parser_->fixed_temperature();

  std::vector<double> enthalpies;
  enthalpies.assign(num_species,0.0);

  // Splitting RHS into chemistry, convection, and diffusion terms
  // for readability/future use
  std::vector<double> rhs_chem, rhs_conv, rhs_diff;
  rhs_chem.assign(num_local_states,0.0);
  rhs_conv.assign(num_local_states,0.0);
  rhs_diff.assign(num_local_states,0.0);

  double cp_flux_sum, mass_fraction_sum;
  double relative_volume_j, relative_volume_jp, relative_volume_jm;
  int transport_error;

  double local_sum;
  double local_max;

  double thermal_diffusivity, continuity_error;

  // set the residual to zero
  for(int j=0; j<num_local_states; ++j) {
    ydot_ptr[j] = 0.0;
  }

  // Compute relative volume
  const double RuTref_p = params->reactor_->GetGasConstant()*
    params->ref_temperature_/params->parser_->pressure();

  for(int j=0; j<num_local_points; ++j) {
    int temp_id = j*num_states+num_species + 1;
    mass_fraction_sum = 0.0;
    for(int k=0; k<num_species; ++k)
      mass_fraction_sum += params->inv_molecular_mass_[k]*y_ptr[j*num_states+k];
    params->rel_vol_[j] = RuTref_p*y_ptr[temp_id]*mass_fraction_sum;
  }

  if(!finite_separation) {
    // Update mass flux BCs
    // Left BC
    if(my_pe == 0) {
      int j=0;
      params->mass_flux_fuel_ = y_ptr[j*num_states+num_species] +
        (1.0+params->simulation_type_)*params->y_ext_[(j+nover-1)*num_states+num_species+2]*
        ref_momentum/params->rel_vol_ext_[j+nover-1]*params->dz_[j];
    }
    MPI_Bcast(&params->mass_flux_fuel_, 1, MPI_DOUBLE, 0, comm);

    // Right BC
    if(my_pe == npes-1) {
      int j=num_local_points-1;
      params->mass_flux_oxidizer_ = y_ptr[j*num_states+num_species] -
        (1.0+params->simulation_type_)*y_ptr[j*num_states+num_species+2]*
        ref_momentum/params->rel_vol_[j]*params->dz_[num_total_points];
    }
    int nodeFrom = npes-1;
    MPI_Bcast(&params->mass_flux_oxidizer_, 1, MPI_DOUBLE, nodeFrom, comm);
  }

  //--------------------------------------------------------------------------
  // Perform parallel communications
  MPI_Status status;
  dsize = num_states*nover;

  std::vector<double> dz, dzm, inv_dz, inv_dzm;
  dz.assign( num_local_points+(2*nover), 0.0);
  dzm.assign( num_local_points+(2*nover), 0.0);
  inv_dz.assign( num_local_points+(2*nover), 0.0);
  inv_dzm.assign( num_local_points+(2*nover), 0.0);

  // Copy y_ptr data into larger arrays
  for (int j=0; j<num_states*num_local_points; ++j) {
    params->y_ext_[num_states*nover + j] = y_ptr[j];
  }

  for (int j=0; j<num_local_points; ++j)
    params->rel_vol_ext_[nover+j] = params->rel_vol_[j];

  for (int j=0; j<num_local_points+2*nover; ++j) {
    dz[j] = params->dz_local_[j];
    dzm[j] = params->dzm_local_[j];
    inv_dz[j] = params->inv_dz_local_[j];
    inv_dzm[j] = params->inv_dzm_local_[j];
  }

  // Apply boundary conditions
  // First proc: fuel conditions in ghost cells
  if (my_pe == 0) {
    for(int j=0; j<nover; ++j) {
      if(params->flame_type_ == 0) {
        for(int k=0; k<num_species; ++k) {
          params->y_ext_[j*num_states + k] = params->fuel_mass_fractions_[k];
        }
        params->rel_vol_ext_[j]   = params->fuel_relative_volume_;
      } else if (params->flame_type_ == 1 || params->flame_type_ == 2) {
        for(int k=0; k<num_species; ++k) {
          params->y_ext_[j*num_states + k] = params->inlet_mass_fractions_[k];
        }
        params->rel_vol_ext_[j] = params->inlet_relative_volume_;
      }
      params->y_ext_[j*num_states+num_species  ] = params->mass_flux_fuel_;
      params->y_ext_[j*num_states+num_species+1] = params->fuel_temperature_;
      if(finite_separation) { // U=0, zero gradient on P
        params->y_ext_[j*num_states+num_species+2] = 0.0;
        params->P_left_ = params->y_ext_[nover*num_states+num_species+3];
        params->y_ext_[j*num_states+num_species+3] = params->P_left_;
      } else { // Zero gradient
        params->y_ext_[j*num_states+num_species+2] =
          params->y_ext_[nover*num_states+num_species+2];
      }
    }
  }
  MPI_Bcast(&params->P_left_, 1, MPI_DOUBLE, 0, comm);

  // Last proc: oxidizer conditions in ghost cells
  if (my_pe == npes-1) {
    for(int j=num_local_points+nover; j<num_local_points+2*nover; ++j) {
      if(params->flame_type_ == 1) { //zero gradient on Y, 1/rho, T, G
        for(int k=0; k<num_species; ++k) {
          params->oxidizer_mass_fractions_[k] =
            params->y_ext_[(num_local_points+nover-1)*num_states+k];
        }
        params->oxidizer_relative_volume_ =
          params->rel_vol_ext_[(num_local_points+nover-1)];
        params->oxidizer_temperature_ =
          params->y_ext_[(num_local_points+nover-1)*num_states+num_species+1];
      }
      for(int k=0; k<num_species; ++k) {
        params->y_ext_[j*num_states + k] = params->oxidizer_mass_fractions_[k];
      }
      params->rel_vol_ext_[j] = params->oxidizer_relative_volume_;
      params->y_ext_[j*num_states+num_species+1] = params->oxidizer_temperature_;
      if(finite_separation) {
        if(params->flame_type_ == 0 || params->flame_type_ == 2) {
          params->y_ext_[j*num_states+num_species+2] = 0.0;
        } else if (params->flame_type_ == 1) {
          params->G_right_ = params->y_ext_[(num_local_points+nover-1)*num_states+num_species+2];
          params->y_ext_[j*num_states+num_species+2] = params->G_right_;
        }
        params->P_right_ = params->y_ext_[(num_local_points+nover-1)*num_states+num_species+3];
        params->y_ext_[j*num_states+num_species+3] = params->P_right_;

      } else {
        params->y_ext_[j*num_states+num_species+2] =
          params->y_ext_[(num_local_points+nover-1)*num_states+num_species+2];//dG/dx=0
      }
      params->y_ext_[j*num_states+num_species] = params->mass_flux_oxidizer_;
    }
  }
  MPI_Bcast(&params->P_right_, 1, MPI_DOUBLE, npes-1, comm);
  if(params->flame_type_ == 1) {
    MPI_Bcast(&params->G_right_, 1, MPI_DOUBLE, npes-1, comm);
    MPI_Bcast(&params->oxidizer_temperature_, 1, MPI_DOUBLE, npes-1, comm);
    MPI_Bcast(&params->oxidizer_relative_volume_, 1, MPI_DOUBLE, npes-1, comm);
    MPI_Bcast(&params->oxidizer_mass_fractions_[0], num_species, MPI_DOUBLE, npes-1, comm);
  }
  //--------------------------------------------------------------------------

  // compute the constant pressure reactor source term
  // using Zero-RK
  for(int j=0; j<num_local_points; ++j) {
    params->reactor_->GetTimeDerivativeSteady(&y_ptr[j*num_states],
                                              &params->step_limiter_[0],
                                              &rhs_chem[j*num_states]);
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
      (params->y_ext_[jext*num_states+num_species+1] +
       params->y_ext_[(jext-1)*num_states+num_species+1]);

    // mid point temperature gradient
    params->transport_input_.grad_temperature_[0] = inv_dz[jext]*ref_temperature*
      (params->y_ext_[jext*num_states+num_species+1] -
       params->y_ext_[(jext-1)*num_states+num_species+1]);

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
	    ref_temperature*params->y_ext_[jext*num_states+num_species+1],
	    &params->y_ext_[jext*num_states],
	    &params->species_specific_heats_[num_species*j]);
    }

    //mixture molecular mass at mid point
    // for frozen thermo only
    double mass_fraction_weight_sum = 0.0;
    for(int k=0; k<num_species; ++k) {
      mass_fraction_weight_sum +=
        params->inv_molecular_mass_[k]*params->transport_input_.mass_fraction_[k];
    }
    params->molecular_mass_mix_mid_[j] = 1.0/mass_fraction_weight_sum;


    // compute the conductivity at the upstream mid point (j-1/2)
    transport_error = params->trans_->GetMixtureConductivity(
      params->transport_input_,
      &params->thermal_conductivity_[j]);
    if(transport_error != transport::NO_ERROR) {
      return transport_error;
    }

    // compute the viscosity at the upstream mid point (j-1/2)
    transport_error = params->trans_->GetMixtureViscosity(
      params->transport_input_,
      &params->mixture_viscosity_[j]);
    if(transport_error != transport::NO_ERROR) {
      return transport_error;
    }

    // compute the species mass flux at the upstream mid point
    // user can choose whether to use the diffusion correction
    if(params->parser_->diffusion_correction()) {
      transport_error = params->trans_->GetSpeciesMassFlux(
        params->transport_input_,
        num_species,
        &params->thermal_conductivity_[j],
        &params->mixture_specific_heat_mid_[j],
        &params->species_mass_flux_[j*num_species],
        &params->species_lewis_numbers_[j*num_species]);
    } else {
      transport_error = params->trans_->GetSpeciesMassFluxFrozenThermo(
        params->transport_input_,
        num_species,
        &params->thermal_conductivity_[j],
        &params->mixture_specific_heat_mid_[j],
        &params->species_mass_flux_[j*num_species],
        &params->species_lewis_numbers_[j*num_species]);
    }
    if(transport_error != transport::NO_ERROR) {
      return transport_error;
    }

  } // for j<num_local_points+1

  //--------------------------------------------------------------------------
  // Compute convective and diffusive terms for species, temperature, and momentum
  for(int j=0; j<num_local_points; ++j) {
    int jext = j + nover;
    int jglobal = j + my_pe*num_local_points;

    relative_volume_j  = params->rel_vol_ext_[jext];
    relative_volume_jp = params->rel_vol_ext_[jext+1];
    relative_volume_jm = params->rel_vol_ext_[jext-1];

    double a=0,b=0,c=0,d=0,e=0;; //coefficients of j+2, j+1, j, j-1, j-2 terms
    if(convective_scheme_type == 0) {
      // First order upwind
      if(params->y_ext_[jext*num_states+num_species]*relative_volume_j > 0) {
        a = 0;
        b = 0;
        c =  inv_dz[jext];
        d = -inv_dz[jext];
        e = 0;
      } else {
        a = 0;
        b =  inv_dz[jext+1];
        c = -inv_dz[jext+1];
        d = 0;
        e = 0;
      }
    } else if(convective_scheme_type == 1) {
      // Second order upwind
      if (params->y_ext_[jext*num_states+num_species]*relative_volume_j > 0) {
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
      MPI_Finalize();
      exit(0);
    }

    // compute the species mass fraction advection and diffusion
    for(int k=0; k<num_species; ++k) {

      rhs_conv[j*num_states+k] -= relative_volume_j*
	(a*params->y_ext_[(jext+2)*num_states+k] +
	 b*params->y_ext_[(jext+1)*num_states+k] +
	 c*params->y_ext_[ jext   *num_states+k] +
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

    // Compute the temperature advection (will be multiplied my mass flux)
    rhs_conv[j*num_states+num_species+1] -= relative_volume_j*
	(a*params->y_ext_[(jext+2)*num_states+num_species+1] +
	 b*params->y_ext_[(jext+1)*num_states+num_species+1] +
	 c*params->y_ext_[ jext   *num_states+num_species+1] +
	 d*params->y_ext_[(jext-1)*num_states+num_species+1] +
	 e*params->y_ext_[(jext-2)*num_states+num_species+1]);

    rhs_diff[j*num_states+num_species+1] -= relative_volume_j*
      cp_flux_sum/params->mixture_specific_heat_[j]*
	(a*params->y_ext_[(jext+2)*num_states+num_species+1] +
	 b*params->y_ext_[(jext+1)*num_states+num_species+1] +
	 c*params->y_ext_[ jext   *num_states+num_species+1] +
	 d*params->y_ext_[(jext-1)*num_states+num_species+1] +
	 e*params->y_ext_[(jext-2)*num_states+num_species+1]);

    // Add the thermal conductivity contribution to dT[j]/dt
    rhs_diff[j*num_states+num_species+1] +=
      (relative_volume_j*inv_dzm[jext]/params->mixture_specific_heat_[j])*
      (params->thermal_conductivity_[j+1]*inv_dz[jext+1]*
       (params->y_ext_[(jext+1)*num_states+num_species+1] -
        params->y_ext_[jext*num_states+num_species+1])
       -params->thermal_conductivity_[j]*inv_dz[jext]*
       (params->y_ext_[jext*num_states+num_species+1] -
        params->y_ext_[(jext-1)*num_states+num_species+1]));

    // Mass flux equation -- dV/dx + beta*rho*G
    if(finite_separation) {
      //left to right (assumes axisymmetric)
      rhs_diff[j*num_states+num_species] =
        -(params->y_ext_[jext*num_states+num_species] -
          params->y_ext_[(jext-1)*num_states+num_species])*inv_dz[jext] -
        (params->y_ext_[jext*num_states+num_species+2]/relative_volume_j +
         params->y_ext_[(jext-1)*num_states+num_species+2]/relative_volume_jm)
        *ref_momentum;
    } else {
      if(jglobal < params->jContBC_) {
        // Left of stagnation plane
        // Right to left propagation
        rhs_diff[j*num_states+num_species] =
          -(params->y_ext_[(jext+1)*num_states+num_species] -
            params->y_ext_[ jext   *num_states+num_species])*inv_dz[jext+1] -
          (1.0+params->simulation_type_)*params->y_ext_[jext*num_states+num_species+2]
          /relative_volume_j*ref_momentum;
      } else if (jglobal == params->jContBC_) {
        // Drive to 0
        rhs_diff[j*num_states+num_species] = params->y_ext_[jext*num_states+num_species];
      } else {
        // Right of stagnation plane
        // Left to right propagation
        rhs_diff[j*num_states+num_species] =
        - (params->y_ext_[jext*num_states+num_species] -
         params->y_ext_[(jext-1)*num_states+num_species])*inv_dz[jext] -
          (1.0+params->simulation_type_)*params->y_ext_[(jext-1)*num_states+num_species+2]
          /relative_volume_jm*ref_momentum;
      }
    }

    // Momentum equation
    // Compute the momentum advection term (will be multiplied by mass flux)
    rhs_conv[j*num_states+num_species+2] -= relative_volume_j*
      (a*params->y_ext_[(jext+2)*num_states+num_species+2] +
       b*params->y_ext_[(jext+1)*num_states+num_species+2] +
       c*params->y_ext_[ jext   *num_states+num_species+2] +
       d*params->y_ext_[(jext-1)*num_states+num_species+2] +
       e*params->y_ext_[(jext-2)*num_states+num_species+2]);

    // Compute momentum strain term P
    // Finite separation
    if(finite_separation) {
      rhs_diff[j*num_states+num_species+2] -= params->y_ext_[jext*num_states+num_species+3]
        *relative_volume_j;
    } else {
      // Infinite separation
      rhs_diff[j*num_states+num_species+2] += params->strain_rate_*params->strain_rate_/
        (1.0+params->simulation_type_)/(1.0+params->simulation_type_)*
        relative_volume_j/params->oxidizer_relative_volume_/ref_momentum;
    }

    // G*G
    rhs_diff[j*num_states+num_species+2] -= params->y_ext_[jext*num_states+num_species+2]*
      params->y_ext_[jext*num_states+num_species+2]*ref_momentum;

    // Compute momentum diffusion term
    rhs_diff[j*num_states+num_species+2] +=
      (inv_dzm[jext]*relative_volume_j)*
      (params->mixture_viscosity_[j+1]*inv_dz[jext+1]*
       (params->y_ext_[(jext+1)*num_states+num_species+2] -
        params->y_ext_[ jext   *num_states+num_species+2])
       -params->mixture_viscosity_[j]*inv_dz[jext]*
       (params->y_ext_[ jext   *num_states+num_species+2] -
        params->y_ext_[(jext-1)*num_states+num_species+2]));

    // Pstrain equation
    // Pstrain is calculated for finite separation
    // Pstrain is imposed for infinite separation
    if(finite_separation) {
      if (jglobal == (npes*num_local_points-1)) { //lastpoint: dV/dx+beta*rho*G
        rhs_diff[j*num_states+num_species+3] =
          -(params->y_ext_[(jext+1)*num_states+num_species]-
            params->y_ext_[jext*num_states+num_species])*inv_dz[jext+1] -
          (params->y_ext_[(jext+1)*num_states+num_species+2]/relative_volume_jp + //this is =0?
           params->y_ext_[jext*num_states+num_species+2]/relative_volume_j)*ref_momentum;
      }  else { // dP/dx
        //right to left
        rhs_diff[j*num_states+num_species+3] =
          (params->y_ext_[(jext+1)*num_states+num_species+3] -
           params->y_ext_[ jext   *num_states+num_species+3]);
      }
    }

  } // for(int j=0; j<num_local_points; ++j) // loop computing rhs

  // -------------------------------------------------------------------------
  // Compute the final residuals
  for(int j=0; j<num_local_points; ++j) {
    int mflux_id  = j*num_states+num_species; // relative volume index of pt j
    int temp_id   = mflux_id+1;               // temperature index of pt j
    int mom_id    = mflux_id+2;               // momentum index of pt j

    for(int k=0; k<num_species; ++k) {
      ydot_ptr[j*num_states+k] = rhs_conv[j*num_states+k]*y_ptr[mflux_id] +
        rhs_diff[j*num_states+k] + rhs_chem[j*num_states+k];
    }

    ydot_ptr[mflux_id] = rhs_diff[mflux_id];

    if(fixed_temperature) {
      ydot_ptr[temp_id] = y_ptr[temp_id] - params->fixed_temperature_[j];
    } else {
      ydot_ptr[temp_id] = rhs_conv[temp_id]*y_ptr[mflux_id] + rhs_diff[temp_id] + rhs_chem[temp_id];
    }

    ydot_ptr[mom_id] = rhs_conv[mom_id]*y_ptr[mflux_id] + rhs_diff[mom_id];

    if(finite_separation) {
      int strain_id = mflux_id+3;               // strain index of pt j
      ydot_ptr[strain_id] = rhs_diff[strain_id];
    }

    // Copy rhsConv in params for use in jacobian
    for(int k=0; k<num_states; ++k) {
      params->rhsConv_[j*num_states + k] = rhs_conv[j*num_states+k];
    }
  }

  // Add time derivative term if pseudo unsteady
  if(params->pseudo_unsteady_) {
    for(int j=0; j<num_local_points; ++j) {
      int mflux_id  = j*num_states+num_species; // relative volume index of pt j
      int temp_id   = mflux_id+1;               // temperature index of pt j
      int mom_id    = mflux_id+2;               // momentum index of pt j

      for(int k=0; k<num_species; ++k) {
        ydot_ptr[j*num_states+k] -= (y_ptr[j*num_states+k] - params->y_old_[j*num_states+k])/
          params->dt_;
      }

      ydot_ptr[temp_id] -= (y_ptr[temp_id] - params->y_old_[temp_id])/params->dt_;
      ydot_ptr[mom_id]  -= (y_ptr[mom_id]  - params->y_old_[mom_id] )/params->dt_;
    }
  }

  //------------------------------------------------------------------
  // Parallel communication for finite difference jacobian
  // TO DO: Move to a separate function?
  if(params->integrator_type_ == 2 || params->integrator_type_ == 3) {
    MPI_Status status;
    long int dsize = num_states*nover;

    // Copy y_ptr and ydot_ptr into larger arrays
    for (int j=0; j<num_states*num_local_points; ++j) {
      params->rhs_ext_[num_states*nover + j] = ydot_ptr[j];
    }

    // MPI sendrecv
    int nodeDest = my_pe-1;
    if (nodeDest < 0) nodeDest = npes-1;
    int nodeFrom = my_pe+1;
    if (nodeFrom > npes-1) nodeFrom = 0;
    MPI_Sendrecv(&params->rhs_ext_[nover*num_states], dsize, PVEC_REAL_MPI_TYPE, nodeDest, 0,
                 &params->rhs_ext_[num_states*(num_local_points+nover)], dsize,
                 PVEC_REAL_MPI_TYPE, nodeFrom, 0, comm, &status);

    nodeDest = my_pe+1;
    if (nodeDest > npes-1) nodeDest = 0;
    nodeFrom = my_pe-1;
    if (nodeFrom < 0) nodeFrom = npes-1;
    MPI_Sendrecv(&params->rhs_ext_[num_states*num_local_points], dsize,
                 PVEC_REAL_MPI_TYPE, nodeDest, 0, &params->rhs_ext_[0], dsize,
                 PVEC_REAL_MPI_TYPE, nodeFrom, 0, comm, &status);
  }

  // -------------------------------------------------------------------------
  // Compute fuel burning rate/laminar flame speed = int(omega_F)/rho_u/YF_u
  double sum_omega_F = 0.0;
  int num_fuel_species = params->fuel_species_id_.size();
  local_sum = 0.0;
  for(int j=0; j<num_local_points; ++j) {
    int jext = j + nover;
    for(int k=0; k<num_fuel_species; ++k) {
      local_sum -= rhs_chem[j*num_states + params->fuel_species_id_[k]]*
        dzm[jext]/params->rel_vol_[j];
    }
  }
  MPI_Allreduce(&local_sum,&sum_omega_F,1,PVEC_REAL_MPI_TYPE,MPI_SUM,comm);
  double sum_inlet_fuel_mass_fractions = 0.0;
  if(params->flame_type_ == 1 || params->flame_type_ == 2) {
    // premixed flame
    for(int k=0; k<num_fuel_species; ++k) {
      sum_inlet_fuel_mass_fractions += params->inlet_mass_fractions_[params->fuel_species_id_[k]];
    }
    sum_omega_F /= sum_inlet_fuel_mass_fractions/params->inlet_relative_volume_;
  } else {
    //diffusion flame
    sum_inlet_fuel_mass_fractions = 1.0;
    sum_omega_F /= sum_inlet_fuel_mass_fractions/params->fuel_relative_volume_;
  }
  params->flame_speed_ = sum_omega_F;

  if(finite_separation) {
    // Compute characteristic strain rate
    // Compute normal strain rate (dv/dz)
    std::vector<double> strain_rate_abs, velocity;
    strain_rate_abs.assign(num_local_points, 0.0);
    velocity.assign(num_local_points, 0.0);
    for(int j=0; j<num_local_points; ++j) {
      int jext = j + nover;
      velocity[j] = params->y_ext_[jext*num_states+num_species]*params->rel_vol_ext_[jext];
      strain_rate_abs[j] = fabs(
        (params->y_ext_[(jext+1)*num_states+num_species]*params->rel_vol_ext_[jext+1] -
         velocity[j])*inv_dz[jext]);
    }

    // Method 1: Doesn't always work?
    /*
    // Find the max normal strain rate location
    double max_strain;
    int jglobal_max_strain;
    max_strain = FindMaximumParallel(num_local_points, &strain_rate_abs[0], &jglobal_max_strain);

    // Find the minimum velocity ahead of the max strain location
    struct { double value; int index;} in, out;
    in.value = 10000;
    in.index = 0;
    for(int j=0; j<num_local_points; ++j) {
      int jglobal = j + my_pe*num_local_points;
      if(in.value > velocity[j] and jglobal < jglobal_max_strain) {
        in.value = velocity[j];
        in.index = jglobal;
      }
    }
    MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MINLOC, comm);
    int jglobal_min_vel = out.index;

    // Find the max strain rate ahead of the minimum velocity point
    in.value = -10000;
    in.index = 0;
    for(int j=0; j<num_local_points; ++j) {
      int jglobal = j + my_pe*num_local_points;
      if(in.value < strain_rate_abs[j] and jglobal < jglobal_min_vel) {
        in.value = strain_rate_abs[j];
        in.index = jglobal;
      }
    }
    MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, comm);
    params->strain_rate_ = out.value;
    */

    // Method 2:
    // ONLY WORKS IN SERIAL FOR NOW
    long int dsize;
    double *sbuf;
    if(my_pe == 0)
      sbuf = (double *)malloc(num_local_points*npes*sizeof(double));

    // Gather strain rate on root
    dsize = num_local_points;
    //MPI_Allgather(&strain_rate_abs[0],
    MPI_Gather(&strain_rate_abs[0],
               dsize,
               PVEC_REAL_MPI_TYPE,
               sbuf,
               dsize,
               PVEC_REAL_MPI_TYPE,
               0,
               comm);

    if(my_pe == 0) {
      params->strain_rate_ = -100000;
      for(int j=0; j<num_local_points*npes; ++j) {
        if(sbuf[j] > params->strain_rate_) {
          params->strain_rate_ = sbuf[j];
        } else {
          break;
        }
      }
    }
    if(my_pe == 0) free(sbuf);
    MPI_Bcast(&params->strain_rate_, 1, MPI_DOUBLE, 0, comm);

  }

  double local_temperature;
  local_max = 0.0;
  for(int j=0; j<num_local_points; ++j) {
    int jext = j + nover;
    local_temperature = ref_temperature*params->y_ext_[jext*num_states+num_species+1];
    if(local_temperature > local_max) {
      local_max = local_temperature;
    }
  }
  MPI_Allreduce(&local_max,&params->max_temperature_,1,PVEC_REAL_MPI_TYPE,MPI_MAX,comm);

  double gradT;
  local_max = 0.0;
  for(int j=0; j<num_local_points; ++j) {
    int jext = j + nover;
    gradT = fabs(inv_dz[jext]*ref_temperature*
		 (params->y_ext_[(jext+1)*num_states+num_species+1] -
                  params->y_ext_[jext*num_states+num_species+1]));
    if (gradT > local_max) {
      local_max = gradT;
    }
  }
  MPI_Allreduce(&local_max,&params->flame_thickness_,1,PVEC_REAL_MPI_TYPE,MPI_MAX,comm);
  params->flame_thickness_ = (params->max_temperature_-params->fuel_temperature_)/
    params->flame_thickness_;

  // compute the max thermal diffusivity using the average value of the
  // conductivity and the up and downstream interfaces
  local_max = 0.0;
  for(int j=0; j<num_local_points; ++j) {
    thermal_diffusivity =
      fabs(0.5*(params->thermal_conductivity_[j]+
                params->thermal_conductivity_[j+1])*
           params->rel_vol_ext_[nover+j]/params->mixture_specific_heat_[j]);
    if(thermal_diffusivity > local_max) {
      local_max = thermal_diffusivity;
    }
  }
  MPI_Allreduce(&local_max,&params->max_thermal_diffusivity_,1,PVEC_REAL_MPI_TYPE,MPI_MAX,comm);

  return 0;
}

//------------------------------------------------------------------
// Banded Block Diagonal preconditioner, factorized with SuperLU
#if defined SUNDIALS2
int ReactorBBDSetup(N_Vector y, // [in] state vector
                    N_Vector yscale, // [in] state scaler
                    N_Vector ydot, // [in] state derivative
                    N_Vector ydotscale, // [in] state derivative scaler
                    void *user_data, // [in/out]
                    N_Vector tmp1, N_Vector tmp2)
{
#elif defined SUNDIALS3 || defined SUNDIALS4
int ReactorBBDSetup(N_Vector y, // [in] state vector
                    N_Vector yscale, // [in] state scaler
                    N_Vector ydot, // [in] state derivative
                    N_Vector ydotscale, // [in] state derivative scaler
                    void *user_data) // [in/out]
{
#endif
  FlameParams *params    = (FlameParams *)user_data;
  const int num_local_points   = params->num_local_points_;
  const int num_states   = params->reactor_->GetNumStates();
  const int num_species   = params->reactor_->GetNumSpecies();
  const int num_nonzeros_loc = params->num_nonzeros_loc_;
  const int num_local_states = num_states*num_local_points;
  const int num_total_points = params->num_points_;
  const int num_total_states = num_states*num_total_points;
  double *y_ptr          = NV_DATA_P(y);
  double *ydot_ptr       = NV_DATA_P(ydot);
  int error_flag = 0;
  double alpha = 1.0e-6;
  double beta = 1.0e-14;
  double delta;

  int my_pe = params->my_pe_;
  int npes  = params->npes_;
  const int nover=params->nover_;

  // Create work arrays
  std::vector<double> y_saved,rhs_ext_saved;
  y_saved.assign(num_local_points*num_states,0.0);
  rhs_ext_saved.assign((num_local_points+2*nover)*num_states,0.0);

  int group, width;
  int mkeep = params->num_off_diagonals_;
  width = 2*mkeep + 1;

  std::vector<double> jac_bnd;
  jac_bnd.assign( (num_local_points+2*nover)*num_states*width, 0.0);

  // Compute RHS
  ConstPressureFlame(y, ydot, user_data);

  // Save copy of state vector and rhs
  for (int j=0; j<num_local_states; ++j)
    y_saved[j] = y_ptr[j];
  for (int j=0; j<num_states*(num_local_points+2*nover); ++j)
    rhs_ext_saved[j] = params->rhs_ext_[j];

  // Banded jacobian
  int j, jglobal, i1global, i2global, iloc, jext, iext, jstate;
  for (group = 1; group <= width; group++) {

    // Perturb y
    for (jglobal=group-1; jglobal<num_total_states; jglobal+=width) {
      j = jglobal - my_pe*num_local_points*num_states;
      if (j>=0 && j<num_local_states) {
        delta = alpha*y_saved[j] + beta;
        y_ptr[j] = y_saved[j] + delta;
      }
    }//for j=group-1 +=width

    // Compute RHS
    ConstPressureFlame(y, ydot, user_data);

    // Compute jacobian
    // here j is the COLUMN and i is the ROW
    for (jglobal=group-1; jglobal<num_total_states; jglobal+=width) {
      j = jglobal - my_pe*num_local_states;
      jstate = jglobal % num_states;
      if (j>=0 && j<num_local_states) {
        i1global = max(0, jglobal - jstate - num_states);
        i2global = min(jglobal + (num_states-1 - jstate) + num_states, num_total_states-1);
        jext = j + nover*num_states;
        for (int i=i1global; i<=i2global; i++) {
          iloc = i-my_pe*num_local_states;
          iext = iloc + nover*num_states;
          jac_bnd[jext*width + i-jglobal+mkeep] =
            (params->rhs_ext_[iext] - rhs_ext_saved[iext]) / (y_ptr[j]- y_saved[j]);
          /*
          if(jstate==num_species+2 and i % num_states == num_species+2)
            printf("col: %d, row: %d, Jac: %5.3e\n",
                   jglobal / num_states,
                   i / num_states,
                   jac_bnd[jext*width + i-jglobal+mkeep]);
          */
        } // for i < i2
        y_ptr[j] = y_saved[j];
      } //if j exists on this processor
    } // for j=group-1 +=width

  } // for group < width

  // Restore the state and rhs vectors back to original values
  for (int j=0; j<num_local_states; ++j) {
    jext = j + nover*num_states;
    y_ptr[j] = y_saved[j];
    ydot_ptr[j] = rhs_ext_saved[jext];
  }

  // Perform parallel communication of jacobian
  MPI_Comm comm = params->comm_;
  MPI_Status status;
  long int dsize_jac_bnd = nover*num_states*width;
  // MPI sendrecv
  int nodeDest = my_pe-1;
  if (nodeDest < 0) nodeDest = npes-1;
  int nodeFrom = my_pe+1;
  if (nodeFrom > npes-1) nodeFrom = 0;
  MPI_Sendrecv(&jac_bnd[nover*num_states*width], dsize_jac_bnd, PVEC_REAL_MPI_TYPE, nodeDest, 0, &jac_bnd[num_states*(num_local_points+nover)*width], dsize_jac_bnd, PVEC_REAL_MPI_TYPE, nodeFrom, 0, comm, &status);

  nodeDest = my_pe+1;
  if (nodeDest > npes-1) nodeDest = 0;
  nodeFrom = my_pe-1;
  if (nodeFrom < 0) nodeFrom = npes-1;
  MPI_Sendrecv(&jac_bnd[num_states*num_local_points*width], dsize_jac_bnd, PVEC_REAL_MPI_TYPE, nodeDest, 0, &jac_bnd[0], dsize_jac_bnd, PVEC_REAL_MPI_TYPE, nodeFrom, 0, comm, &status);

  // Get pattern "manually" for now
  // TODO: find a cleaner way
  int innz=0;
  // here j is the ROW and i is the COLUMN
  for (j=0; j<num_local_states; j++) {
    jglobal = j + my_pe*num_local_states;
    jext = j + nover*num_states;
    int jstate = jglobal % num_states;
    i1global = max(0, jglobal - jstate - num_states);
    i2global = min(jglobal + (num_states-1 - jstate) + num_states, num_total_states-1);
    for (int i=i1global; i<=i2global; i++) {
      iloc = i-my_pe*num_local_states;
      iext = iloc + nover*num_states;
      int istate = i % num_states;
      int dense_id = num_states*istate + jstate; //i is column and j is row
      //Diagonal block.
      if (i>= jglobal-jstate && i<=jglobal+num_states-1-jstate) {
        if (params->dense_to_sparse_[dense_id] == 1) {
          params->reactor_jacobian_dist_[innz] = jac_bnd[iext*width + jglobal-i+mkeep];
          innz++;
        }
      }
      //Off-diagonal blocks
      if (i<jglobal-jstate || i>jglobal+num_states-1-jstate) {
        if (params->dense_to_sparse_offdiag_[dense_id] == 1) {
          params->reactor_jacobian_dist_[innz] = jac_bnd[iext*width + jglobal-i+mkeep];
          innz++;
        }
      }
    } // for i1 to i2
  } // for j < num_local_states

  // Factorize with SuperLU (parallel is default, serial if specified in input)
  if(params->superlu_serial_) {
    if(params->sparse_matrix_->IsFirstFactor()) {
      error_flag =
        params->sparse_matrix_->FactorNewPatternCRS(num_nonzeros_loc,
                                                    &params->col_id_[0],
                                                    &params->row_sum_[0],
                                                    &params->reactor_jacobian_dist_[0]);
    } else {
      error_flag =
        params->sparse_matrix_->FactorSamePattern(&params->reactor_jacobian_dist_[0]);
    } //if first factor
  } else {
    if(params->sparse_matrix_dist_->IsFirstFactor_dist()) {
      error_flag =
        params->sparse_matrix_dist_->FactorNewPatternCCS_dist(num_nonzeros_loc,
                                                              &params->col_id_[0],
                                                              &params->row_sum_[0],
                                                              &params->reactor_jacobian_dist_[0]);
    } else {
      error_flag =
        params->sparse_matrix_dist_->FactorSamePatternCCS_dist(num_nonzeros_loc,
                                                               &params->col_id_[0],
                                                               &params->row_sum_[0],
                                                               &params->reactor_jacobian_dist_[0]);
    } //if first factor
  } // if superlu serial

  return error_flag;
}
// Banded block diagonal finite difference Jacobian, solved with SuperLU
#if defined SUNDIALS2
int ReactorBBDSolve(N_Vector y, // [in] state vector
                    N_Vector yscale, // [in] state scaler
                    N_Vector ydot, // [in] state derivative
                    N_Vector ydotscale, // [in] state derivative scaler
                    N_Vector vv, // [in/out] rhs/solution vector
                    void *user_data, // [in/out]
                    N_Vector tmp)
{
#elif defined SUNDIALS3 || defined SUNDIALS4
int ReactorBBDSolve(N_Vector y, // [in] state vector
                    N_Vector yscale, // [in] state scaler
                    N_Vector ydot, // [in] state derivative
                    N_Vector ydotscale, // [in] state derivative scaler
                    N_Vector vv, // [in/out] rhs/solution vector
                    void *user_data) // [in/out]
{
#endif
  FlameParams *params = (FlameParams *)user_data;
  double *solution = NV_DATA_P(vv);
  int error_flag = 0;

  if(params->superlu_serial_) {
    error_flag = params->sparse_matrix_->Solve(&solution[0],&solution[0]);
  } else {
    error_flag = params->sparse_matrix_dist_->Solve_dist(&solution[0],&solution[0]);
  }

  return error_flag;
}

//------------------------------------------------------------------
// Approximate factorization preconditioner
// Chemistry Jacobian is (sparse) block diagonal. Each n_sp x n_sp block
// factorized separately with SuperLU
// Transport is tridiagonal over whole domain, factorized with LAPACK
#if defined SUNDIALS2
int ReactorAFSetup(N_Vector y, // [in] state vector
                   N_Vector yscale, // [in] state scaler
                   N_Vector ydot, // [in] state derivative
                   N_Vector ydotscale, // [in] state derivative scaler
                   void *user_data, // [in/out]
                   N_Vector tmp1, N_Vector tmp2)
{
#elif defined SUNDIALS3 || defined SUNDIALS4
int ReactorAFSetup(N_Vector y, // [in] state vector
                   N_Vector yscale, // [in] state scaler
                   N_Vector ydot, // [in] state derivative
                   N_Vector ydotscale, // [in] state derivative scaler
                   void *user_data) // [in/out]
{
#endif
  FlameParams *params    = (FlameParams *)user_data;
  const int num_local_points   = params->num_local_points_;
  const int num_states   = params->reactor_->GetNumStates();
  const int num_species = params->reactor_->GetNumSpecies();
  const int num_total_points = params->num_points_;
  const int num_nonzeros_zerod = params->reactor_->GetJacobianSize();
  const int num_states_local = params->num_states_local_;
  double *y_ptr          = NV_DATA_P(y);
  int error_flag = 0;
  bool Tfix = false;
  double constant = 1.0e5;//1.0e4
  int my_pe  = params->my_pe_;
  const int nover = params->nover_;

  const double ref_momentum = params->ref_momentum_;
  const bool finite_separation = params->parser_->finite_separation();
  const bool fixed_temperature = params->parser_->fixed_temperature();

  // Initialize transport Jacobian
  for(int j=0; j<num_local_points*5*num_states; j++)
    params->banded_jacobian_[j] = 0.0;

  // Get grid spacing
  std::vector<double> dz, dzm, inv_dz, inv_dzm;
  dz.assign( num_local_points+(2*nover), 0.0);
  dzm.assign( num_local_points+(2*nover), 0.0);
  inv_dz.assign( num_local_points+(2*nover), 0.0);
  inv_dzm.assign( num_local_points+(2*nover), 0.0);
  for (int j=0; j<num_local_points+2*nover; ++j) {
    dz[j] = params->dz_local_[j];
    dzm[j] = params->dzm_local_[j];
    inv_dz[j] = params->inv_dz_local_[j];
    inv_dzm[j] = params->inv_dzm_local_[j];
  }

  // Evaluate analytic transport J
  const int convective_scheme_type = params->convective_scheme_type_;
  double relative_volume_j, relative_volume_jp, relative_volume_jm;
  for(int j=0; j<num_local_points; j++) {
    int jext = j + nover;
    int jglobal = j + my_pe*num_local_points;

    double bm=0,b=0,c=0,d=0,dp=0; //coefficients of  j+1, j, j-1 terms
    if(convective_scheme_type == 0) {
      // First order upwind
      if(y_ptr[j*num_states+num_species] > 0) {
        bm = 0.0;
        b  = 0.0;
        c  =  inv_dz[jext];
        d  = -inv_dz[jext];
        dp = -inv_dz[jext+1];
      } else {
        bm =  inv_dz[jext];
        b  =  inv_dz[jext+1];
        c  = -inv_dz[jext+1];
        d  = 0.0;
        dp = 0.0;
      }
    } else if(convective_scheme_type == 1) {
      // Second order upwind
      if (y_ptr[j*num_states+num_species] > 0) {
        // Use points upstream
        bm = 0.0;
        b  = 0.0;
        c  = inv_dz[jext] + 1.0/(dz[jext]+dz[jext-1]);
        d  = -(dz[jext]+dz[jext-1])/(dz[jext]*dz[jext-1]);
	dp = -(dz[jext+1]+dz[jext])/(dz[jext+1]*dz[jext]);
      } else {
        // Use points downstream
        bm = inv_dz[jext] + inv_dz[jext+1];
        b  = inv_dz[jext+1] + inv_dz[jext+2];
        c  = -inv_dz[jext+1] - 1.0/(dz[jext+1]+dz[jext+2]);
        d  = 0.0;
        dp = 0.0;
      }
    } else if(convective_scheme_type == 2) {
      // Centered
      bm = dz[jext-1]/dz[jext]/(dz[jext-1]+dz[jext]);
      b  = dz[jext]/dz[jext+1]/(dz[jext]+dz[jext+1]);
      c  = (dz[jext+1]-dz[jext])/dz[jext+1]/dz[jext];
      d  = -dz[jext+1]/dz[jext]/(dz[jext]+dz[jext+1]);
      dp = -dz[jext+2]/dz[jext+1]/(dz[jext+1]+dz[jext+2]);
    } else {
      printf("Undefined convective scheme \n");
      MPI_Finalize();
      exit(0);
    }

    relative_volume_j  = params->rel_vol_ext_[jext];
    relative_volume_jp = params->rel_vol_ext_[jext+1];
    relative_volume_jm = params->rel_vol_ext_[jext-1];

    // Species
    for(int k=0; k<num_species; k++) {
      // Diagonal drhs_j/dY_j
      params->banded_jacobian_[k*(num_local_points*5) + j*5 + 1 + 2 + 0] =
        (-params->thermal_conductivity_[j+1]*inv_dz[jext+1]/
         params->mixture_specific_heat_mid_[j+1]/
         params->species_lewis_numbers_[k] -
         params->thermal_conductivity_[j]*inv_dz[jext]/
         params->mixture_specific_heat_mid_[j]/
         params->species_lewis_numbers_[k])*
        relative_volume_j*inv_dzm[jext] -
        c*y_ptr[j*num_states+num_species]*relative_volume_j;

      // drhs_j-1/dY_j
      if(jglobal > 0) {
        params->banded_jacobian_[k*(num_local_points*5) + j*5 + 1 + 2 - 1] =
          (params->thermal_conductivity_[j]*inv_dz[jext]/
           params->mixture_specific_heat_mid_[j]/
           params->species_lewis_numbers_[k])*
          relative_volume_jm*inv_dzm[jext-1]-
          bm*params->y_ext_[(jext-1)*num_states+num_species]*relative_volume_jm;
      }

      // drhs_j+1/dY_j
      if(jglobal < num_total_points-1) {
        params->banded_jacobian_[k*(num_local_points*5) + j*5 + 1 + 2 + 1] =
          (params->thermal_conductivity_[j+1]*inv_dz[jext+1]/
           params->mixture_specific_heat_mid_[j+1]/
           params->species_lewis_numbers_[k])*
          relative_volume_jp*inv_dzm[jext+1] -
          dp*params->y_ext_[(jext+1)*num_states+num_species]*relative_volume_jp;
      }

    } // for k<num_species

    // Temperature
    if(fixed_temperature) {
      params->banded_jacobian_[(num_species+1)*(num_local_points*5) + j*5 + 1 + 2 + 0] = 1.0;
    } else {
      // Diagonal drhs_j/dT_j
      params->banded_jacobian_[(num_species+1)*(num_local_points*5) + j*5 + 1 + 2 + 0] =
        (-params->thermal_conductivity_[j+1]*inv_dz[jext+1]/
         params->mixture_specific_heat_mid_[j+1] -
         params->thermal_conductivity_[j]*inv_dz[jext]/
         params->mixture_specific_heat_mid_[j])*
        relative_volume_j*inv_dzm[jext] -
        c*y_ptr[j*num_states+num_species]*relative_volume_j;

      // drhs_j-1/dY_j
      if(jglobal > 0) {
        params->banded_jacobian_[(num_species+1)*(num_local_points*5) + j*5 + 1 + 2 - 1] =
          (params->thermal_conductivity_[j]*inv_dz[jext]/
           params->mixture_specific_heat_mid_[j])*
          relative_volume_jm*inv_dzm[jext-1] -
          bm*params->y_ext_[(jext-1)*num_states+num_species]*relative_volume_jm;
      }

      // drhs_j+1/dY_j
      if(jglobal < num_total_points-1) {
        params->banded_jacobian_[(num_species+1)*(num_local_points*5) + j*5 + 1 + 2 + 1] =
          (params->thermal_conductivity_[j+1]*inv_dz[jext+1]/
           params->mixture_specific_heat_mid_[j+1])*
          relative_volume_jp*inv_dzm[jext+1] -
          dp*params->y_ext_[(jext+1)*num_states+num_species]*relative_volume_jp;
      }
    }

    // Mass flux
    if(finite_separation) {
      params->banded_jacobian_[num_species*(num_local_points*5) + j*5 + 1 + 2 + 0] = -inv_dz[jext];
      if(jglobal < num_total_points-1)
        params->banded_jacobian_[num_species*(num_local_points*5) + j*5 + 1 + 2 + 1] = inv_dz[jext+1];
      if(jglobal > 0)
        params->banded_jacobian_[num_species*(num_local_points*5) + j*5 + 1 + 2 - 1] = 0.0;
    } else {
      if (jglobal == params->jContBC_) {
        params->banded_jacobian_[num_species*(num_local_points*5) + j*5 + 1 + 2 + 0] = 1.0;

        params->banded_jacobian_[num_species*(num_local_points*5) + j*5 + 1 + 2 + 1] = inv_dz[jext+1];
        params->banded_jacobian_[num_species*(num_local_points*5) + j*5 + 1 + 2 - 1] = -inv_dz[jext];
      }
      if (jglobal > params->jContBC_) {
        params->banded_jacobian_[num_species*(num_local_points*5) + j*5 + 1 + 2 + 0] = -inv_dz[jext];
        if(jglobal < num_total_points-1)
          params->banded_jacobian_[num_species*(num_local_points*5) + j*5 + 1 + 2 + 1] = inv_dz[jext+1];
        params->banded_jacobian_[num_species*(num_local_points*5) + j*5 + 1 + 2 - 1] = 0.0;
      }

      if (jglobal < params->jContBC_) {
        params->banded_jacobian_[num_species*(num_local_points*5) + j*5 + 1 + 2 + 0] = inv_dz[jext+1];
        params->banded_jacobian_[num_species*(num_local_points*5) + j*5 + 1 + 2 + 1] = 0.0;
        if(jglobal > 0)
          params->banded_jacobian_[num_species*(num_local_points*5) + j*5 + 1 + 2 - 1] = -inv_dz[jext];
      }
      /*
      if (jglobal == params->jContBC_ + 1)
        params->banded_jacobian_[num_species*(num_local_points*5) + j*5 + 1 + 2 - 1] = 0.0;
      if (jglobal == params->jContBC_ - 1)
        params->banded_jacobian_[num_species*(num_local_points*5) + j*5 + 1 + 2 + 1] = 0.0;
      */
    }

    // Momentum
    // Diagonal drhs_j/dU_j
    // Special treatment for zero gradient BC with infinite separation
    if(finite_separation) {
      params->banded_jacobian_[(num_species+2)*(num_local_points*5) + j*5 + 1 + 2 + 0] =
        (-params->mixture_viscosity_[j+1]*inv_dz[jext+1] -
         params->mixture_viscosity_[j]*inv_dz[jext])*relative_volume_j*inv_dzm[jext] -
        c*y_ptr[j*num_states+num_species]*relative_volume_j -
        2.0*y_ptr[j*num_states+num_species+2]*ref_momentum;
    } else {
      if(jglobal == 0) {
        params->banded_jacobian_[(num_species+2)*(num_local_points*5) + j*5 + 1 + 2 + 0] =
          -params->mixture_viscosity_[j+1]*inv_dz[jext+1]*relative_volume_j*inv_dzm[jext] -
          (c+d)*y_ptr[j*num_states+num_species]*relative_volume_j -
          2.0*y_ptr[j*num_states+num_species+2]*ref_momentum;
      } else if (jglobal == num_total_points -1 ) {
        params->banded_jacobian_[(num_species+2)*(num_local_points*5) + j*5 + 1 + 2 + 0] =
          -params->mixture_viscosity_[j]*inv_dz[jext]*relative_volume_j*inv_dzm[jext] -
          (b+c)*y_ptr[j*num_states+num_species]*relative_volume_j -
          2.0*y_ptr[j*num_states+num_species+2]*ref_momentum;
      } else {
        params->banded_jacobian_[(num_species+2)*(num_local_points*5) + j*5 + 1 + 2 + 0] =
          (-params->mixture_viscosity_[j+1]*inv_dz[jext+1] -
           params->mixture_viscosity_[j]*inv_dz[jext])*relative_volume_j*inv_dzm[jext] -
          c*y_ptr[j*num_states+num_species]*relative_volume_j -
      2.0*y_ptr[j*num_states+num_species+2]*ref_momentum;
      }
    }

    // drhs_j-1/dY_j
    if(jglobal > 0) {
      params->banded_jacobian_[(num_species+2)*(num_local_points*5) + j*5 + 1 + 2 - 1] =
        params->mixture_viscosity_[j]*inv_dz[jext]*relative_volume_jm*inv_dzm[jext-1] -
        bm*params->y_ext_[(jext-1)*num_states+num_species]*relative_volume_jm;
    }

    // drhs_j+1/dY_j
    if(jglobal < num_total_points-1) {
      params->banded_jacobian_[(num_species+2)*(num_local_points*5) + j*5 + 1 + 2 + 1] =
        params->mixture_viscosity_[j+1]*inv_dz[jext+1]*relative_volume_jp*inv_dzm[jext+1] -
        dp*params->y_ext_[(jext+1)*num_states+num_species]*relative_volume_jp;
    }

    /*
    printf("j: %d, -diag: %5.3e\n", j, params->banded_jacobian_[(num_species+2)*(num_local_points*5) + j*5 + 1 + 2 - 1]);
    printf("j: %d, diag: %5.3e\n", j, params->banded_jacobian_[(num_species+2)*(num_local_points*5) + j*5 + 1 + 2 + 0]);
    printf("j: %d, +diag: %5.3e\n", j, params->banded_jacobian_[(num_species+2)*(num_local_points*5) + j*5 + 1 + 2 + 1]);
    */

    // Pstrain
    if(finite_separation) {
      // Diagonal drhs_j/dP_j
      /*
      if(jglobal < num_total_points-1) {
        params->banded_jacobian_[(num_species+3)*(num_local_points*5) + j*5 + 1 + 2 + 0] = -1.0;
        params->banded_jacobian_[(num_species+3)*(num_local_points*5) + j*5 + 1 + 2 + 1] = 0.0;
        params->banded_jacobian_[(num_species+3)*(num_local_points*5) + j*5 + 1 + 2 - 1] = 1.0;
      } else { // Last point
        params->banded_jacobian_[(num_species+3)*(num_local_points*5) + j*5 + 1 + 2 + 0] = -1.0;//0.0; //is actually zero but can't invert with 0 and it seems to work with -1?
        params->banded_jacobian_[(num_species+3)*(num_local_points*5) + j*5 + 1 + 2 - 1] = 1.0;
      }
      */
      params->banded_jacobian_[(num_species+3)*(num_local_points*5) + j*5 + 1 + 2 + 0] = -1.0;
      if(jglobal > 0)
        params->banded_jacobian_[(num_species+3)*(num_local_points*5) + j*5 + 1 + 2 - 1] = 1.0;
      if(jglobal < num_total_points-1)
        params->banded_jacobian_[(num_species+3)*(num_local_points*5) + j*5 + 1 + 2 + 1] = 0.0;
    }

  } // for j<num_local_points

  // Local chemistry Jacobian (and mass flux)
  if(params->store_jacobian_) {
    params->saved_jacobian_chem_.assign(num_nonzeros_zerod*num_local_points, 0.0);
    for(int j=0; j<num_local_points; ++j) {
      int jglobal = j + my_pe*num_local_points;
      if(jglobal == num_total_points-1) {
        Tfix = true;
      } else {
        Tfix = false;
      }
      // Get Jacobian
      params->reactor_->GetJacobianSteady(&y_ptr[j*num_states],
                                          &params->rhsConv_[j*num_states],
                                          Tfix,
                                          ref_momentum,
                                          &params->step_limiter_[0],
                                          &params->saved_jacobian_chem_[j*num_nonzeros_zerod]);

      if(params->pseudo_unsteady_) {
        // Add -1/dt term to Yi, T, and G
        for(int k=0; k<num_species; ++k) {
          params->saved_jacobian_chem_[j*num_nonzeros_zerod+params->diagonal_id_chem_[k]] -= 1.0/params->dt_;
        }
        params->saved_jacobian_chem_[j*num_nonzeros_zerod+params->diagonal_id_chem_[num_species+1]] -= 1.0/params->dt_;
        params->saved_jacobian_chem_[j*num_nonzeros_zerod+params->diagonal_id_chem_[num_species+2]] -= 1.0/params->dt_;
      }

      //Add/subtract identity
      for(int k=0; k<num_states; ++k) {
        params->saved_jacobian_chem_[j*num_nonzeros_zerod+params->diagonal_id_chem_[k]] -= constant;
      }

    } // for j<num_local_points

    for(int j=0; j<num_local_points; ++j) {
      for(int k=0; k<num_nonzeros_zerod; ++k) {
        params->reactor_jacobian_chem_[k] =
          params->saved_jacobian_chem_[j*num_nonzeros_zerod+k];
      }
      // factor the numerical jacobian
      if(params->sparse_matrix_chem_[j]->IsFirstFactor()) {
        error_flag =
          params->sparse_matrix_chem_[j]->FactorNewPatternCCS(num_nonzeros_zerod,
                                                              &params->row_id_chem_[0],
                                                              &params->column_sum_chem_[0],
                                                              &params->reactor_jacobian_chem_[0]);
      } else {
        error_flag =
          params->sparse_matrix_chem_[j]->FactorSamePattern(&params->reactor_jacobian_chem_[0]);
      }
      if(error_flag != 0) {
        printf("Sparse matrix error at point %d\n", j);
        params->logger_->PrintF(
                                "# DEBUG: grid point %d (z = %.18g [m]) reactor produced a\n"
                                "#        sparse matrix error flag = %d\n",
                                j,
                                params->z_[j],
                                error_flag);
	return error_flag;
      }

    } // for j<num_local_points

  } else {
    // recompute and factor the Jacobian, there is no saved data
    // TODO: offer option for the fake update
    for(int j=0; j<num_local_points; ++j) {
      int jglobal = j + my_pe*num_local_points;
      if(jglobal == num_total_points-1) {
        Tfix = true;
      } else {
        Tfix = false;
      }
      params->reactor_->GetJacobianSteady(&y_ptr[j*num_states],
                                          &params->rhsConv_[j*num_states],
                                          Tfix,
                                          ref_momentum,
                                          &params->step_limiter_[0],
                                          &params->reactor_jacobian_chem_[0]);

      if(params->pseudo_unsteady_) {
        // Add -1/dt term to Yi, T, and G
        for(int k=0; k<num_species; ++k) {
          params->saved_jacobian_chem_[j*num_nonzeros_zerod+params->diagonal_id_chem_[k]] -= 1.0/params->dt_;
        }
        params->saved_jacobian_chem_[j*num_nonzeros_zerod+params->diagonal_id_chem_[num_species+1]] -= 1.0/params->dt_;
        params->saved_jacobian_chem_[j*num_nonzeros_zerod+params->diagonal_id_chem_[num_species+2]] -= 1.0/params->dt_;
      }

      //Add/subtract identity
      for(int k=0; k<num_states; ++k) {
        params->reactor_jacobian_chem_[params->diagonal_id_chem_[k]] -= constant;
      }

      // factor the numerical jacobian
      if(params->sparse_matrix_chem_[j]->IsFirstFactor()) {
        error_flag =
          params->sparse_matrix_chem_[j]->FactorNewPatternCCS(num_nonzeros_zerod,
                                                              &params->row_id_chem_[0],
                                                              &params->column_sum_chem_[0],
                                                              &params->reactor_jacobian_chem_[0]);
    } else {
	error_flag =
          params->sparse_matrix_chem_[j]->FactorSamePattern(&params->reactor_jacobian_chem_[0]);
      }
      if(error_flag != 0) {
        printf("Sparse matrix error flag = %d\n", error_flag);
        params->logger_->PrintF(
                                "# DEBUG: grid point %d (z = %.18g [m]) reactor produced a\n"
                                "#        sparse matrix error flag = %d\n",
                                j,
                                params->z_[j],
                                error_flag);
        return error_flag;
      }
    } // for(int j=0; j<num_local_points; ++j)
  } // if(params->store_jacobian_)

  // Add/Subtract identity to/from transport jacobian
  for(int j=0; j<num_local_points; ++j) {
    for(int k=0; k<num_states; ++k) {
      params->banded_jacobian_[k*(num_local_points*5) + j*5 + 1 + 2 + 0] += constant;
    }
  }

  // Multiply by inverse of chemical jacobian
  // TO DO: Make it work when there is no saved jacobian
  double inverse_chem_jacobian;
  for(int j=0; j<num_local_points; ++j) {
    for(int k=0; k<num_states; ++k) {
      inverse_chem_jacobian = 1.0/params->saved_jacobian_chem_[j*num_nonzeros_zerod+params->diagonal_id_chem_[k]];
      params->banded_jacobian_[k*(num_local_points*5) + j*5 + 1 + 2 + 0] *= inverse_chem_jacobian;
      params->banded_jacobian_[k*(num_local_points*5) + j*5 + 1 + 2 - 1] *= inverse_chem_jacobian;
      params->banded_jacobian_[k*(num_local_points*5) + j*5 + 1 + 2 + 1] *= inverse_chem_jacobian;
    }
  }

  // Add identity
  for(int j=0; j<num_local_points; ++j) {
    for(int k=0; k<num_states; ++k) {
      params->banded_jacobian_[k*(num_local_points*5) + j*5 + 1 + 2 + 0] += 1.0;
    }
  }

  // Communications to solve banded transport Jacobian
  // Each processor handles the full grid for a subset of species
  MPI_Comm comm = params->comm_;
  long int dsize = num_local_points*5;
  int nodeDest;
  // Gather the Jacobian to have all grid points for each species
  for(int j=0; j<num_states; ++j) {
    nodeDest = j/params->num_states_per_proc_;
    int jlocal = j % params->num_states_per_proc_;
    int start_band = j*(num_local_points*5);
    int start_band2 = jlocal*(num_total_points*5);

    MPI_Gather(&params->banded_jacobian_[start_band],
               dsize,
               PVEC_REAL_MPI_TYPE,
               &params->banded_jacobian2_[start_band2],
               dsize,
               PVEC_REAL_MPI_TYPE,
               nodeDest,
               comm);
  }

  // Reorder
  for(int j=0; j<num_states_local; ++j) {
    for(int i=0; i<num_total_points; ++i) {
      for(int s=0; s<4; ++s) {
        params->banded_jacobian_serial_[j*(num_total_points*4) + i*4 + s] =
          params->banded_jacobian2_[j*(num_total_points*5) + i*5 + s + 1];
      }
    }
  }

  // Factorize for each species
  int one = 1;
  int LDAB = 4;
  int dim = num_total_points;
  for(int j=0; j<num_states_local; ++j) {
    dgbtrf_(&dim,
            &dim,
            &one,
            &one,
            &params->banded_jacobian_serial_[j*num_total_points*4],
            &LDAB,
            &params->pivots_serial_[j*num_total_points],
            &error_flag);
  }

  return error_flag;
}

#if defined SUNDIALS2
int ReactorAFSolve(N_Vector y, // [in] state vector
                   N_Vector yscale, // [in] state scaler
                   N_Vector ydot, // [in] state derivative
                   N_Vector ydotscale, // [in] state derivative scaler
                   N_Vector vv, // [in/out] rhs/solution vector
                   void *user_data, // [in/out]
                   N_Vector tmp)
{
#elif defined SUNDIALS3 || defined SUNDIALS4
int ReactorAFSolve(N_Vector y, // [in] state vector
                   N_Vector yscale, // [in] state scaler
                   N_Vector ydot, // [in] state derivative
                   N_Vector ydotscale, // [in] state derivative scaler
                   N_Vector vv, // [in/out] rhs/solution vector
                   void *user_data) // [in/out]
{
#endif
  double *solution = NV_DATA_P(vv);
  int error_flag = 0;

  error_flag = AFSolve(&solution[0], user_data);
  return error_flag;

}

// Error function handling
void ErrorFunction(int error_code,
                   const char *module,
                   const char *function,
                   char *msg,
                   void *user_data)
{
  FlameParams *params = (FlameParams *)user_data;
  MPI_Comm comm = params->comm_;

  if(params->my_pe_ == 0)
    printf("# KINErr: %s\n",msg);

  //MPI_Abort(comm, error_code);

}

// Solve approximately factorized Jacobian
int AFSolve(double solution[],
            void *user_data)
{
  FlameParams *params = (FlameParams *)user_data;
  const int num_total_points  = params->num_points_;
  const int num_local_points  = params->num_local_points_;
  const int num_states  = params->reactor_->GetNumStates();
  const int num_states_local = params->num_states_local_;
  int error_flag = 0;

  // Solve Local sparse chemistry with SuperLU
  int start_id=0;
  for(int j=0; j<num_local_points; ++j) {
    error_flag = params->sparse_matrix_chem_[j]->Solve(&solution[start_id],
                                                       &solution[start_id]);
    start_id += num_states;
    if(error_flag != 0) {
      printf("AFSolve sparse matrix error: %d\n", error_flag);
      return error_flag;
    }
  }

  // Banded transport
  // Communications for banded_jacobian2
  MPI_Comm comm = params->comm_;
  long int dsize = num_local_points;
  int nodeDest, nodeFrom;

  std::vector<double> solution_allspecies, solution_species;
  solution_allspecies.assign(num_total_points*num_states_local, 0.0);
  solution_species.assign(num_local_points*num_states, 0.0);

  // Reorder solution vector by species
  for(int j=0; j<num_states; ++j)
    for(int i=0; i<num_local_points; ++i)
      solution_species[j*num_local_points+i] = solution[j+i*num_states];

  // Gather all grid points for each species
  for(int j=0; j<num_states; ++j) {
    nodeDest = j/params->num_states_per_proc_;
    int jlocal = j % params->num_states_per_proc_;
    int start_id = j*num_local_points;
    int start_id2 = jlocal*num_total_points;

    MPI_Gather(&solution_species[start_id],
               dsize,
               PVEC_REAL_MPI_TYPE,
               &solution_allspecies[start_id2],
               dsize,
               PVEC_REAL_MPI_TYPE,
               nodeDest,
               comm);
  }

  // Solve banded matrix for each species
  int dim = num_total_points;
  int one = 1;
  int LDAB = 4;
  int LDB = num_total_points;
  for(int j=0; j<num_states_local; ++j) {
    dgbtrs_("N",
            &dim,
            &one,
            &one,
            &one,
            &params->banded_jacobian_serial_[j*num_total_points*4],
            &LDAB,
            &params->pivots_serial_[j*num_total_points],
            &solution_allspecies[j*num_total_points],
            &LDB,
            &error_flag);

    if(error_flag != 0)
      printf("AFSolve banded matrix error: %d\n", error_flag);
  }

  //Scatter back the solution vector for each species
  for(int j=0; j<num_states; ++j) {
    nodeFrom = j/params->num_states_per_proc_;
    int jlocal = j % params->num_states_per_proc_;
    int start_id = j*num_local_points;
    int start_id2 = jlocal*num_total_points;

    MPI_Scatter(&solution_allspecies[start_id2],
                dsize,
                PVEC_REAL_MPI_TYPE,
		&solution_species[start_id],
                dsize,
                PVEC_REAL_MPI_TYPE,
                nodeFrom,
                comm);
  }

  // Reorder solution vector by grid points
  for(int j=0; j<num_states; ++j)
    for(int i=0; i<num_local_points; ++i)
      solution[j+i*num_states] = solution_species[j*num_local_points+i];

  return error_flag;
}

static double FindMaximumParallel(const int num_points,
                                  const double f[],
                                  int *j_at_max,
                                  MPI_Comm comm)
{
  int myrank;
  struct {
    double value;
    int index;
  } in, out;

  // Compute local maximum
  in.value = f[0];
  in.index = 0;
  for(int j=1; j<num_points; ++j) {
    if(in.value < f[j]) {
      in.value = f[j];
      in.index = j;
    }
  }

  // Compute global maximum
  MPI_Comm_rank(comm, &myrank);
  in.index += myrank*num_points;

  MPI_Allreduce(&in,&out,1,MPI_DOUBLE_INT, MPI_MAXLOC, comm);
  *j_at_max = out.index;
  return out.value;
}


static double FindMinimumParallel(const int num_points,
                                  const double f[],
                                  int *j_at_min,
                                  MPI_Comm comm)
{
  int myrank;
  struct {
    double value;
    int index;
  } in, out;

  // Compute local minimum
  in.value = f[0];
  in.index = 0;
  for(int j=1; j<num_points; ++j) {
    if(in.value > f[j]) {
      in.value = f[j];
      in.index = j;
    }
  }

  // Compute global maximum
  MPI_Comm_rank(comm, &myrank);
  in.index += myrank*num_points;

  MPI_Allreduce(&in,&out,1,MPI_DOUBLE_INT, MPI_MINLOC, comm);

  *j_at_min = out.index;

  return out.value;
}
