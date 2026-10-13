#include "sparse_matrix.h"
#include "cvode_functions.h"
#include "flame_params.h"

int sign(const double x)
{
    return (x > 0) ? 1 :
           (x < 0) ? -1
                   : 0;
}

int sign(const int x)
{
    return (x > 0) ? 1 :
           (x < 0) ? -1
                   : 0;
}

static double FindMaximumParallel(const int num_points,
                                  const double f[],
                                  int *j_at_max);

static double FindMinimumParallel(const int num_points,
                                  const double f[],
                                  int *j_at_min);

// Main RHS function
int ConstPressureFlame(realtype t,
		       N_Vector y,
		       N_Vector ydot,
		       void *user_data)
{
  FlameParams *params = (FlameParams *)user_data;
  const int num_local_points = params->num_local_points_;
  const int num_states  = params->reactor_->GetNumStates();
  long int Nlocal = num_local_points*num_states;

  // MPI calls are in Local, no need for Comm function
  ConstPressureFlameLocal(Nlocal, t, y, ydot, user_data);

  return 0;
}

// RHS function
int ConstPressureFlameLocal(long int nlocal,
			    realtype t,
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
  int my_pe = params->my_pe_;
  int npes  = params->npes_;
  int nover = params->nover_;

  const double ref_temperature = params->ref_temperature_;
  const double ref_momentum = params->ref_momentum_;
  const bool finite_separation = params->parser_->finite_separation();

  std::vector<double> enthalpies;
  enthalpies.assign(num_species,0.0);

  // Splitting RHS into chemistry, convection, and diffusion terms
  // for readability/future use
  std::vector<double> rhs_chem, rhs_conv, rhs_diff;
  rhs_chem.assign(num_local_points*num_states,0.0);
  rhs_conv.assign(num_local_points*num_states,0.0);
  rhs_diff.assign(num_local_points*num_states,0.0);

  double cp_flux_sum, mass_fraction_sum;
  double relative_volume_j;
  int transport_error;

  double local_sum;
  double local_max;

  double thermal_diffusivity, continuity_error;

  // set the derivative to zero
  for(int j=0; j<num_local_states; ++j) {
    ydot_ptr[j] = 0.0;
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
  MPI_Comm comm = params->comm_;
  MPI_Status status;
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
    params->mass_flux_ext_[nover+j] = params->mass_flux_[j];
  }

  for (int j=0; j<num_local_points+2*nover; ++j) {
    dz[j] = params->dz_local_[j];
    dzm[j] = params->dzm_local_[j];
    inv_dz[j] = params->inv_dz_local_[j];
    inv_dzm[j] = params->inv_dzm_local_[j];
  }

  // Update ghost cells with send/receive
  int nodeDest = my_pe-1;
  if (nodeDest < 0) nodeDest = npes-1;
  int nodeFrom = my_pe+1;
  if (nodeFrom > npes-1) nodeFrom = 0;
  MPI_Sendrecv(&params->y_ext_[nover*num_states], dsize, PVEC_REAL_MPI_TYPE, nodeDest, 0,
               &params->y_ext_[num_states*(num_local_points+nover)], dsize, PVEC_REAL_MPI_TYPE,
               nodeFrom, 0, comm, &status);

  nodeDest = my_pe+1;
  if (nodeDest > npes-1) nodeDest = 0;
  nodeFrom = my_pe-1;
  if (nodeFrom < 0) nodeFrom = npes-1;
  MPI_Sendrecv(&params->y_ext_[num_states*num_local_points], dsize,
               PVEC_REAL_MPI_TYPE, nodeDest, 0,
               &params->y_ext_[0], dsize, PVEC_REAL_MPI_TYPE, nodeFrom, 0, comm, &status);

  // Apply boundary conditions
  // First proc: fuel conditions in ghost cells
  if (my_pe ==0) {
    for(int j=0; j<nover; ++j) {
      if(params->flame_type_ == 0) {
        for(int k=0; k<num_species; ++k) {
          params->y_ext_[j*num_states + k] = params->fuel_mass_fractions_[k];
        }
        params->y_ext_[j*num_states+num_species]   = params->fuel_relative_volume_;
      } else if (params->flame_type_ == 1 || params->flame_type_ == 2) {
        for(int k=0; k<num_species; ++k) {
          params->y_ext_[j*num_states + k] = params->inlet_mass_fractions_[k];
        }
        params->y_ext_[j*num_states+num_species]   = params->inlet_relative_volume_;
      }
      params->y_ext_[j*num_states+num_species+1] = params->fuel_temperature_;
      if(finite_separation) {
        params->y_ext_[j*num_states+num_species+2] = 0.0;
        params->P_left_ = params->y_ext_[nover*num_states+num_species+3];
        params->y_ext_[j*num_states+num_species+3] = params->P_left_;
      } else {
        params->y_ext_[j*num_states+num_species+2] =
          params->y_ext_[nover*num_states+num_species+2];//zero gradient
      }
      params->mass_flux_ext_[j] = params->mass_flux_fuel_;
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
          params->y_ext_[(num_local_points+nover-1)*num_states+num_species];
        params->oxidizer_temperature_ =
          params->y_ext_[(num_local_points+nover-1)*num_states+num_species+1];
      }
      for(int k=0; k<num_species; ++k) {
        params->y_ext_[j*num_states + k] = params->oxidizer_mass_fractions_[k];
      }
      params->y_ext_[j*num_states+num_species] = params->oxidizer_relative_volume_;
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
      params->mass_flux_ext_[j] = params->mass_flux_oxidizer_;
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
  // Compute mass flux -- Integrate continuity equation
  // dmdot/dx = -beta*rho*G -> mdot = int(-beta*rho*G, dx)
  const double RuTref_p = params->reactor_->GetGasConstant()*
    params->ref_temperature_/params->parser_->pressure();

  // Compute Stagnation plane location
  // ONLY WORKS IN SERIAL FOR NOW
  double *gbuf, *vbuf, *mbuf, *vloc, *gloc;
  gloc = (double *)malloc(num_local_points*sizeof(double));
  vloc = (double *)malloc(num_local_points*sizeof(double));
  for(int j=0; j<num_local_points; j++) {
    gloc[j] = y_ptr[j*num_states + num_species + 2]*ref_momentum;
    vloc[j] = y_ptr[j*num_states + num_species];
    //printf("j: %d, gloc: %5.3e, vloc: %5.3e\n", my_pe*num_local_points+j, gloc[j], vloc[j]);
  }
  //if(my_pe == 0) {
    gbuf = (double *)malloc(num_total_points*sizeof(double));
    vbuf = (double *)malloc(num_total_points*sizeof(double));
    mbuf = (double *)malloc(num_total_points*sizeof(double));
    //}
  // Gather global G, relative_volume, and mass flux on root
  dsize = num_local_points;
  // TODO: Replace MPI_Allgather with MPI_Gather!
  MPI_Allgather(gloc, dsize, PVEC_REAL_MPI_TYPE,
             gbuf, dsize, PVEC_REAL_MPI_TYPE, comm);
  MPI_Allgather(vloc, dsize, PVEC_REAL_MPI_TYPE,
             vbuf, dsize, PVEC_REAL_MPI_TYPE, comm);
  MPI_Allgather(&params->mass_flux_[0], dsize, PVEC_REAL_MPI_TYPE,
             mbuf, dsize, PVEC_REAL_MPI_TYPE, comm);

  if(my_pe == 0) {
    int jContBC = 0;
    if(params->flame_type_ == 0 || params->flame_type_ == 2) {
      // Find grid point closest to old stagnation point location
      int jStart = num_total_points-1;
      for(int j=0; j<num_total_points; j++) {
        if(params->z_[j] > params->stagnation_plane_) {
          jStart = j;
          break;
        }
      }
      jContBC = jStart;
      for(int i=1; jStart+i < num_total_points || jStart-i >= 0; i++) {
        if(jStart+i < num_total_points && sign(mbuf[jStart+i]) != sign(mbuf[jStart])) {
          jContBC = jStart + i;
          break;
        } else if (jStart-i>=0 && sign(mbuf[jStart-i]) != sign(mbuf[jStart])) {
          jContBC = jStart - i;
          break;
        }
      }
      if(jContBC == 0) {
        assert(mbuf[0] <=0);
        params->stagnation_plane_ = params->z_[0] - mbuf[0]*params->dz_[0]/(mbuf[1]-mbuf[0]);
      } else {
        assert(mbuf[jContBC]*mbuf[jContBC-1] <=0 || mbuf[num_total_points-1] >=0); // test opposite sign
        params->stagnation_plane_ = params->z_[jContBC] - mbuf[jContBC]*params->dz_[jContBC]/
          (mbuf[jContBC]-mbuf[jContBC-1]);
      }
    } else if (params->flame_type_ == 1) {
      params->stagnation_plane_ = params->length_;
      jContBC = params->num_points_;
    }

    // Integrate continuity to compute mass flux
    double dVdx0;
    double beta = 1.0 + params->simulation_type_;

    if(finite_separation) {
      // Integrate left to right
      // First point uses fuel BC
      // use midpoint rhoG?
      int jj = 0;
      mbuf[jj] = params->mass_flux_fuel_ - (
        params->y_ext_[(jj+nover)*num_states+num_species+2]/
        params->y_ext_[(jj+nover)*num_states+num_species] +
        params->y_ext_[(jj+nover-1)*num_states+num_species+2]/
        params->y_ext_[(jj+nover-1)*num_states+num_species])
        *ref_momentum*params->dz_[jj];
      // Rest of domain
      for(int j=0; j<num_total_points-1; j++) {
        mbuf[j+1] = mbuf[j] - (gbuf[j]/vbuf[j]+gbuf[j+1]/vbuf[j+1])*params->dz_[j+1];
        /*
        printf("j: %d, mbuf: %5.3e, gbuf: %5.3e, vbuf: %5.3e, dz: %5.3e\n",
                   j,  mbuf[j+1],   gbuf[j+1],   vbuf[j+1],   params->dz_[j+1]);
        */
      }

    } else {
      // Integrate from both sides of stagnation plane and update BC values
      // Stagnation plane
      int jj = jContBC;
      dVdx0 = - beta*gbuf[jj]/vbuf[jj];
      if(jContBC != 0) {
        dVdx0 = 0.5*dVdx0 - 0.5*beta*gbuf[jj-1]/vbuf[jj-1];
      }
      mbuf[jj] = (params->z_[jj]-params->stagnation_plane_) * dVdx0;

      // Right of stagnation plane
      for(int j=jContBC; j<num_total_points-1; j++) {
        mbuf[j+1] = mbuf[j] - beta*gbuf[j]/vbuf[j]*params->dz_[j+1];
      }
      // Right BC
      jj = num_total_points-1;
      params->mass_flux_oxidizer_ = mbuf[jj] - beta*gbuf[jj]/vbuf[jj]*params->dz_[jj+1];

      // Left of stagnation plane
      if(jContBC != 0) {
        jj = jContBC-1;
        mbuf[jj] = (params->z_[jj] - params->stagnation_plane_) * dVdx0;
        for(int j=jContBC-1; j>0; j--) {
          mbuf[j-1] = mbuf[j] + beta*gbuf[j-1]/vbuf[j-1]*params->dz_[j];
        }
      }
      jj = 0;
      params->mass_flux_fuel_ = mbuf[jj] +
        beta*params->y_ext_[(jj+nover-1)*num_states+num_species+2]*ref_momentum/
        params->y_ext_[(jj+nover-1)*num_states+num_species]*params->dz_[jj];
    }
  } // End mass flux integration on root

  // Scatter mass flux to all procs
  MPI_Scatter(mbuf,
              dsize,
              PVEC_REAL_MPI_TYPE,
              &params->mass_flux_[0],
              dsize,
              PVEC_REAL_MPI_TYPE,
              0,
              comm);

  // Broadcast mass flux BCs and stagnation plane
  MPI_Bcast(&params->mass_flux_fuel_, 1, MPI_DOUBLE, 0, comm);
  MPI_Bcast(&params->mass_flux_oxidizer_, 1, MPI_DOUBLE, 0, comm);
  MPI_Bcast(&params->stagnation_plane_, 1, MPI_DOUBLE, 0, comm);

  //  Copy into extended vector
  for(int j=0; j<num_local_points; ++j) {
    int jext = j + nover;
    params->mass_flux_ext_[jext] = params->mass_flux_[j];
  }

  // Parallel communication of mass flux
  dsize = nover;
  nodeDest = my_pe-1;
  if (nodeDest < 0) nodeDest = npes-1;
  nodeFrom = my_pe+1;
  if (nodeFrom > npes-1) nodeFrom = 0;
  MPI_Sendrecv(&params->mass_flux_ext_[nover], dsize, PVEC_REAL_MPI_TYPE, nodeDest, 0,
               &params->mass_flux_ext_[num_local_points+nover], dsize,
               PVEC_REAL_MPI_TYPE, nodeFrom, 0, comm, &status);

  nodeDest = my_pe+1;
  if (nodeDest > npes-1) nodeDest = 0;
  nodeFrom = my_pe-1;
  if (nodeFrom < 0) nodeFrom = npes-1;
  MPI_Sendrecv(&params->mass_flux_ext_[num_local_points], dsize,
               PVEC_REAL_MPI_TYPE, nodeDest, 0, &params->mass_flux_ext_[0],
               dsize, PVEC_REAL_MPI_TYPE, nodeFrom, 0, comm, &status);

  // Add BCs to mass_flux_ext
  // First proc: fuel conditions in ghost cells
  if (my_pe ==0) {
    for(int j=0; j<nover; ++j) {
      params->mass_flux_ext_[j] = params->mass_flux_fuel_;
    }
  }

  // Last proc: oxidizer conditions in ghost cells
  if (my_pe == npes-1) {
    for(int j=num_local_points+nover; j<num_local_points+2*nover; ++j) {
        params->mass_flux_ext_[j] = params->mass_flux_oxidizer_;
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
        &params->species_specific_heats_[num_species*j]);

    // Reset species cp
    for(int k=0; k<num_species; k++) {
      params->species_specific_heats_[num_species*j+k] = 0.0;
    }

    // specific heat at grid point j
    params->mixture_specific_heat_[j] =
      params->reactor_->GetMixtureSpecificHeat_Cp(
        ref_temperature*params->y_ext_[jext*num_states+num_species+1],
        &params->y_ext_[jext*num_states],
        &params->species_specific_heats_[num_species*j]);

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

    // compute the species diffusion mass flux at the upstream mid point
    // always use corrected diffusion flux in unsteady solver
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
  // Compute convective and diffusive terms for species, temperature, and momentum
  for(int j=0; j<num_local_points; ++j) {
    int jext = j + nover;
    int jglobal = j + my_pe*num_local_points;

    relative_volume_j  = params->y_ext_[jext*num_states+num_species];

    double a=0,b=0,c=0,d=0,e=0;; //coefficients of j+2, j+1, j, j-1, j-2 terms
    if(convective_scheme_type == 0) {
      // First order upwind
      if(params->mass_flux_ext_[jext]*relative_volume_j > 0) {
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
      if (params->mass_flux_ext_[jext]*relative_volume_j > 0) {
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
	 c*params->y_ext_[ jext  *num_states+k] +
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

    // Momentum equation
    // Compute the momentum advection term (will be multiplied by mass flux)
    rhs_conv[j*num_states+num_species+2] -= relative_volume_j*
      (a*params->y_ext_[(jext+2)*num_states+num_species+2] +
       b*params->y_ext_[(jext+1)*num_states+num_species+2] +
       c*params->y_ext_[ jext   *num_states+num_species+2] +
       d*params->y_ext_[(jext-1)*num_states+num_species+2] +
       e*params->y_ext_[(jext-2)*num_states+num_species+2]);

    // Compute momentum strain term P
    if(finite_separation) {
      rhs_diff[j*num_states+num_species+2] -= params->y_ext_[jext*num_states+num_species+3]
        *relative_volume_j;
    } else {
      //rhoInf*a^2/beta^2
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
    // Pstrain is imposed for infinite separation
    if(finite_separation) {
      if (jglobal == (npes*num_local_points-1)) { //lastpoint: dV/dx+beta*rho*G
        rhs_diff[j*num_states+num_species+3] =
          (params->mass_flux_ext_[jext+1]-params->mass_flux_ext_[jext])*inv_dz[jext+1] +
          (params->y_ext_[(jext+1)*num_states+num_species+2]/
           params->y_ext_[(jext+1)*num_states+num_species] +
           params->y_ext_[jext*num_states+num_species+2]/
           params->y_ext_[(jext+1)*num_states+num_species])*ref_momentum;
      } else { // dP/dx
        rhs_diff[j*num_states+num_species+3] =
          (params->y_ext_[(jext+1)*num_states+num_species+3] -
           params->y_ext_[jext*num_states+num_species+3])*inv_dz[jext+1]*100;
        // P adjusts faster with *100 but maybe it makes the ODEs stiffer/harder to solve?
      }
    }

  } // for(int j=0; j<num_local_points; ++j) // loop computing rhs


  // -------------------------------------------------------------------------
  // Compute the rate of change of the relative volume using the ideal
  // gas equation of state, and the FINAL derivatives of temperature and
  // mass fractions.
  //
  // dv/dt = v/T * dT/dt + RuT/p * \sum_i (1/mw[i] * dy[i]/dt)
  // dv/dt   current units [m^3/kg/s]
  for(int j=0; j<num_local_points; ++j) {
    int rvol_id  = j*num_states+num_species; // relative volume index of pt j
    int temp_id  = rvol_id+1;                // temperature index of pt j
    int mom_id   = rvol_id+2;                // momentum index of pt j

    mass_fraction_sum = 0.0;
    for(int k=0; k<num_species; ++k) {
      ydot_ptr[j*num_states+k] = rhs_conv[j*num_states+k]*params->mass_flux_[j]
        + rhs_chem[j*num_states+k] + rhs_diff[j*num_states+k];

      mass_fraction_sum += params->inv_molecular_mass_[k]*ydot_ptr[j*num_states+k];
    }

    ydot_ptr[temp_id] = rhs_conv[temp_id]*params->mass_flux_[j]
      + rhs_chem[temp_id] + rhs_diff[temp_id];

    ydot_ptr[rvol_id] = y_ptr[rvol_id]*ydot_ptr[temp_id]/y_ptr[temp_id] +
      RuTref_p*y_ptr[temp_id]*mass_fraction_sum;

    ydot_ptr[mom_id] = rhs_conv[mom_id]*params->mass_flux_[j] + rhs_diff[mom_id];

    if(finite_separation) {
      int strain_id = rvol_id+3;               // strain index of pt j
      ydot_ptr[strain_id] = rhs_diff[strain_id];
    }
  }

  // -------------------------------------------------------------------------
  // For output to the screen/logfile
  params->mass_change_ = 0.0;

  // Compute fuel burning rate/laminar flame speed = int(omega_F)/rho_u/YF_u
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


  // Compute characteristic strain rate
  // Compute normal strain rate (dv/dz)
  std::vector<double> strain_rate_abs, velocity;
  strain_rate_abs.assign(num_local_points, 0.0);
  velocity.assign(num_local_points, 0.0);
  for(int j=0; j<num_local_points; ++j) {
    int jext = j + nover;
    velocity[j] = params->y_ext_[jext*num_states+num_species]*params->mass_flux_ext_[jext];
    strain_rate_abs[j] = fabs(
      (params->y_ext_[(jext+1)*num_states+num_species]*params->mass_flux_ext_[jext+1] -
       velocity[j])*inv_dz[jext]);
  }

  if(finite_separation) {
    // Method 1: Sometimes fails at high strain rates
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

    // Method 2: highest absolute value before "turnaround" point
    // ONLY WORKS IN SERIAL FOR NOW
    long int dsize;
    double *sbuf;
    if(my_pe == 0)
      sbuf = (double *)malloc(num_local_points*npes*sizeof(double));

    // Gather strain rate on root
    dsize = num_local_points;
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
    MPI_Bcast(&params->strain_rate_, 1, MPI_DOUBLE, 0, comm);
  }

  // initialize the max variables used for explicit time step information
  // compute the max velocity from the mass flux and relative volume stored
  // in the state vector
  local_max = 0.0;
  for(int j=0; j<num_local_points; ++j) {
    if(velocity[j] > local_max) {
      local_max = velocity[j];
    }
  }
  MPI_Allreduce(&local_max,&params->max_velocity_,1,PVEC_REAL_MPI_TYPE,MPI_MAX,comm);

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
		 (params->y_ext_[ jext   *num_states+num_species+1] -
                  params->y_ext_[(jext-1)*num_states+num_species+1]));
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
           y_ptr[j*num_states+num_species]/params->mixture_specific_heat_[j]);
    if(thermal_diffusivity > local_max) {
      local_max = thermal_diffusivity;
    }
  }
  MPI_Allreduce(&local_max,&params->max_thermal_diffusivity_,1,PVEC_REAL_MPI_TYPE,MPI_MAX,comm);

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
  double *y_ptr          = NV_DATA_P(y); //_S // caution: assumes realtype == double
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
  double *rhs         = NV_DATA_P(r);  // pointers to data array for N_Vector
  double *solution    = NV_DATA_P(z);  // pointers to data array for N_Vector
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

static double FindMinimumAbsParallel(const size_t num_points,
                                     const double x[],
                                     const size_t x_stride,
                                     const double f[],
                                     const size_t f_stride,
                                     const bool use_quadratic,
                                     double *x_at_min,
                                     int *j_at_min)
{
  int myrank;
  struct {
    double value;
    int index;
  } in, out;

  // Compute local minimum of |f|
  in.value = fabs(f[0]);
  in.index = 0;
  for(int j=1; j<(int)num_points; ++j) {
    if(in.value > fabs(f[j*f_stride]) ) {
      in.value = fabs(f[j*f_stride]);
      in.index = j;
    }
  }

  // Compute global minimum
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  in.index += myrank*num_points;

  MPI_Allreduce(&in,&out,1,MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

  *x_at_min = x[out.index];
  *j_at_min = out.index;

  return out.value;
}

static double FindMaximumParallel(const int num_points,
                                  const double f[],
                                  int *j_at_max)
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
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  in.index += myrank*num_points;

  MPI_Allreduce(&in,&out,1,MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
  *j_at_max = out.index;
  return out.value;
}

static double FindMinimumParallel(const int num_points,
                                  const double f[],
                                  int *j_at_min)
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
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  in.index += myrank*num_points;

  MPI_Allreduce(&in,&out,1,MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

  *j_at_min = out.index;

  return out.value;
}
