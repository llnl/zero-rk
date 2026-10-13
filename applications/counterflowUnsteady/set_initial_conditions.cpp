#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <fstream>
#include <utilities/string_utilities.h>
#include <utilities/math_utilities.h>
#include <utilities/file_utilities.h>
#include <utilities/restart_reader.h>

#include "set_initial_conditions.h"

#include "equil/zerork_equilibrium_solver.h"


// Set initial composition
void SetInitialComposition(FlameParams &flame_params, double *y, double *time)
{
  const int num_points = flame_params.z_.size();
  const int num_local_points  = flame_params.num_local_points_;
  const int num_states  = flame_params.reactor_->GetNumStates();
  const int num_species = flame_params.reactor_->GetNumSpecies();

  const double pressure = flame_params.parser_->pressure();
  const double ref_temperature = flame_params.parser_->ref_temperature();
  const double ref_momentum = flame_params.parser_->ref_momentum();

  const int nover = flame_params.nover_;

  int my_pe = flame_params.my_pe_;
  MPI_Comm comm = flame_params.comm_;

  double species_left, species_right, species_stoich, species;
  double temperature_left, temperature_right, temperature,
    temperature_stoich, temperature_eq;
  double velocity_fuel, velocity_oxidizer;
  double thickness;
  double relative_volume, relative_volume_eq;
  double mixture_molecular_mass = 0.0;
  std::vector<double> species_molecular_mass;
  std::vector<double> eq_mass_fractions;
  double length = flame_params.length_;
  double strain_rate, x0, f, conductivity, cp, zeta, zmix, zst;
  double sumY;
  int transport_error;

  double beta = 2.0; //1.0 + flame_params.simulation_type_; // only axisymmetric

  std::string state_name= "MassFraction_OH";
  int OH_id = flame_params.reactor_->GetIdOfState(state_name.c_str());

  species_molecular_mass.assign(num_species, 0.0);
  flame_params.reactor_->GetSpeciesMolecularWeight(&species_molecular_mass[0]);

  eq_mass_fractions.assign(num_species, 0.0);

  temperature_stoich = flame_params.parser_->ignition_temperature()/ref_temperature;

  x0 = length*0.5; //put flame in the middle

  double centerWidth = flame_params.parser_->thickness();
  double slopeWidth = flame_params.parser_->thickness();
  double scale = 0.8*length / (centerWidth + 2*slopeWidth);
  if(scale < 1.0) {
    centerWidth *= scale;
    slopeWidth *= scale;
  }
  double dz = flame_params.z_[int(num_points/2)] - flame_params.z_[int(num_points/2)-1];
  int centerPointCount = int(0.5 + 0.5*centerWidth/dz);
  int slopePointCount = int(0.5 + slopeWidth/dz);
  int jm = int(num_points /2);
  int jl2 = jm - centerPointCount;
  int jl1 = jl2 - slopePointCount;
  int jr1 = jm + centerPointCount;
  int jr2 = jr1 + slopePointCount;

  if(flame_params.flame_type_ == 1) {
    x0 *= 2;
    jl1 *= 2;
    jl2 *= 2;
    jr1 *= 2;
    jr2 *= 2;
  }

  zst = flame_params.stoichiometric_mixture_fraction_;

#ifdef ZERORK_HAVE_EQUILIBRIUM_SOLVER
  if(flame_params.parser_->use_equilibrium_for_init()) {
    // ------------ BEGIN Constrained Equibrium calc  -----------//
    // Only on root
    if(my_pe == 0) {
      if(flame_params.flame_type_ == 0) {
        temperature_eq = 0.5*(flame_params.fuel_temperature_ + flame_params.oxidizer_temperature_)*
                         flame_params.ref_temperature_;
      } else {
        temperature_eq = flame_params.fuel_temperature_*flame_params.ref_temperature_;
      }
      for(int k=0; k<num_species; k++) {
        if(flame_params.flame_type_ == 0) {
          eq_mass_fractions[k] = zst*flame_params.fuel_mass_fractions_[k]
                                 + (1.0-zst)*flame_params.oxidizer_mass_fractions_[k];
        } else {
          eq_mass_fractions[k] = flame_params.inlet_mass_fractions_[k];
        }
      }
      zerork::equilibrium_solver zeq(*flame_params.mechanism_);
      zeq.equilibrate(pressure, temperature_eq,
                      &eq_mass_fractions[0]);
      temperature_eq /= flame_params.ref_temperature_;
    }
    // Broadcast eq. T & Y to all procs
    MPI_Bcast(&temperature_eq, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&eq_mass_fractions[0], num_species, MPI_DOUBLE, 0, comm);
    // ------------ END Constrained Equibrium calc  -----------//
  } else
#else
  if(flame_params.parser_->use_equilibrium_for_init()) {
    if(my_pe == 0) {
      printf("No equilibrium interface enabled.  Please re-configure/re-build Zero-RK to enable CEQ or Cantera equilibrium interface.\n");
      printf("Falling back to \"simple\" initialization\n");
    }
  }
#endif
  {
    // "Simple" initialization sometimes works better than EQ
    sumY = 0.0;
    for(int k=0; k<num_species; ++k){
      if(flame_params.flame_type_ == 0) {
        eq_mass_fractions[k] = zst*flame_params.fuel_mass_fractions_[k] +
          (1.0-zst)*flame_params.oxidizer_mass_fractions_[k];
      } else {
        eq_mass_fractions[k] = flame_params.inlet_mass_fractions_[k];
      }

      if(k==OH_id)
        eq_mass_fractions[k] += 0.003; // add OH to help ignition
      sumY += eq_mass_fractions[k];
    }
    for(int k=0; k<num_species; ++k)
      eq_mass_fractions[k] /= sumY;

    temperature_eq = temperature_stoich;
  }

  // Renormalize and compute mixture weight
  mixture_molecular_mass = 0.0;
  for(int k=0; k<num_species; ++k) {
    mixture_molecular_mass += eq_mass_fractions[k]/species_molecular_mass[k];
  }
  mixture_molecular_mass = 1.0/mixture_molecular_mass;

  // Relative volume
  relative_volume_eq = flame_params.reactor_->GetGasConstant()*
    ref_temperature*temperature_eq/(pressure*mixture_molecular_mass);


  // Compute inlet velocities to approximate strain rate
  if(flame_params.flame_type_ == 0) {
    velocity_fuel = flame_params.mass_flux_fuel_*flame_params.fuel_relative_volume_;
    velocity_oxidizer = flame_params.mass_flux_oxidizer_*flame_params.oxidizer_relative_volume_;
  } else if (flame_params.flame_type_ == 1) {
    velocity_fuel = flame_params.mass_flux_fuel_*flame_params.inlet_relative_volume_;
    velocity_oxidizer = 0.0;
  } else if (flame_params.flame_type_ == 2) {
    velocity_fuel = flame_params.mass_flux_fuel_*flame_params.inlet_relative_volume_;
    velocity_oxidizer = flame_params.mass_flux_oxidizer_*relative_volume_eq;
  }

  if(flame_params.parser_->finite_separation()) {
    strain_rate = (fabs(velocity_fuel) +
                   fabs(velocity_oxidizer))/length;
  } else {
    strain_rate = flame_params.strain_rate_;
  }

  // For R-to-P flame, overwrite oxidizer quantities with eq. quantities
  if(flame_params.flame_type_ == 2) {
    flame_params.oxidizer_temperature_ = temperature_eq;
    flame_params.oxidizer_relative_volume_ = relative_volume_eq;
    for(int k=0; k<num_species; ++k) {
      flame_params.oxidizer_mass_fractions_[k] = eq_mass_fractions[k];
    }
  }

  // Set right/left T
  if(flame_params.flame_type_ == 0 || flame_params.flame_type_ == 2) {
    temperature_right = flame_params.oxidizer_temperature_;
    temperature_left = flame_params.fuel_temperature_;
  } else if (flame_params.flame_type_ == 1) {
    temperature_left = flame_params.fuel_temperature_;
    temperature_right= flame_params.fuel_temperature_;
  }

  for(int j=0; j<num_local_points; ++j) {
    int jglobal = j + my_pe*num_local_points;
    int jext = j + nover;

    double ramp = 0.0;
    if(jglobal <= jl1) {
      temperature = temperature_left;
      for(int k=0; k<num_species; ++k) {
        if(flame_params.flame_type_ == 0) {
          y[j*num_states+k] = flame_params.fuel_mass_fractions_[k];
        } else {
          y[j*num_states+k] = flame_params.inlet_mass_fractions_[k];
        }
      }
    } else if (jglobal > jl1 && jglobal <= jl2) {
      ramp = ((double)jglobal-(double)jl1)/((double)jl2-(double)jl1);
      temperature = temperature_left + (temperature_eq-temperature_left)*ramp;
      for(int k=0; k<num_species; ++k) {
        if(flame_params.flame_type_ == 0) {
          y[j*num_states+k] = flame_params.fuel_mass_fractions_[k] +
            (eq_mass_fractions[k]-flame_params.fuel_mass_fractions_[k])*ramp;
        } else {
          y[j*num_states+k] = flame_params.inlet_mass_fractions_[k] +
            (eq_mass_fractions[k]-flame_params.inlet_mass_fractions_[k])*ramp;
        }
      }
    } else if (jglobal > jl2 && jglobal <= jr1) {
      temperature = temperature_eq;
      for(int k=0; k<num_species; ++k) {
        y[j*num_states+k] = eq_mass_fractions[k];}
    } else if (jglobal > jr1 && jglobal <= jr2) {
      ramp = ((double)jglobal-(double)jr1)/((double)jr2-(double)jr1);
      temperature = temperature_eq - (temperature_eq-temperature_right)*ramp;
      for(int k=0; k<num_species; ++k) {
        if(flame_params.flame_type_ == 0) {
          y[j*num_states+k] = eq_mass_fractions[k] -
            (eq_mass_fractions[k]-flame_params.oxidizer_mass_fractions_[k])*ramp;
        } else {
          y[j*num_states+k] = eq_mass_fractions[k] -
            (eq_mass_fractions[k]-flame_params.inlet_mass_fractions_[k])*ramp;
        }
      }
    } else {
      temperature = temperature_right;
      for(int k=0; k<num_species; ++k) {
        if(flame_params.flame_type_ == 0) {
          y[j*num_states+k] = flame_params.oxidizer_mass_fractions_[k];
        } else {
          y[j*num_states+k] = flame_params.inlet_mass_fractions_[k];
        }
      }
    }

    // Renormalize and compute mixture weight
    sumY = 0.0;
    for(int k=0; k<num_species; ++k)
      sumY += y[j*num_states+k];

    mixture_molecular_mass = 0.0;
    for(int k=0; k<num_species; ++k) {
      y[j*num_states+k] /= sumY;
      mixture_molecular_mass += y[j*num_states+k]/species_molecular_mass[k];
    }
    mixture_molecular_mass = 1.0/mixture_molecular_mass;

    // Relative volume
    relative_volume = flame_params.reactor_->GetGasConstant()*
      ref_temperature*temperature/(pressure*mixture_molecular_mass);

    y[j*num_states + num_species] = relative_volume;

    // Mass flux
    if(flame_params.parser_->finite_separation()) {
      flame_params.mass_flux_[j] = velocity_fuel + flame_params.z_[jglobal]/length*
        (-fabs(velocity_oxidizer)-velocity_fuel);
      flame_params.mass_flux_[j] /= relative_volume;
    }

    // Temperature
    y[j*num_states + num_species+1] = temperature;

    // Momentum
    if(flame_params.parser_->finite_separation()) {
      if(flame_params.z_[jglobal] <= x0)
        y[j*num_states + num_species+2] = flame_params.z_[jglobal]/x0*strain_rate/ref_momentum;
      else
        y[j*num_states + num_species+2] =
          (strain_rate-(flame_params.z_[jglobal]-x0)/(length-x0)*strain_rate)/ref_momentum;
    } else {
      y[j*num_states + num_species+2] = flame_params.strain_rate_/beta*
        sqrt(relative_volume/flame_params.oxidizer_relative_volume_)/
        ref_momentum;
    }

    if(flame_params.parser_->finite_separation()) {
      // Pstrain/lambda -- initialize to a^2/beta^2
      y[j*num_states + num_species + 3] = -strain_rate*strain_rate/beta/beta/ref_momentum;
    }

  } // j spatial loop

  // gather y to have all grid points
  if(!flame_params.parser_->finite_separation()) {
    std::vector<double> mass_flux;
    mass_flux.assign(num_points,0.0);

    long int dsize = num_local_points*num_states;
    double *ybuf;
    if(my_pe == 0)
      ybuf = (double *)malloc(flame_params.npes_*dsize*sizeof(double));

    int nodeDest, nodeFrom;
    nodeDest = 0; // gather to root
    nodeFrom = 0; // scatter from root

    MPI_Gather(&y[0],
               dsize,
               PVEC_REAL_MPI_TYPE,
               ybuf,
               dsize,
               PVEC_REAL_MPI_TYPE,
               nodeDest,
               flame_params.comm_);

    if(my_pe == 0){
      // Mass flux
      int stag;
      if(flame_params.flame_type_ == 0 || flame_params.flame_type_ == 2) {
        stag = num_points/4;
      } else if (flame_params.flame_type_ == 1) {
        stag = num_points-1;
      }

      mass_flux[stag] = 0.0;
      for(int j=stag+1; j<num_points; j++) {
        mass_flux[j] = mass_flux[j-1] -
          ybuf[j*num_states + num_species+2]/ybuf[j*num_states + num_species]*
          beta*flame_params.dz_[j]*ref_momentum;
      }

      for(int j=stag-1; j>-1; j--) {
        mass_flux[j] = mass_flux[j+1] +
          ybuf[j*num_states + num_species+2]/ybuf[j*num_states + num_species]*
          beta*flame_params.dz_[j+1]*ref_momentum;
      }
    }

    dsize = num_local_points;
    MPI_Scatter(&mass_flux[0],
                dsize,
                PVEC_REAL_MPI_TYPE,
                &flame_params.mass_flux_[0],
                dsize,
                PVEC_REAL_MPI_TYPE,
                nodeFrom,
                flame_params.comm_);
  }

  // Local mass flux vector
  for(int j=0; j<num_local_points; j++) {
    int jext = j + nover;
    flame_params.mass_flux_ext_[jext] =  flame_params.mass_flux_[j];
  }

  // Restart using binary file if provided
  if(flame_params.parser_->restart_file() != std::string(zerork::utilities::null_filename)) {
    int offset = 0;
    if(flame_params.my_pe_ == 0) {
      offset = 1;
    }
    int buffer_size = num_local_points+offset;
    if(flame_params.my_pe_ == flame_params.npes_-1) {
      buffer_size += 1;
    }

#ifdef ZERORK_MPI
    zerork::utilities::RestartReader reader(flame_params.parser_->restart_file(), buffer_size, flame_params.comm_);
#else
    zerork::utilities::RestartReader reader(flame_params.parser_->restart_file(), buffer_size);
#endif
    zerork::utilities::RestartReaderStatus read_status = reader.Status();
    if(read_status == zerork::utilities::READER_OK) {
      if(flame_params.my_pe_ == 0) cout << "# Restart file: " << flame_params.parser_->restart_file() <<"\n";
      *time = reader.RestartTime();

      // Initialize y to 0
      for (int k=0; k<num_local_points*num_states; ++k) {
        y[k] = 0.0;
      }

      // Read data for each variable
      std::vector<double> buffer(buffer_size, 0.0);
      for(int j=0; j<num_states; ++j) {
        std::string state_name = flame_params.reactor_->GetNameOfStateId(j);
        reader.ReadField(state_name, buffer, &read_status);

        if(read_status == zerork::utilities::READER_OK) {
          // Set y values
          for (int k=0; k<num_local_points; ++k) {
            y[k*num_states + j] = buffer[k+offset];
            if(j==num_species+1){
              y[k*num_states + j] /= flame_params.ref_temperature_;
            }
            if(j==num_species+2){
              y[k*num_states + j] /= flame_params.ref_momentum_;
            }
            if(j==num_species+3){
              y[k*num_states + j] /= flame_params.ref_momentum_;
            }
          }
        } else {
          if(read_status == zerork::utilities::READER_MISSING_FIELD) {
            // Variable not found
            if(flame_params.my_pe_ == 0) cerr << "WARNING: " << state_name << " not found in restart file.\n";
          } else {
            if(flame_params.my_pe_ == 0) cerr << "WARNING: unknown error reading " << state_name << " from restart file.\n";
	  }
        }
      } // for j<num_states

      // Read mass flux
      reader.ReadField(std::string("mass_flux"), buffer, &read_status);
      if(read_status == zerork::utilities::READER_MISSING_FIELD) {
        if(flame_params.my_pe_ == 0) cerr << "WARNING: mass flux not found in restart file.\n";
      }

      for (int k=0; k<num_local_points; ++k) {
        flame_params.mass_flux_[k] = buffer[k+offset];
        flame_params.mass_flux_ext_[k+2] = flame_params.mass_flux_[k];
      }
    } else {
      if(flame_params.my_pe_ == 0) cerr << "ERROR reading restart file:" << flame_params.parser_->restart_file() <<". Quiting.\n";
      exit(-1);
    }
  } // if restart_file

}
