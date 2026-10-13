#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <fstream>
#include <utilities/string_utilities.h>
#include <utilities/math_utilities.h>
#include <utilities/file_utilities.h>
#include <utilities/restart_reader.h>

#include "set_initial_conditions.h"


// Set the mass fractions and temperature to the fuel composition
// if the grid point position is less than or equal to the
// stoichiometric mixture fraction and to the oxidizer composition
// otherwise.
void SetInitialCompositionAndWallTemp(FlameParams &flame_params, double *y, double *time)
{
  //  const int num_points  = flame_params.z_.size();
  const int num_local_points  = flame_params.num_local_points_;
  const int num_states  = flame_params.reactor_->GetNumStates();
  const int num_species = flame_params.fuel_mass_fractions_.size();

  const double pressure = flame_params.parser_->pressure();
  const double ref_temperature = flame_params.parser_->ref_temperature();

  int my_pe = flame_params.my_pe_;

  double species_left, species_right, delta_species, species;
  double temperature_left, temperature_right, delta_temperature_left,
    delta_temperature_right, temperature, temperature_stoich;
  double thickness;
  double mixture_molecular_mass = 0.0;
  std::vector<double> species_molecular_mass;

  // Add OH to facilitate ignition
  std::string state_name= "MassFraction_OH";
  int OH_id = flame_params.reactor_->GetIdOfState(state_name.c_str());

  species_molecular_mass.assign(num_species, 0.0);
  flame_params.reactor_->GetSpeciesMolecularWeight(&species_molecular_mass[0]);

  temperature_right = flame_params.fuel_temperature_;
  temperature_left = flame_params.oxidizer_temperature_;
  temperature_stoich = flame_params.parser_->ignition_temperature()/ref_temperature;
  delta_temperature_left = temperature_stoich - temperature_left;
  delta_temperature_right = temperature_stoich - temperature_right;

  thickness = flame_params.parser_->thickness();

  for(int j=0; j<num_local_points; ++j) {
    int jglobal = j + my_pe*num_local_points;

    if(flame_params.fix_temperature_) {
      temperature = flame_params.fixed_temperature_[jglobal];
    } else {
      if(flame_params.z_[jglobal] <= flame_params.stoichiometric_mixture_fraction_) {
        temperature = temperature_left + delta_temperature_left*flame_params.z_[jglobal]/
          flame_params.stoichiometric_mixture_fraction_;
      } else {
        temperature = temperature_stoich - delta_temperature_right*
          (flame_params.z_[jglobal]-flame_params.stoichiometric_mixture_fraction_)/
          (1.0-flame_params.stoichiometric_mixture_fraction_);
      }
    }

    for(int k=0; k<num_species; ++k) {
      species_right = flame_params.fuel_mass_fractions_[k];
      species_left= flame_params.oxidizer_mass_fractions_[k];
      delta_species = species_right - species_left;

      species = (species_left + 0.5*delta_species) +
        0.5*delta_species*tanh(0.5*(flame_params.z_[jglobal]-flame_params.stoichiometric_mixture_fraction_)/thickness);

      y[j*num_states+k] = species;
    }

    // ADD OH
    if(abs(flame_params.z_[jglobal] -flame_params.stoichiometric_mixture_fraction_) <= 2.0*thickness) {
      y[j*num_states + OH_id] += 0.003;
    }

    // Renormalize and compute mixture weight
    double sumY = 0.0;
    for(int k=0; k<num_species; ++k)
      sumY += y[j*num_states+k];

    mixture_molecular_mass = 0.0;
    for(int k=0; k<num_species; ++k) {
      y[j*num_states+k] /= sumY;
      mixture_molecular_mass += y[j*num_states+k]/species_molecular_mass[k];
    }
    mixture_molecular_mass = 1.0/mixture_molecular_mass;

    y[j*num_states + num_species]   = flame_params.reactor_->GetGasConstant()*ref_temperature*
      temperature/(pressure*mixture_molecular_mass);

    y[j*num_states + num_species + 1] = temperature;

  } // j spatial loop


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
            if(j==num_states-1){
              y[k*num_states + j] /= flame_params.ref_temperature_;
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
    } else {
      if(flame_params.my_pe_ == 0) cerr << "ERROR reading restart file:" << flame_params.parser_->restart_file() <<". Skipping.\n";
    }
  } // if restart_file

}
