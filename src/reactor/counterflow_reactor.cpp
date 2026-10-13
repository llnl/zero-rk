#include <stdlib.h> // exit()
#include <math.h>

#include <string>   // C++ std::string

#include "reactor_base.h"
#include "counterflow_reactor.h"


// implementation class for the CounterflowReactor
class CounterflowReactor::Impl: public ReactorBase
{
 public:
  // member functions that must be defined for the ReactorBase class
  Impl(const char mechanism_name[],
       const char thermodynamics_name[],
       const char parser_log_name[],
       const MatrixType matrix_type,
       const double pressure,
       const bool finite_separation);

  ~Impl();

  ReactorError GetTimeDerivative(const double reactor_time,
                                 const double state[],
				 double derivative[]);

  ReactorError GetTimeDerivativeLimiter(const double reactor_time,
					const double state[],
					const double step_limiter[],
					double derivative[]);

  ReactorError GetTimeDerivativeSteady(const double state[],
                                       const double step_limiter[],
				       double derivative[]);

  ReactorError GetJacobian(const double reactor_time,
                           const double state[],
			   double jacobian[]);

  ReactorError GetJacobianSteady(const double state[],
				 const double convective[],
				 const bool Tfix,
                                 const double ref_momentum,
                                 const double step_limiter[],
				 double jacobian[]);

  ReactorError GetJacobianLimiter(const double reactor_time,
				  const double state[],
				  const double step_limiter[],
				  double jacobian[]);

  int GetJacobianPattern(int row_id[],
                         int column_id[]);
  // --------------------------------------
  // additional functions specific to the reactor type
  void SetReferenceTemperature(const double ref_temperature);
  double GetReferenceTemperature() const;
  void SetPressure(const double pressure);
  double GetPressure() const;

  // specific heat calculation
  double GetMixtureSpecificHeat_Cp(const double state[]);
  double GetMixtureSpecificHeat_Cp(const double state[],
                                   double *species_cp);
  double GetMixtureSpecificHeat_Cp(const double temperature,
                                   const double mass_fractions[],
                                   double *species_cp);

 private:
  int BuildSparseJacobianArrays();
  ReactorError GetSparseJacobian(const double reactor_time,
                                 const double state[],
			         double jacobian[]);
  ReactorError GetSparseJacobianSteady(const double state[],
				       const double convective[],
				       const bool Tfix,
                                       const double ref_momentum,
                                       const double step_limiter[],
				       double jacobian[]);
  ReactorError GetSparseJacobianLimiter(const double reactor_time,
					const double state[],
					const double step_limiter[],
					double jacobian[]);
  void GetJacobianColumnFromPerturbation(const int column_id,
                                         const double dstate,
                                         const double reactor_time,
                                         const double original_state[],
                                         const double original_derivative[],
                                         double perturbed_state[],
                                         double perturbed_derivative[],
                                         double jacobian_column[]);
  void GetJacobianColumnFromPerturbationSteady(const int column_id,
					       const double dstate,
					       const double original_state[],
					       const double original_derivative[],
                                               const double step_limiter[],
					       double perturbed_state[],
					       double perturbed_derivative[],
					       double jacobian_column[]);
  void GetJacobianColumnFromPerturbationLimiter(const int column_id,
						const double dstate,
						const double reactor_time,
						const double original_state[],
						const double original_derivative[],
						const double step_limiter[],
						double perturbed_state[],
						double perturbed_derivative[],
						double jacobian_column[]);
  int GetNetStoichiometry(const int species_id, const int step_id);

  double ref_temperature_;
  double inv_ref_temperature_;
  double pressure_;
  bool use_scaled_state_;
  bool finite_separation_;

  // time derivative storage arrays
  std::vector<double> concentrations_;
  std::vector<double> enthalpies_;
  std::vector<double> specific_heats_;
  std::vector<double> net_reaction_rates_;
  std::vector<double> creation_rates_;
  std::vector<double> destruction_rates_;
  std::vector<double> step_rates_;

  // Jacobian storage arrays
  std::vector<int> destroy_concentration_id_;
  std::vector<int> destroy_step_id_;
  std::vector<int> destroy_sparse_id_;
  std::vector<int> create_concentration_id_;
  std::vector<int> create_step_id_;
  std::vector<int> create_sparse_id_;
  std::vector<int> jacobian_row_id_;
  std::vector<int> jacobian_column_id_;
  std::vector<int> jacobian_column_sum_;

  std::vector<double> molecular_mass_;
  std::vector<double> inv_molecular_mass_;

  // Jacobian calculation work arrays
  std::vector<double> inv_concentrations_;
  std::vector<double> original_state_;
  std::vector<double> perturbed_state_;
  std::vector<double> original_derivative_;
  std::vector<double> perturbed_derivative_;

  // non-integer reaction network
  int num_noninteger_jacobian_nonzeros_;
  std::vector<double> noninteger_jacobian_;
  std::vector<int> noninteger_sparse_id_;

};

// Implementation of the pure member functions from the ReactorBase class
CounterflowReactor::Impl::Impl(const char mechanism_name[],
                               const char thermodynamics_name[],
                               const char parser_log_name[],
                               const MatrixType matrix_type,
                               const double pressure,
                               const bool finite_separation)
{
  int jacobian_size;
  std::string info;
  std::string current_name;
  std::string species_prefix;
  std::vector<std::string> state_names;
  zerork::mechanism *mechanism_ptr;

  // set constructor arguments
  SetMatrixType(matrix_type);

  // set reference scales for state quanitites
  // TODO: scale relative volume
  use_scaled_state_ = false;
  inv_ref_temperature_ = ref_temperature_ = 1.0;
  SetPressure(pressure);
  finite_separation_ = finite_separation;

  BuildMechanism(mechanism_name,
                 thermodynamics_name,
                 parser_log_name);

  // create the vector of state names
  state_names.clear();
  mechanism_ptr = GetMechanism();
  species_prefix = std::string("MassFraction_");
  const int num_species = mechanism_ptr->getNumSpecies();
  const int num_steps   = mechanism_ptr->getNumSteps();

  for(int j=0; j<num_species; ++j) {
    current_name = std::string(mechanism_ptr->getSpeciesName(j));
    current_name.insert(0,species_prefix); // prepend prefix
    state_names.push_back(current_name);
  }
  state_names.push_back(std::string("RelativeVolume")); //or mass flux for steady
  state_names.push_back(std::string("Temperature"));
  state_names.push_back(std::string("Momentum"));
  if(finite_separation_)
    state_names.push_back(std::string("PStrain"));
  BuildStateNamesMap(state_names);
  const int num_states = static_cast<int>(state_names.size());

  // noninteger reaction network info needs to be set before entering
  // BuildSparseJacobianArrays()
  num_noninteger_jacobian_nonzeros_ =
    mechanism_ptr->getNonIntegerReactionNetwork()->GetNumJacobianNonzeros();

  noninteger_jacobian_.assign(num_noninteger_jacobian_nonzeros_,0.0);
  noninteger_sparse_id_.assign(num_noninteger_jacobian_nonzeros_,0);

  // set the jacobian size with the number of non-zero terms
  // and build the associated data structures for the sparse Jacobian
  // if used.
  if(matrix_type == DENSE_COL_MAJOR) {
    jacobian_size  = GetNumStates();
    jacobian_size *= jacobian_size;
  } else if(matrix_type == COMPRESSED_COL_STORAGE) {
    jacobian_size = BuildSparseJacobianArrays();
  } else {
    printf("ERROR: unrecognized matrix_type %d\n",
           static_cast<int>(matrix_type));
    fflush(stdout);
    exit(-1);
  }
  SetJacobianSize(jacobian_size);

  // provide basic info about the reactor model
  info  = "Counterflow reactor\n";
  info += "with number of species N = ";
  // TO DO: make the string formation cleaner
  char tmp[32];
  sprintf(tmp,"%d",num_species);
  info += std::string(tmp);
  info += std::string("\nstate[0]:   ") + state_names[0];
  info += std::string("\nstate[N-1]: ") + state_names[num_species-1];
  info += std::string("\nstate[N]:   ") + state_names[num_species];
  info += std::string("\nstate[N+1]: ") + state_names[num_species+1];
  info += std::string("\nstate[N+2]: ") + state_names[num_species+2];
  if(finite_separation_)
    info += std::string("\nstate[N+3]: ") + state_names[num_species+3];
  info += "\n-----------------------\n";
  SetReactorInfo(info);

  // pre-assign the size of the internal time-derivative vectors
  concentrations_.assign(num_species,0.0);
  enthalpies_.assign(num_species,0.0);
  specific_heats_.assign(num_species,0.0);
  net_reaction_rates_.assign(num_species,0.0);
  creation_rates_.assign(num_species,0.0);
  destruction_rates_.assign(num_species,0.0);
  step_rates_.assign(num_steps,0.0);
  // pre-assign the size of the internal jacobian space vectors
  inv_concentrations_.assign(num_species,0.0);
  original_state_.assign(num_states,0.0);
  perturbed_state_.assign(num_states,0.0);
  original_derivative_.assign(num_states,0.0);
  perturbed_derivative_.assign(num_states,0.0);

  // assign the molecular mass arrays
  molecular_mass_.assign(num_species,0.0);
  mechanism_ptr->getMolWtSpc(&molecular_mass_[0]);

  inv_molecular_mass_.assign(num_species,0.0);
  for(int j=0; j<num_species; ++j) {
    inv_molecular_mass_[j] = 1.0/molecular_mass_[j];
  }
}

CounterflowReactor::Impl::~Impl()
{
  DestroyMechanism();
}

ReactorError
  CounterflowReactor::Impl::GetTimeDerivative(const double reactor_time,
                                              const double state[],
                                              double derivative[])
{
  // define local constants of class members to enable loop vectorization
  const int num_species            = GetNumSpecies();
  const double ref_temperature     = ref_temperature_;
  const double inv_ref_temperature = inv_ref_temperature_;
  const double pressure            = pressure_;
  const double temperature         = ref_temperature*state[num_species+1];

  double mix_mass_cp, RuT;
  double mass_sum, enthalpy_sum;
  zerork::mechanism *mechanism_ptr;

  mechanism_ptr = GetMechanism();

  // compute the product of the universal gas constant and temperature
  RuT = temperature*mechanism_ptr->getGasConstant();

  // get relative volume and density
  const double relative_volume     = state[num_species];
  const double density             = 1.0/relative_volume;

  // compute concentration, total_moles and temperature from the current state
  for(int j=0; j<num_species; ++j) {
    concentrations_[j] = density*state[j]*inv_molecular_mass_[j];
  }
  // compute the rate of change of the species concentration
  mechanism_ptr->getReactionRates_perturbROP(temperature,
                                             &concentrations_[0],
                                             GetAMultipliers(),
                                             &net_reaction_rates_[0],
                                             &creation_rates_[0],
                                             &destruction_rates_[0],
                                             &step_rates_[0]);

  // compute the mass specific heat of the mixture
  mix_mass_cp = mechanism_ptr->getMassCpFromTY(temperature,
                                               state,
                                               &specific_heats_[0]);
  // compute the species enthalpies
  mechanism_ptr->getEnthalpy_RT(temperature,
                                &enthalpies_[0]);

  // compute the rate of change of the mass fraction of each species
  mass_sum = 0.0;
  enthalpy_sum = 0.0;
  for(int j=0; j<num_species; ++j) {
    // Natural formulas:
    //derivative[j] = relative_volume*net_reaction_rates_[j]*molecular_mass_[j];
    //mass_sum += derivative[j]*inv_molecular_mass_[j];
    //enthalpy_sum += derivative[j]*enthalpies_[j]*inv_molecular_mass_[j];

    // Multiplication saving formulas:
    derivative[j] = relative_volume*net_reaction_rates_[j];
    mass_sum += derivative[j];
    enthalpy_sum += derivative[j]*enthalpies_[j];
    derivative[j] *= molecular_mass_[j];
  }

  // compute the rate of temperature change
  // dT/dt = -1/Cp_mix * \sum_i (mass_enthalpy[i]*dy[i]/dt)
  //         current units [K/s]
  derivative[num_species+1] = -(RuT/mix_mass_cp)*enthalpy_sum;

  // compute the rate of change of the relative volume
  // dv/dt = v/T * dT/dt + RuT/p * \sum_i (1/mw[i] * dy[i]/dt)
  // dv/dt   current units [m^3/kg/s]
  derivative[num_species] = (relative_volume/temperature)*
    derivative[num_species+1] + (RuT/pressure)*mass_sum;

  // rescale derivatives
  //TODO: non-dimensionalize relative volume
  //  derivative[num_species]  = dv/dt *
  derivative[num_species+1] *= inv_ref_temperature;

  // Momentum derivative
  derivative[num_species+2] = 0.0;

  // P strain
  //derivative[num_species+3] = 0.0;

  return NONE; // no error
}

ReactorError
  CounterflowReactor::Impl::GetTimeDerivativeSteady(const double state[],
                                                    const double step_limiter[],
                                                    double derivative[])
{
  // define local constants of class members to enable loop vectorization
  const int num_species            = GetNumSpecies();
  const double ref_temperature     = ref_temperature_;
  const double inv_ref_temperature = inv_ref_temperature_;
  const double pressure            = pressure_;
  const double temperature         = ref_temperature*state[num_species+1];

  double mix_mass_cp, RuT;
  double mass_sum, enthalpy_sum;
  zerork::mechanism *mechanism_ptr;

  mechanism_ptr = GetMechanism();

  // compute the product of the universal gas constant and temperature
  RuT = temperature*mechanism_ptr->getGasConstant();

  // compute relative volume and density
  // for steady flames, state[num_species] contains mass flux
  double one_over_wmix = 0.0;
  for(int j=0; j<num_species; ++j) {
     one_over_wmix += state[j]*inv_molecular_mass_[j];
  }
  const double relative_volume = RuT*one_over_wmix/pressure;
  const double density         = 1.0/relative_volume;

  const double mass_flux = state[num_species];

  // compute concentration, total_moles and temperature from the current state
  for(int j=0; j<num_species; ++j) {
    concentrations_[j] = density*state[j]*inv_molecular_mass_[j];
  }

  // compute the rate of change of the species concentration
  mechanism_ptr->getReactionRatesLimiter_perturbROP(temperature,
                                                    &concentrations_[0],
                                                    &step_limiter[0],
                                                    GetAMultipliers(),
                                                    &net_reaction_rates_[0],
                                                    &creation_rates_[0],
                                                    &destruction_rates_[0],
                                                    &step_rates_[0]);

  // compute the mass specific heat of the mixture
  mix_mass_cp = mechanism_ptr->getMassCpFromTY(temperature,
                                               state,
                                               &specific_heats_[0]);
  // compute the species enthalpies
  mechanism_ptr->getEnthalpy_RT(temperature,
                                &enthalpies_[0]);

  // compute the rate of change of the mass fraction of each species
  mass_sum = 0.0;
  enthalpy_sum = 0.0;
  for(int j=0; j<num_species; ++j) {
    // Natural formulas:
    //derivative[j] = relative_volume*net_reaction_rates_[j]*molecular_mass_[j];
    //mass_sum += derivative[j]*inv_molecular_mass_[j];
    //enthalpy_sum += derivative[j]*enthalpies_[j]*inv_molecular_mass_[j];

    // Multiplication saving formulas:
    derivative[j] = relative_volume*net_reaction_rates_[j];
    mass_sum += derivative[j];
    enthalpy_sum += derivative[j]*enthalpies_[j];
    derivative[j] *= molecular_mass_[j];
  }

  // compute the rate of temperature change
  // dT/dt = -1/Cp_mix * \sum_i (mass_enthalpy[i]*dy[i]/dt)
  //         current units [K/s]
  derivative[num_species+1] = -(RuT/mix_mass_cp)*enthalpy_sum;

  // compute the rate of change of the mass flux
  derivative[num_species] = 0.0;

  // rescale derivatives
  //TODO: non-dimensionalize relative volume
  //  derivative[num_species]  = dv/dt *
  derivative[num_species+1] *= inv_ref_temperature;

  // Momentum derivative
  // source term computed here
  derivative[num_species+2] = 0.0;

  // PStrain
  //derivative[num_species+3] = 0.0;

  return NONE; // no error
}

ReactorError
  CounterflowReactor::Impl::GetTimeDerivativeLimiter(const double reactor_time,
                                                     const double state[],
                                                     const double step_limiter[],
                                                     double derivative[])
{
  // define local constants of class members to enable loop vectorization
  const int num_species            = GetNumSpecies();
  const double ref_temperature     = ref_temperature_;
  const double inv_ref_temperature = inv_ref_temperature_;
  const double pressure            = pressure_;
  const double temperature         = ref_temperature*state[num_species+1];

  double mix_mass_cp, RuT;
  double mass_sum, enthalpy_sum;
  zerork::mechanism *mechanism_ptr;

  mechanism_ptr = GetMechanism();

  // compute the product of the universal gas constant and temperature
  RuT = temperature*mechanism_ptr->getGasConstant();

  // get relative_volume and density
  const double relative_volume     = state[num_species];
  const double density             = 1.0/relative_volume;

  // compute concentration, total_moles and temperature from the current state
  for(int j=0; j<num_species; ++j) {
    concentrations_[j] = density*state[j]*inv_molecular_mass_[j];
  }

  // compute the rate of change of the species concentration
  mechanism_ptr->getReactionRatesLimiter(temperature,
					 &concentrations_[0],
					 &step_limiter[0],
					 &net_reaction_rates_[0],
					 &creation_rates_[0],
					 &destruction_rates_[0],
					 &step_rates_[0]);

  // compute the mass specific heat of the mixture
  mix_mass_cp = mechanism_ptr->getMassCpFromTY(temperature,
                                               state,
                                               &specific_heats_[0]);
  // compute the species enthalpies
  mechanism_ptr->getEnthalpy_RT(temperature,
                                &enthalpies_[0]);

  // compute the rate of change of the mass fraction of each species
  mass_sum = 0.0;
  enthalpy_sum = 0.0;
  for(int j=0; j<num_species; ++j) {
    // Natural formulas:
    //derivative[j] = relative_volume*net_reaction_rates_[j]*molecular_mass_[j];
    //mass_sum += derivative[j]*inv_molecular_mass_[j];
    //enthalpy_sum += derivative[j]*enthalpies_[j]*inv_molecular_mass_[j];

    // Multiplication saving formulas:
    derivative[j] = relative_volume*net_reaction_rates_[j];
    mass_sum += derivative[j];
    enthalpy_sum += derivative[j]*enthalpies_[j];
    derivative[j] *= molecular_mass_[j];
  }

  // compute the rate of temperature change
  // dT/dt = -1/Cp_mix * \sum_i (mass_enthalpy[i]*dy[i]/dt)
  //         current units [K/s]
  derivative[num_species+1] = -(RuT/mix_mass_cp)*enthalpy_sum;

  // compute the rate of change of the relative volume
  // dv/dt = v/T * dT/dt + RuT/p * \sum_i (1/mw[i] * dy[i]/dt)
  // dv/dt   current units [m^3/kg/s]
  derivative[num_species] = (relative_volume/temperature)*
    derivative[num_species+1] + (RuT/pressure)*mass_sum;

  // rescale derivatives
  //TODO: non-dimensionalize relative volume
  //  derivative[num_species]  = dv/dt *
  derivative[num_species+1] *= inv_ref_temperature;

  // Momentum derivative
  // source term computed here
  derivative[num_species+2] = 0.0;

  // P strain
  //derivative[num_species+3] = 0.0;

  return NONE; // no error
}

ReactorError
  CounterflowReactor::Impl::GetJacobian(const double reactor_time,
                                          const double state[],
                                          double jacobian[])
{

  if(GetMatrixType() == DENSE_COL_MAJOR) {
    printf("ERROR: GetJacobian not currently defined for\n");
    printf("       DENSE_COL_MAJOR.\n");
    return UNKNOWN;
  } else if(GetMatrixType() == COMPRESSED_COL_STORAGE) {
    return GetSparseJacobian(reactor_time, state, jacobian);
  }

  return UNKNOWN;
}

ReactorError
  CounterflowReactor::Impl::GetJacobianSteady(const double state[],
                                              const double convective[],
                                              const bool Tfix,
                                              const double ref_momentum,
                                              const double step_limiter[],
                                              double jacobian[])
{

  if(GetMatrixType() == DENSE_COL_MAJOR) {
    printf("ERROR: GetJacobian not currently defined for\n");
    printf("       DENSE_COL_MAJOR.\n");
    return UNKNOWN;
  } else if(GetMatrixType() == COMPRESSED_COL_STORAGE) {
    return GetSparseJacobianSteady(state, convective, Tfix, ref_momentum, step_limiter, jacobian);
  }

  return UNKNOWN;
}

ReactorError
  CounterflowReactor::Impl::GetJacobianLimiter(const double reactor_time,
                                               const double state[],
                                               const double step_limiter[],
                                               double jacobian[])
{

  if(GetMatrixType() == DENSE_COL_MAJOR) {
    printf("ERROR: GetJacobian not currently defined for\n");
    printf("       DENSE_COL_MAJOR.\n");
    return UNKNOWN;
  } else if(GetMatrixType() == COMPRESSED_COL_STORAGE) {
    return GetSparseJacobianLimiter(reactor_time, state, step_limiter, jacobian);
  }

  return UNKNOWN;
}

int CounterflowReactor::Impl::GetJacobianPattern(int row_id[],
                                                   int column_id[])
{
  const int num_states    = GetNumStates();
  const int jacobian_size = GetJacobianSize();
  if(row_id != NULL && column_id != NULL) {

    if(GetMatrixType() == DENSE_COL_MAJOR) {
      int dense_id = 0;
      for(int j=0; j<num_states; ++j) {
        // column - j
        for(int k=0; k<num_states; ++k) {
          // row - k
          row_id[dense_id]    = k;
          column_id[dense_id] = j;
          ++dense_id;
        }
      }

    } else if(GetMatrixType() == COMPRESSED_COL_STORAGE) {
      for(int j=0; j<jacobian_size; ++j) {
        row_id[j]    = jacobian_row_id_[j];
        column_id[j] = jacobian_column_id_[j];
      }
    }
  }

  return jacobian_size;
}

int CounterflowReactor::Impl::BuildSparseJacobianArrays()
{
  std::map<int, int> dense_to_sparse_map;
  std::map<int, int>::iterator map_iter;
  std::vector<int> noninteger_row_id;
  std::vector<int> noninteger_column_id;
  zerork::mechanism *mechanism_ptr;
  int row_id, column_id, dense_id, sparse_id;

  mechanism_ptr = GetMechanism();
  if(mechanism_ptr == NULL) {
    printf("ERROR: In BuildSparseJacobianArrays(),\n");
    printf("       mechanism is not allocated.\n");
    fflush(stdout);
    return -1;
  }
  const int num_species = mechanism_ptr->getNumSpecies();
  const int num_steps   = mechanism_ptr->getNumSteps();
  int num_states  = num_species + 3; // relative volume/mass flux, temperature, momentum
  if(finite_separation_)
    num_states  = num_species + 4; // relative volume/mass flux, temperature, momentum, pstrain

  // clear the jacobian arrays
  destroy_concentration_id_.clear();
  destroy_step_id_.clear();
  destroy_sparse_id_.clear();
  create_concentration_id_.clear();
  create_step_id_.clear();
  create_sparse_id_.clear();
  jacobian_row_id_.clear();
  jacobian_column_id_.clear();
  jacobian_column_sum_.clear();
  // temporary map to store distinct array points
  // also will count the total number of jacobian terms (elementary steps)
  // that w
  dense_to_sparse_map.clear();

  // loop over all steps
  for(int j=0; j<num_steps; ++j) {

    int num_reactants = mechanism_ptr->getOrderOfStep(j);
    int num_products  = mechanism_ptr->getNumProductsOfStep(j);

    for(int k=0; k<num_reactants; ++k) {
      // get the index of the species being perturbed for the Jacobian
      column_id = mechanism_ptr->getSpecIdxOfStepReactant(j,k);

      // set the Jacobian terms related to the destruction of a particular
      // species affected by the perturbation of species column_id
      for(int m=0; m<num_reactants; ++m) {

        row_id   = mechanism_ptr->getSpecIdxOfStepReactant(j,m);

        if(GetNetStoichiometry(row_id,j) != 0) {
          // Only add the Jacobian term if there is a net change in the
          // species, otherwise changes in the rate of progess of the step
          // won't change the net production rate of the species
          dense_id = row_id+column_id*num_states;

          destroy_concentration_id_.push_back(column_id);
          destroy_step_id_.push_back(j);
          destroy_sparse_id_.push_back(dense_id);

          // record position in dense matrix
          map_iter = dense_to_sparse_map.find(dense_id);
          if(map_iter == dense_to_sparse_map.end()) {
            // first term for dense_id
            dense_to_sparse_map[dense_id] = 1;
          } else {
            // increment number of terms for dense_id
            dense_to_sparse_map[dense_id] += 1;
          }
        }

      } // m-loop over reactants of step j

      // set the Jacobian terms related to the creation of a particular
      // species affected by the perturbation of species column_id
      for(int m=0; m<num_products; ++m) {
        row_id = mechanism_ptr->getSpecIdxOfStepProduct(j,m);

        if(GetNetStoichiometry(row_id,j) != 0) {
          // Only add the Jacobian term if there is a net change in the
          // species, otherwise changes in the rate of progess of the step
          // won't change the net production rate of the species
          dense_id = row_id+column_id*num_states;

          create_concentration_id_.push_back(column_id);
          create_step_id_.push_back(j);
          create_sparse_id_.push_back(dense_id);

          // record position in dense matrix
          map_iter = dense_to_sparse_map.find(dense_id);
          if(map_iter == dense_to_sparse_map.end()) {
            // first term for dense_id
            dense_to_sparse_map[dense_id] = 1;
          } else {
            // increment number of terms for dense_id
            dense_to_sparse_map[dense_id] += 1;
          }
        }
      } // m-loop over products of step j

    } // k-loop over reactants of step j

  } // j-loop over steps

  // non-integer reaction network
  if(num_noninteger_jacobian_nonzeros_ > 0) {

    noninteger_row_id.assign(num_noninteger_jacobian_nonzeros_, 0);
    noninteger_column_id.assign(num_noninteger_jacobian_nonzeros_, 0);

    mechanism_ptr->getNonIntegerReactionNetwork()->GetJacobianPattern(
       &noninteger_row_id[0],&noninteger_column_id[0]);

    for(int j=0; j<num_noninteger_jacobian_nonzeros_; ++j) {

      dense_id = noninteger_row_id[j]+noninteger_column_id[j]*num_states;

      // record position in dense matrix
      map_iter = dense_to_sparse_map.find(dense_id);
      if(map_iter == dense_to_sparse_map.end()) {
        // first term for dense_id
        dense_to_sparse_map[dense_id] = 1;
      } else {
	// increment number of terms for dense_id
	dense_to_sparse_map[dense_id] += 1;
      }

    }

  } // end if(num_noninteger_jacobian_nonzeros_ > 0)

  // add dense rows for the relative volume and temperature states
  // affected by perturbations in the species concentrations
  for(int j=0; j<num_species; ++j) {
    // add relative volume row (row_id = num_species)
    dense_id = num_species + j*num_states;
    dense_to_sparse_map[dense_id] = 1;
    // add temperature row (row_id = num_species+1)
    dense_id = num_species + 1 + j*num_states;
    dense_to_sparse_map[dense_id] = 1;
  }
  // add dense columns for all the states affected by perturbations in
  // the relative volume and the temperature
  for(int j=0; j<num_states; ++j) {
    // add relative volume column (column_id = num_species)
    dense_id = j + num_species*num_states;
    dense_to_sparse_map[dense_id] = 1;
    // add temperature column (column_id = num_species+1)
    dense_id = j + (num_species+1)*num_states;
    dense_to_sparse_map[dense_id] = 1;
  }

  // add some non-chemistry terms
  // d(rhs mdot)/dU
  dense_id = num_species + (num_species+2)*num_states;
  dense_to_sparse_map[dense_id] = 1;
  if(finite_separation_) {
    // d(rhs U)/dP
    dense_id = num_species+2 + (num_species+3)*num_states;
    dense_to_sparse_map[dense_id] = 1;
    // d(rhs P)/dU (only at last grid point)
    dense_id = num_species+3 + (num_species+2)*num_states;
    dense_to_sparse_map[dense_id] = 1;
  }

  // add the diagonal for all states
  for(int j=0; j<num_states; ++j) {
    dense_id = j+j*num_states;
    // record position in dense matrix
    map_iter = dense_to_sparse_map.find(dense_id);
    if(map_iter == dense_to_sparse_map.end()) {
      // first term for dense_id
      dense_to_sparse_map[dense_id] = 1;
    }
  }

  // record the non-zero addresses for the jacobian_column_id_ and
  // jacobian_row_id_, and update the dense_to_sparse_map storing the sparse_id
  // for all the existing non-zero dense elements
  dense_id = 0;
  sparse_id = 0;
  for(int j=0; j<num_states; ++j) {
    // column-j
    jacobian_column_sum_.push_back(sparse_id);
    for(int k=0; k<num_states; ++k) {
      // row-k
      map_iter = dense_to_sparse_map.find(dense_id);
      if(map_iter != dense_to_sparse_map.end()) {
        // found non-zero element recorded in dense_to_sparse_map
        dense_to_sparse_map[dense_id] = sparse_id;
        jacobian_row_id_.push_back(k);
        jacobian_column_id_.push_back(j);
        ++sparse_id;
      }
      ++dense_id;
    }
  }
  jacobian_column_sum_.push_back(sparse_id);

  // use the dense to sparse map to transform the dense_id originally stored
  // in destroy_sparse_id_ and create_sparse_id_.
  const int num_destroy = static_cast<int>(destroy_sparse_id_.size());
  for(int j=0; j<num_destroy; ++j) {
    dense_id = destroy_sparse_id_[j];
    destroy_sparse_id_[j] = dense_to_sparse_map[dense_id];
  }
  const int num_create  = static_cast<int>(create_sparse_id_.size());
  for(int j=0; j<num_create; ++j) {
    dense_id = create_sparse_id_[j];
    create_sparse_id_[j] = dense_to_sparse_map[dense_id];
  }

  for(int j=0; j<num_noninteger_jacobian_nonzeros_; ++j) {
    dense_id = noninteger_row_id[j]+noninteger_column_id[j]*num_states;
    noninteger_sparse_id_[j] = dense_to_sparse_map[dense_id];
  }

  return static_cast<int>(dense_to_sparse_map.size()); // number of non-zeros

}


ReactorError
  CounterflowReactor::Impl::GetSparseJacobian(const double reactor_time,
                                              const double state[],
                                              double jacobian[])
{
  const double perturb_factor = 1.0e-8;
  // define local constants of class members to enable loop vectorization
  const int num_species            = GetNumSpecies();
  const int num_states             = GetNumStates();
  const int num_nonzeros           = GetJacobianSize();
  const double ref_temperature     = ref_temperature_;
  const double inv_ref_temperature = inv_ref_temperature_;
  const double pressure            = pressure_;
  const double temperature         = ref_temperature*state[num_species+1];
  const int num_destroy  = static_cast<int>(destroy_sparse_id_.size());
  const int num_create   = static_cast<int>(create_sparse_id_.size());

  double mix_mass_cp, RuT;
  double mass_sum, enthalpy_sum;
  double d_temperature, d_relative_volume;
  double min_concentration;
  zerork::mechanism *mechanism_ptr;

  mechanism_ptr = GetMechanism();

  // compute the product of the universal gas constant and temperature
  RuT = temperature*mechanism_ptr->getGasConstant();

  // get relative_volume and density
  const double relative_volume     = state[num_species];
  const double density             = 1.0/relative_volume;

  // initialize the jacobian array to zero
  for(int j=0; j<num_nonzeros; ++j) {
    jacobian[j] = 0.0;
  }

  // Copy the original state and correct any mass_fractions that get adjusted
  // by the minimum concentration. Note that one molecule in a 1 m^3 volume is
  // taken to be the minimum concentration
  for(int j=0; j<num_states; ++j) {
    original_state_[j] = state[j];
  }
  min_concentration = 1.0/mechanism_ptr->getAvogadroNumber();
  for(int j=0; j<num_species; ++j) {
    concentrations_[j] = density*state[j]*inv_molecular_mass_[j];
    if(fabs(concentrations_[j]) < min_concentration) {
      // preserve the sign of the concentration when pushing to the minimum
      // magnitude
      if(concentrations_[j] >= 0.0) {
        concentrations_[j] = min_concentration;
      } else {
        concentrations_[j] = -min_concentration;
      }
      // recompute original state from new concentrations
      original_state_[j] =
        relative_volume*concentrations_[j]*molecular_mass_[j];
    }
    inv_concentrations_[j] = 1.0/concentrations_[j];
  }

  // compute the rate of change of the species concentration
  mechanism_ptr->getReactionRates_perturbROP(temperature,
                                             &concentrations_[0],
                                             GetAMultipliers(),
                                             &net_reaction_rates_[0],
                                             &creation_rates_[0],
                                             &destruction_rates_[0],
                                             &step_rates_[0]);

  // use the step rates and inverse concentrations with the elementary
  // Jacobian term lists to compute dwdot[i]/dC[j]
  // Destruction terms
  for(int j=0; j<num_destroy; ++j) {
    int conc_id   = destroy_concentration_id_[j];
    int step_id   = destroy_step_id_[j];
    int sparse_id = destroy_sparse_id_[j];
    // [START DEBUG]
    //int print_row_id = 3;
    //int print_col_id = 1;
    //if(jacobian_row_id_[sparse_id] == print_row_id &&
    //   jacobian_column_id_[sparse_id] == print_col_id) {
    //  printf("Destroy term[%d,%d](sparse %d) = -%24.18e\n",
    //         print_row_id, print_col_id,sparse_id,
    //         step_rates_[step_id]*inv_concentrations_[conc_id]);
    //}
    // [END DEBUG]
    jacobian[sparse_id] -= step_rates_[step_id]*inv_concentrations_[conc_id];
  }
  // Creation terms
  for(int j=0; j<num_create; ++j) {
    int conc_id   = create_concentration_id_[j];
    int step_id   = create_step_id_[j];
    int sparse_id = create_sparse_id_[j];
    // [START DEBUG]
    //int print_row_id = 3;
    //int print_col_id = 1;
    //if(jacobian_row_id_[sparse_id] == print_row_id &&
    //   jacobian_column_id_[sparse_id] == print_col_id) {
    //  printf("Create  term[%d,%d](sparse %d) = +%24.18e\n",
    //         print_row_id, print_col_id,sparse_id,
    //         step_rates_[step_id]*inv_concentrations_[conc_id]);
    //}
    // [END DEBUG]
    jacobian[sparse_id] += step_rates_[step_id]*inv_concentrations_[conc_id];
  }

  // process the non-integer Jacobian information
  const int num_noninteger_jacobian_nonzeros =
    num_noninteger_jacobian_nonzeros_;

  for(int j=0; j<num_noninteger_jacobian_nonzeros; ++j) {
    noninteger_jacobian_[j] = 0.0;
  }

  mechanism_ptr->getNonIntegerReactionNetwork()->GetSpeciesJacobian(
    &inv_concentrations_[0],
    &step_rates_[0],
    &noninteger_jacobian_[0]);

  for(int j=0; j<num_noninteger_jacobian_nonzeros; ++j) {
    jacobian[noninteger_sparse_id_[j]] += noninteger_jacobian_[j];
  }

  // Use the molecular mass arrays to convert the composition Jacobian
  // from concentration to mass fraction.
  //
  // TODO: make the composition Jacobian operations more reusable.
  //       Currently, the operations in this function assumes compressed
  //       column storage with dense rows and columns for the relative
  //       volume and temperature.
  for(int j=0; j<num_species; ++j) {
    // j^th column
    int num_nonzero_rows = jacobian_column_sum_[j+1] - jacobian_column_sum_[j];
    int sparse_id = jacobian_column_sum_[j];

    for(int k=0; k<(num_nonzero_rows-2); ++k) { //-2 or -4??

      int row_id = jacobian_row_id_[sparse_id];
      jacobian[sparse_id] *= molecular_mass_[row_id]*inv_molecular_mass_[j];
      ++sparse_id;
    }
  }

  mechanism_ptr->getCp_R_Enthalpy_RT(temperature,
                                     &specific_heats_[0],
                                     &enthalpies_[0]);
  mix_mass_cp = 0.0;
  mass_sum = 0.0;
  enthalpy_sum = 0.0;
  for(int j=0; j<num_species; ++j) {
    mix_mass_cp +=
      original_state_[j]*inv_molecular_mass_[j]*specific_heats_[j];
    // mix_mass_cp is non-dimensionalized by Ru
    original_derivative_[j] =
      relative_volume*net_reaction_rates_[j]*molecular_mass_[j];

    mass_sum     += original_derivative_[j]*inv_molecular_mass_[j];
    enthalpy_sum +=
      original_derivative_[j]*enthalpies_[j]*inv_molecular_mass_[j];
    // enthalpy sum is non-dimensionalized bu RuT
  }

  // compute the rate of temperature change
  // dT/dt = -1/Cp_mix * \sum_i (mass_enthalpy[i]*dy[i]/dt)
  //         current units [K/s]
  original_derivative_[num_species+1] =
    -(temperature/mix_mass_cp)*enthalpy_sum;

  // compute the rate of change of the relative volume
  // dv/dt = v/T * dT/dt + RuT/p * \sum_i (1/mw[i] * dy[i]/dt)
  // dv/dt   current units [m^3/kg/s]
  original_derivative_[num_species] = (relative_volume/temperature)*
    original_derivative_[num_species+1] + (RuT/pressure)*mass_sum;

  // rescale the derivatives
  // TODO: non-dimensionalize relative volume
  //  derivative[num_species]  = dv/dt *
  original_derivative_[num_species+1] *= inv_ref_temperature;

  // compute the rate of momentum change
  original_derivative_[num_species+2] = 0.0;

  // P strain
  //original_derivative_[num_species+3] = 0.0;

  // compute the temperature derivative row
  for(int j=0; j<num_species; ++j) {
    // j^th column
    int num_nonzero_rows = jacobian_column_sum_[j+1] - jacobian_column_sum_[j];
    int sparse_id = jacobian_column_sum_[j];
    double jacobian_sum = 0.0;

    for(int k=0; k<(num_nonzero_rows-2); ++k) {

      int row_id = jacobian_row_id_[sparse_id];
      jacobian_sum +=
        enthalpies_[row_id]*jacobian[sparse_id]*inv_molecular_mass_[row_id];
      ++sparse_id;
    }
    // because of the dense row assumption (sparse_id) currently points at the
    // relative_volume address for column j, and (sparse_id+1) currently
    // points at the temperature address for column j
    jacobian[sparse_id+1] = -jacobian_sum*temperature*inv_ref_temperature
                            -original_derivative_[num_species+1]*
                            specific_heats_[j]*inv_molecular_mass_[j];
    jacobian[sparse_id+1] /= mix_mass_cp;
  }

  // compute the relative volume derivative row, note that it depends on
  // the temperature derivative row which is set above
  for(int j=0; j<num_species; ++j) {
    // j^th column
    int num_nonzero_rows = jacobian_column_sum_[j+1] - jacobian_column_sum_[j];
    int sparse_id = jacobian_column_sum_[j];
    double jacobian_sum = 0.0;

    for(int k=0; k<(num_nonzero_rows-2); ++k) { //-2 or -4

      int row_id = jacobian_row_id_[sparse_id];
      jacobian_sum += inv_molecular_mass_[row_id]*jacobian[sparse_id];
      ++sparse_id;
    }
    // because of the dense row assumption (sparse_id) currently points at the
    // relative_volume address for column j, and (sparse_id+1) currently
    // points at the temperature address for column j
    jacobian[sparse_id] = (RuT/pressure)*jacobian_sum +
      relative_volume*(ref_temperature/temperature)*jacobian[sparse_id+1];
  }
  // [DEBUG]
  //for(int j=0; j<(num_species+2); ++j) {
  //  printf("y[%d]: %24.17e  dy/dt = %24.17e (%s)\n",
  //         j+1,
  //         original_state_[j],
  //         original_derivative_[j],
  //         GetNameOfStateId(j));
  //}

  // perturb the relative volume
  d_relative_volume = original_state_[num_species]*perturb_factor;
  GetJacobianColumnFromPerturbation(num_species,
                                    d_relative_volume,
                                    reactor_time,
                                    &original_state_[0],
                                    &original_derivative_[0],
                                    &perturbed_state_[0],
                                    &perturbed_derivative_[0],
                                    &jacobian[jacobian_column_sum_[num_species]]);
  // perturb the temperature
  d_temperature = original_state_[num_species+1]*perturb_factor;
  GetJacobianColumnFromPerturbation(num_species+1,
                                    d_temperature,
                                    reactor_time,
                                    &original_state_[0],
                                    &original_derivative_[0],
                                    &perturbed_state_[0],
                                    &perturbed_derivative_[0],
                                    &jacobian[jacobian_column_sum_[num_species+1]]);


  // Momentum
  jacobian[jacobian_column_sum_[num_species+2]] = 0.0;

  //jacobian[jacobian_column_sum_[num_species+3]] = 0.0;


 return NONE;
}


ReactorError
  CounterflowReactor::Impl::GetSparseJacobianSteady(const double state[],
                                                    const double convective[],
                                                    const bool Tfix,
                                                    const double ref_momentum,
                                                    const double step_limiter[],
                                                    double jacobian[])
{
  const double perturb_factor = 1.0e-8;
  // define local constants of class members to enable loop vectorization
  const int num_species            = GetNumSpecies();
  const int num_states             = GetNumStates();
  const int num_nonzeros           = GetJacobianSize();
  const double ref_temperature     = ref_temperature_;
  const double inv_ref_temperature = inv_ref_temperature_;
  const double pressure            = pressure_;
  const double temperature         = ref_temperature*state[num_species+1];
  const int num_destroy  = static_cast<int>(destroy_sparse_id_.size());
  const int num_create   = static_cast<int>(create_sparse_id_.size());

  double mix_mass_cp, RuT;
  double mass_sum, enthalpy_sum;
  double d_temperature;
  double min_concentration;
  zerork::mechanism *mechanism_ptr;

  mechanism_ptr = GetMechanism();

  // compute the product of the universal gas constant and temperature
  RuT = temperature*mechanism_ptr->getGasConstant();

  // compute relative_volume and density
  // state[num_spcies] is the mass_flux for steady premixed flames
  double one_over_wmix = 0.0;
  for(int j=0; j<num_species; ++j) {
     one_over_wmix += state[j]*inv_molecular_mass_[j];
  }
  const double relative_volume = RuT*one_over_wmix/pressure;
  const double density         = 1.0/relative_volume;

  // initialize the jacobian array to zero
  for(int j=0; j<num_nonzeros; ++j) {
    jacobian[j] = 0.0;
  }

  // Copy the original state and correct any mass_fractions that get adjusted
  // by the minimum concentration. Note that one molecule in a 1 m^3 volume is
  // taken to be the minimum concentration
  for(int j=0; j<num_states; ++j) {
    original_state_[j] = state[j];
  }
  min_concentration = 1.0/mechanism_ptr->getAvogadroNumber();
  for(int j=0; j<num_species; ++j) {
    concentrations_[j] = density*state[j]*inv_molecular_mass_[j];
    if(fabs(concentrations_[j]) < min_concentration) {
      // preserve the sign of the concentration when pushing to the minimum
      // magnitude
      if(concentrations_[j] >= 0.0) {
        concentrations_[j] = min_concentration;
      } else {
        concentrations_[j] = -min_concentration;
      }
      // recompute original state from new concentrations
      original_state_[j] =
        relative_volume*concentrations_[j]*molecular_mass_[j];
    }
    inv_concentrations_[j] = 1.0/concentrations_[j];
  }

  // compute the rate of change of the species concentration
  mechanism_ptr->getReactionRatesLimiter_perturbROP(temperature,
                                                    &concentrations_[0],
                                                    &step_limiter[0],
                                                    GetAMultipliers(),
                                                    &net_reaction_rates_[0],
                                                    &creation_rates_[0],
                                                    &destruction_rates_[0],
                                                    &step_rates_[0]);


  // use the step rates and inverse concentrations with the elementary
  // Jacobian term lists to compute dwdot[i]/dC[j]
  // Destruction terms
  for(int j=0; j<num_destroy; ++j) {
    int conc_id   = destroy_concentration_id_[j];
    int step_id   = destroy_step_id_[j];
    int sparse_id = destroy_sparse_id_[j];
    // [START DEBUG]
    //int print_row_id = 3;
    //int print_col_id = 1;
    //if(jacobian_row_id_[sparse_id] == print_row_id &&
    //   jacobian_column_id_[sparse_id] == print_col_id) {
    //  printf("Destroy term[%d,%d](sparse %d) = -%24.18e\n",
    //         print_row_id, print_col_id,sparse_id,
    //         step_rates_[step_id]*inv_concentrations_[conc_id]);
    //}
    // [END DEBUG]
    jacobian[sparse_id] -= step_rates_[step_id]*inv_concentrations_[conc_id];
  }
  // Creation terms
  for(int j=0; j<num_create; ++j) {
    int conc_id   = create_concentration_id_[j];
    int step_id   = create_step_id_[j];
    int sparse_id = create_sparse_id_[j];
    // [START DEBUG]
    //int print_row_id = 3;
    //int print_col_id = 1;
    //if(jacobian_row_id_[sparse_id] == print_row_id &&
    //   jacobian_column_id_[sparse_id] == print_col_id) {
    //  printf("Create  term[%d,%d](sparse %d) = +%24.18e\n",
    //         print_row_id, print_col_id,sparse_id,
    //         step_rates_[step_id]*inv_concentrations_[conc_id]);
    //}
    // [END DEBUG]

    jacobian[sparse_id] += step_rates_[step_id]*inv_concentrations_[conc_id];
  }

  // process the non-integer Jacobian information
  const int num_noninteger_jacobian_nonzeros =
    num_noninteger_jacobian_nonzeros_;

  for(int j=0; j<num_noninteger_jacobian_nonzeros; ++j) {
    noninteger_jacobian_[j] = 0.0;
  }

  mechanism_ptr->getNonIntegerReactionNetwork()->GetSpeciesJacobian(
    &inv_concentrations_[0],
    &step_rates_[0],
    &noninteger_jacobian_[0]);

  for(int j=0; j<num_noninteger_jacobian_nonzeros; ++j) {
    jacobian[noninteger_sparse_id_[j]] += noninteger_jacobian_[j];
  }

  // Use the molecular mass arrays to convert the composition Jacobian
  // from concentration to mass fraction.
  //
  // TODO: make the composition Jacobian operations more reusable.
  //       Currently, the operations in this function assumes compressed
  //       column storage with dense rows and columns for the relative
  //       volume and temperature.
  for(int j=0; j<num_species; ++j) {
    // j^th column
    int num_nonzero_rows = jacobian_column_sum_[j+1] - jacobian_column_sum_[j];
    int sparse_id = jacobian_column_sum_[j];

    for(int k=0; k<(num_nonzero_rows-2); ++k) {

      int row_id = jacobian_row_id_[sparse_id];
      jacobian[sparse_id] *= molecular_mass_[row_id]*inv_molecular_mass_[j];
      ++sparse_id;
    }
  }

  mechanism_ptr->getCp_R_Enthalpy_RT(temperature,
                                     &specific_heats_[0],
                                     &enthalpies_[0]);
  mix_mass_cp = 0.0;
  mass_sum = 0.0;
  enthalpy_sum = 0.0;
  for(int j=0; j<num_species; ++j) {
    mix_mass_cp +=
      original_state_[j]*inv_molecular_mass_[j]*specific_heats_[j];
    // mix_mass_cp is non-dimensionalized by Ru
    original_derivative_[j] =
      relative_volume*net_reaction_rates_[j]*molecular_mass_[j];

    mass_sum     += original_derivative_[j]*inv_molecular_mass_[j];
    enthalpy_sum +=
      original_derivative_[j]*enthalpies_[j]*inv_molecular_mass_[j];
    // enthalpy sum is non-dimensionalized bu RuT
  }
  // compute the rate of temperature change
  // dT/dt = -1/Cp_mix * \sum_i (mass_enthalpy[i]*dy[i]/dt)
  //         current units [K/s]
  original_derivative_[num_species+1] =
    -(temperature/mix_mass_cp)*enthalpy_sum;

  // compute the rate of change of the relative volume/mass flux
  original_derivative_[num_species] = 0.0;

  // rescale the derivatives
  original_derivative_[num_species+1] *= inv_ref_temperature;

  // compute the rate of momentum change
  // TODO: non-dimensionalize
  original_derivative_[num_species+2] = 0.0;

  // Pstrain
  //original_derivative_[num_species+3] = 0.0;

  // compute the temperature derivative row
  for(int j=0; j<num_species; ++j) {
    // j^th column
    int num_nonzero_rows = jacobian_column_sum_[j+1] - jacobian_column_sum_[j];
    int sparse_id = jacobian_column_sum_[j];
    double jacobian_sum = 0.0;

    for(int k=0; k<(num_nonzero_rows-2); ++k) {

      int row_id = jacobian_row_id_[sparse_id];
      jacobian_sum +=
	enthalpies_[row_id]*jacobian[sparse_id]*inv_molecular_mass_[row_id];
      ++sparse_id;
    }
    // because of the dense row assumption (sparse_id) currently points at the
    // mass_flux address for column j, and (sparse_id+1) currently
    // points at the temperature address for column j
    jacobian[sparse_id+1] = -jacobian_sum*temperature*inv_ref_temperature
                            -original_derivative_[num_species+1]*
                            specific_heats_[j]*inv_molecular_mass_[j];
    jacobian[sparse_id+1] /= mix_mass_cp;

    /*
    // Tfix is a  flag set to True for one point in the domain where temperature is fixed
    // to anchor the flame. At the point, the temperature residual is swapped with the mass_flux residual
    if(!Tfix) { // Put in temperature row
      jacobian[sparse_id+1] = -jacobian_sum*temperature*inv_ref_temperature
	-original_derivative_[num_species+1]*
	specific_heats_[j]*inv_molecular_mass_[j];
      jacobian[sparse_id+1] /= mix_mass_cp;
    } else { // Put in mass flux row
      jacobian[sparse_id] = -jacobian_sum*temperature*inv_ref_temperature
	-original_derivative_[num_species+1]*
	specific_heats_[j]*inv_molecular_mass_[j];
      jacobian[sparse_id] /= mix_mass_cp;
    }
    */

  }

  // Compute the mass flux column
  // convective terms of other state variables
  for(int j=0; j<num_states; ++j) {
    jacobian[jacobian_column_sum_[num_species]+j] = convective[j];
  }

  // perturb the temperature
  d_temperature = original_state_[num_species+1]*perturb_factor;
  GetJacobianColumnFromPerturbationSteady(num_species+1,
					  d_temperature,
					  &original_state_[0],
					  &original_derivative_[0],
                                          &step_limiter[0],
					  &perturbed_state_[0],
					  &perturbed_derivative_[0],
					  &jacobian[jacobian_column_sum_[num_species+1]]);

  // Momentum column
  //jacobian[jacobian_column_sum_[num_species+2]] = 0.0;
  /**/
  // d(rhs mdot)/dU
  jacobian[jacobian_column_sum_[num_species+2]] = -ref_momentum/relative_volume;
  // d(rhs U)/dU
  jacobian[jacobian_column_sum_[num_species+2]+1] = 0.0;
  // d(rhs P)/dU only at last grid point
  if(Tfix)
    jacobian[jacobian_column_sum_[num_species+2]+2] = -ref_momentum/relative_volume;

  if(finite_separation_) {
    // PStrain column
    // d(rhs U)/dP
    jacobian[jacobian_column_sum_[num_species+3]] = -relative_volume;
    // d(rhs P)/dP
    //jacobian[jacobian_column_sum_[num_species+3]+1] = 0.0;
  }
  /**/

 return NONE;
}

ReactorError
CounterflowReactor::Impl::GetSparseJacobianLimiter(const double reactor_time,
                                                   const double state[],
                                                   const double step_limiter[],
                                                   double jacobian[])
{
  const double perturb_factor = 1.0e-8;
  // define local constants of class members to enable loop vectorization
  const int num_species            = GetNumSpecies();
  const int num_states             = GetNumStates();
  const int num_nonzeros           = GetJacobianSize();
  const double ref_temperature     = ref_temperature_;
  const double inv_ref_temperature = inv_ref_temperature_;
  const double pressure            = pressure_;
  const double temperature         = ref_temperature*state[num_species+1];
  const int num_destroy  = static_cast<int>(destroy_sparse_id_.size());
  const int num_create   = static_cast<int>(create_sparse_id_.size());

  double mix_mass_cp, RuT;
  double mass_sum, enthalpy_sum;
  double d_temperature, d_relative_volume;
  double min_concentration;
  zerork::mechanism *mechanism_ptr;

  mechanism_ptr = GetMechanism();

  // compute the product of the universal gas constant and temperature
  RuT = temperature*mechanism_ptr->getGasConstant();

  // get relative_volume and density
  const double relative_volume     = state[num_species];
  const double density             = 1.0/relative_volume;

  // initialize the jacobian array to zero
  for(int j=0; j<num_nonzeros; ++j) {
    jacobian[j] = 0.0;
  }

  // Copy the original state and correct any mass_fractions that get adjusted
  // by the minimum concentration. Note that one molecule in a 1 m^3 volume is
  // taken to be the minimum concentration
  for(int j=0; j<num_states; ++j) {
    original_state_[j] = state[j];
  }
  min_concentration = 1.0/mechanism_ptr->getAvogadroNumber();
  for(int j=0; j<num_species; ++j) {
    concentrations_[j] = density*state[j]*inv_molecular_mass_[j];
    if(fabs(concentrations_[j]) < min_concentration) {
      // preserve the sign of the concentration when pushing to the minimum
      // magnitude
      if(concentrations_[j] >= 0.0) {
        concentrations_[j] = min_concentration;
      } else {
        concentrations_[j] = -min_concentration;
      }
      // recompute original state from new concentrations
      original_state_[j] =
        relative_volume*concentrations_[j]*molecular_mass_[j];
    }
    inv_concentrations_[j] = 1.0/concentrations_[j];
  }

  // compute the rate of change of the species concentration
  mechanism_ptr->getReactionRatesLimiter_perturbROP(temperature,
                                                    &concentrations_[0],
                                                    &step_limiter[0],
                                                    GetAMultipliers(),
                                                    &net_reaction_rates_[0],
                                                    &creation_rates_[0],
                                                    &destruction_rates_[0],
                                                    &step_rates_[0]);

  // use the step rates and inverse concentrations with the elementary
  // Jacobian term lists to compute dwdot[i]/dC[j]
  // Destruction terms
  for(int j=0; j<num_destroy; ++j) {
    int conc_id   = destroy_concentration_id_[j];
    int step_id   = destroy_step_id_[j];
    int sparse_id = destroy_sparse_id_[j];
    // [START DEBUG]
    //int print_row_id = 3;
    //int print_col_id = 1;
    //if(jacobian_row_id_[sparse_id] == print_row_id &&
    //   jacobian_column_id_[sparse_id] == print_col_id) {
    //  printf("Destroy term[%d,%d](sparse %d) = -%24.18e\n",
    //         print_row_id, print_col_id,sparse_id,
    //         step_rates_[step_id]*inv_concentrations_[conc_id]);
    //}
    // [END DEBUG]
    jacobian[sparse_id] -= step_rates_[step_id]*inv_concentrations_[conc_id];
  }
  // Creation terms
  for(int j=0; j<num_create; ++j) {
    int conc_id   = create_concentration_id_[j];
    int step_id   = create_step_id_[j];
    int sparse_id = create_sparse_id_[j];
    // [START DEBUG]
    //int print_row_id = 3;
    //int print_col_id = 1;
    //if(jacobian_row_id_[sparse_id] == print_row_id &&
    //   jacobian_column_id_[sparse_id] == print_col_id) {
    //  printf("Create  term[%d,%d](sparse %d) = +%24.18e\n",
    //         print_row_id, print_col_id,sparse_id,
    //         step_rates_[step_id]*inv_concentrations_[conc_id]);
    //}
    // [END DEBUG]

    jacobian[sparse_id] += step_rates_[step_id]*inv_concentrations_[conc_id];
  }
  // process the non-integer Jacobian information
  const int num_noninteger_jacobian_nonzeros =
    num_noninteger_jacobian_nonzeros_;

  for(int j=0; j<num_noninteger_jacobian_nonzeros; ++j) {
    noninteger_jacobian_[j] = 0.0;
  }

  mechanism_ptr->getNonIntegerReactionNetwork()->GetSpeciesJacobian(
    &inv_concentrations_[0],
    &step_rates_[0],
    &noninteger_jacobian_[0]);

  for(int j=0; j<num_noninteger_jacobian_nonzeros; ++j) {
    jacobian[noninteger_sparse_id_[j]] += noninteger_jacobian_[j];
  }

  // Use the molecular mass arrays to convert the composition Jacobian
  // from concentration to mass fraction.
  //
  // TODO: make the composition Jacobian operations more reusable.
  //       Currently, the operations in this function assumes compressed
  //       column storage with dense rows and columns for the relative
  //       volume and temperature.
  for(int j=0; j<num_species; ++j) {
    // j^th column
    int num_nonzero_rows = jacobian_column_sum_[j+1] - jacobian_column_sum_[j];
    int sparse_id = jacobian_column_sum_[j];

    for(int k=0; k<(num_nonzero_rows-2); ++k) { //-2 or -4

      int row_id = jacobian_row_id_[sparse_id];
      jacobian[sparse_id] *= molecular_mass_[row_id]*inv_molecular_mass_[j];
      ++sparse_id;
    }
  }

  mechanism_ptr->getCp_R_Enthalpy_RT(temperature,
                                     &specific_heats_[0],
                                     &enthalpies_[0]);
  mix_mass_cp = 0.0;
  mass_sum = 0.0;
  enthalpy_sum = 0.0;
  for(int j=0; j<num_species; ++j) {
    mix_mass_cp +=
      original_state_[j]*inv_molecular_mass_[j]*specific_heats_[j];
    // mix_mass_cp is non-dimensionalized by Ru
    original_derivative_[j] =
      relative_volume*net_reaction_rates_[j]*molecular_mass_[j];

    mass_sum     += original_derivative_[j]*inv_molecular_mass_[j];
    enthalpy_sum +=
      original_derivative_[j]*enthalpies_[j]*inv_molecular_mass_[j];
    // enthalpy sum is non-dimensionalized bu RuT
  }
  // compute the rate of temperature change
  // dT/dt = -1/Cp_mix * \sum_i (mass_enthalpy[i]*dy[i]/dt)
  //         current units [K/s]
  original_derivative_[num_species+1] =
    -(temperature/mix_mass_cp)*enthalpy_sum;

  // compute the rate of change of the relative volume
  // dv/dt = v/T * dT/dt + RuT/p * \sum_i (1/mw[i] * dy[i]/dt)
  // dv/dt   current units [m^3/kg/s]
  original_derivative_[num_species] = (relative_volume/temperature)*
    original_derivative_[num_species+1] + (RuT/pressure)*mass_sum;

  // rescale the derivatives
  // TODO: non-dimensionalize relative volume
  //  derivative[num_species]  = dv/dt *
  original_derivative_[num_species+1] *= inv_ref_temperature;

  // compute the rate of momentum change
  // TODO: non-dimensionalize
  original_derivative_[num_species+2] = 0.0;

  // PStrain
  //original_derivative_[num_species+3] = 0.0;


  // compute the temperature derivative row
  for(int j=0; j<num_species; ++j) {
    // j^th column
    int num_nonzero_rows = jacobian_column_sum_[j+1] - jacobian_column_sum_[j];
    int sparse_id = jacobian_column_sum_[j];
    double jacobian_sum = 0.0;

    for(int k=0; k<(num_nonzero_rows-2); ++k) { // -2 or -4

      int row_id = jacobian_row_id_[sparse_id];
      jacobian_sum +=
        enthalpies_[row_id]*jacobian[sparse_id]*inv_molecular_mass_[row_id];
      ++sparse_id;
    }
    // because of the dense row assumption (sparse_id) currently points at the
    // relative_volume address for column j, and (sparse_id+1) currently
    // points at the temperature address for column j
    jacobian[sparse_id+1] = -jacobian_sum*temperature*inv_ref_temperature
                            -original_derivative_[num_species+1]*
                            specific_heats_[j]*inv_molecular_mass_[j];
    jacobian[sparse_id+1] /= mix_mass_cp;
  }

  // compute the relative volume derivative row, note that it depends on
  // the temperature derivative row which is set above
  for(int j=0; j<num_species; ++j) {
    // j^th column
    int num_nonzero_rows = jacobian_column_sum_[j+1] - jacobian_column_sum_[j];
    int sparse_id = jacobian_column_sum_[j];
    double jacobian_sum = 0.0;

    for(int k=0; k<(num_nonzero_rows-2); ++k) {

      int row_id = jacobian_row_id_[sparse_id];
      jacobian_sum += inv_molecular_mass_[row_id]*jacobian[sparse_id];
      ++sparse_id;
    }
    // because of the dense row assumption (sparse_id) currently points at the
    // relative_volume address for column j, and (sparse_id+1) currently
    // points at the temperature address for column j
    jacobian[sparse_id] = (RuT/pressure)*jacobian_sum +
      relative_volume*(ref_temperature/temperature)*jacobian[sparse_id+1];
  }
  // [DEBUG]
  //for(int j=0; j<(num_species+2); ++j) {
  //  printf("y[%d]: %24.17e  dy/dt = %24.17e (%s)\n",
  //         j+1,
  //         original_state_[j],
  //         original_derivative_[j],
  //         GetNameOfStateId(j));
  //}

  // perturb the relative volume
  d_relative_volume = original_state_[num_species]*perturb_factor;
  GetJacobianColumnFromPerturbationLimiter(num_species,
					   d_relative_volume,
					   reactor_time,
					   &original_state_[0],
					   &original_derivative_[0],
					   &step_limiter[0],
					   &perturbed_state_[0],
					   &perturbed_derivative_[0],
					   &jacobian[jacobian_column_sum_[num_species]]);
  // perturb the temperature
  d_temperature = original_state_[num_species+1]*perturb_factor;
  GetJacobianColumnFromPerturbationLimiter(num_species+1,
					   d_temperature,
					   reactor_time,
					   &original_state_[0],
					   &original_derivative_[0],
					   &step_limiter[0],
					   &perturbed_state_[0],
					   &perturbed_derivative_[0],
					   &jacobian[jacobian_column_sum_[num_species+1]]);

  // Momentum
  jacobian[jacobian_column_sum_[num_species+2]] = 0.0;

  // PStrain
  //jacobian[jacobian_column_sum_[num_species+3]] = 0.0;

 return NONE;
}

void CounterflowReactor::Impl::
  GetJacobianColumnFromPerturbation(const int column_id,
                                    const double dstate,
                                    const double reactor_time,
                                    const double original_state[],
                                    const double original_derivative[],
                                    double perturbed_state[],
                                    double perturbed_derivative[],
                                    double jacobian_column[])
{
  const int num_states = GetNumStates();
  double dstate_exact;

  // copy the original state
  for(int j=0; j<num_states; ++j) {
    perturbed_state[j] = original_state[j];
  }
  perturbed_state[column_id] += dstate;

  // compute the perturbation as represented in the floating point arithmetic
  dstate_exact = perturbed_state[column_id]-original_state[column_id];

  // compute the time derivative at the perturbed state
  GetTimeDerivative(reactor_time,perturbed_state,perturbed_derivative);

  // compute the jacobian column using a finite difference approximation
  dstate_exact = 1.0/dstate_exact;
  for(int j=0; j<num_states; ++j) {
    jacobian_column[j] =
      dstate_exact*(perturbed_derivative[j]-original_derivative[j]);
  }
}

void CounterflowReactor::Impl::
  GetJacobianColumnFromPerturbationSteady(const int column_id,
					  const double dstate,
					  const double original_state[],
					  const double original_derivative[],
                                          const double step_limiter[],
					  double perturbed_state[],
					  double perturbed_derivative[],
					  double jacobian_column[])
{
  const int num_states = GetNumStates();
  double dstate_exact;

  // copy the original state
  for(int j=0; j<num_states; ++j) {
    perturbed_state[j] = original_state[j];
  }
  perturbed_state[column_id] += dstate;

  // compute the perturbation as represented in the floating point arithmetic
  dstate_exact = perturbed_state[column_id]-original_state[column_id];

  // compute the time derivative at the perturbed state
  GetTimeDerivativeSteady(perturbed_state,step_limiter,perturbed_derivative);

  // compute the jacobian column using a finite difference approximation
  dstate_exact = 1.0/dstate_exact;
  for(int j=0; j<num_states; ++j) {
    jacobian_column[j] =
      dstate_exact*(perturbed_derivative[j]-original_derivative[j]);
  }
}

void CounterflowReactor::Impl::
  GetJacobianColumnFromPerturbationLimiter(const int column_id,
					   const double dstate,
					   const double reactor_time,
					   const double original_state[],
					   const double original_derivative[],
					   const double step_limiter[],
					   double perturbed_state[],
					   double perturbed_derivative[],
					   double jacobian_column[])
{
  const int num_states = GetNumStates();
  double dstate_exact;

  // copy the original state
  for(int j=0; j<num_states; ++j) {
    perturbed_state[j] = original_state[j];
  }
  perturbed_state[column_id] += dstate;

  // compute the perturbation as represented in the floating point arithmetic
  dstate_exact = perturbed_state[column_id]-original_state[column_id];

  // compute the time derivative at the perturbed state
  GetTimeDerivativeLimiter(reactor_time,perturbed_state,step_limiter,perturbed_derivative);

  // compute the jacobian column using a finite difference approximation
  dstate_exact = 1.0/dstate_exact;
  for(int j=0; j<num_states; ++j) {
    jacobian_column[j] =
      dstate_exact*(perturbed_derivative[j]-original_derivative[j]);
  }
}

int CounterflowReactor::Impl::GetNetStoichiometry(const int species_id,
                                                    const int step_id)
{
  zerork::mechanism *mechanism_ptr;
  mechanism_ptr = GetMechanism();
  const int num_steps = mechanism_ptr->getNumSteps();
  int species_reactant_count=0;
  int species_product_count=0;

  if(step_id >= 0 && step_id < num_steps) {
    int num_reactants = mechanism_ptr->getOrderOfStep(step_id);
    int num_products  = mechanism_ptr->getNumProductsOfStep(step_id);

    // count the number of times the species appears as a reactant
    for(int j=0; j<num_reactants; ++j) {
      if(species_id == mechanism_ptr->getSpecIdxOfStepReactant(step_id,j)) {
        ++species_reactant_count;
      }
    }
    // count the number of times the species appears as a product
    for(int j=0; j<num_products; ++j) {
      if(species_id == mechanism_ptr->getSpecIdxOfStepProduct(step_id,j)) {
        ++species_product_count;
      }
    }
  }
  return species_product_count-species_reactant_count;
}


void CounterflowReactor::Impl::SetReferenceTemperature(const double ref_temperature)
{
  ref_temperature_     = ref_temperature;
  inv_ref_temperature_ = 1.0/ref_temperature;
  if(ref_temperature == 1.0) {
    use_scaled_state_ = false;
  } else {
    use_scaled_state_ = true;
  }
}

double CounterflowReactor::Impl::GetReferenceTemperature() const
{
  return ref_temperature_;
}
void CounterflowReactor::Impl::SetPressure(const double pressure)
{
  pressure_ = pressure;
  if(pressure_ <= 0.0) {
    printf("WARNING: can not set reactor pressure to %.18g [Pa].\n",
           pressure_);
    pressure_ = 1.01325e5;
    printf("         Setting reactor pressure to %.18g [Pa] as a default.\n",
           pressure_);
    fflush(stdout);
  }
}

double CounterflowReactor::Impl::GetPressure() const
{
  return pressure_;
}

double
  CounterflowReactor::Impl::GetMixtureSpecificHeat_Cp(const double state[])
{
  const int num_species            = GetNumSpecies();
  const double ref_temperature     = ref_temperature_;
  const double temperature         = ref_temperature*state[num_species+1];

  return GetMechanism()->getMassCpFromTY(temperature,
                                         &state[0],
                                         &specific_heats_[0]);
}

double
  CounterflowReactor::Impl::GetMixtureSpecificHeat_Cp(const double state[],
                                                        double *species_cp)
{
  const int num_species            = GetNumSpecies();
  const double ref_temperature     = ref_temperature_;
  const double temperature         = ref_temperature*state[num_species+1];

  double mass_cp = GetMechanism()->getMassCpFromTY(temperature,
                                                   &state[0],
                                                   species_cp);
  return mass_cp;
}

double
CounterflowReactor::Impl::GetMixtureSpecificHeat_Cp(const double temperature,
                                                 const double mass_fractions[],
                                                 double *species_cp)
{

  double mass_cp = GetMechanism()->getMassCpFromTY(temperature,
                                                   &mass_fractions[0],
                                                   species_cp);
  return mass_cp;
}

// ---------------------------------------------------------------------------
// Public facing API
CounterflowReactor::CounterflowReactor(const char mechanism_name[],
                                       const char thermodynamics_name[],
                                       const char parser_log_name[],
                                       const MatrixType matrix_type,
                                       const double pressure,
                                       const bool finite_separation)
{
  impl_ = new Impl(mechanism_name,
                   thermodynamics_name,
                   parser_log_name,
                   matrix_type,
                   pressure,
                   finite_separation);
}

CounterflowReactor::~CounterflowReactor()
{
  if(impl_ != NULL) {
    delete impl_;
  }
}

ReactorError
  CounterflowReactor::GetTimeDerivative(const double reactor_time,
                                           const double state[],
                                           double derivative[])
{
  return impl_->GetTimeDerivative(reactor_time,state,&derivative[0]);
}

ReactorError
  CounterflowReactor::GetTimeDerivativeSteady(const double state[],
                                                const double step_limiter[],
						double derivative[])
{
  return impl_->GetTimeDerivativeSteady(state,step_limiter,&derivative[0]);
}

ReactorError
  CounterflowReactor::GetTimeDerivativeLimiter(const double reactor_time,
						 const double state[],
						 const double step_limiter[],
						 double derivative[])
{
  return impl_->GetTimeDerivativeLimiter(reactor_time,state,step_limiter,&derivative[0]);
}

ReactorError CounterflowReactor::GetJacobian(const double reactor_time,
					       const double state[],
					       double jacobian[])
{
  return impl_->GetJacobian(reactor_time,state,&jacobian[0]);
}

ReactorError CounterflowReactor::GetJacobianSteady(const double state[],
                                                   const double convective[],
                                                   const bool Tfix,
                                                   const double ref_momentum,
                                                   const double step_limiter[],
                                                   double jacobian[])
{
  return impl_->GetJacobianSteady(state,convective,Tfix,ref_momentum,step_limiter,&jacobian[0]);
}

ReactorError CounterflowReactor::GetJacobianLimiter(const double reactor_time,
                                                    const double state[],
                                                    const double step_limiter[],
                                                    double jacobian[])
{
  return impl_->GetJacobianLimiter(reactor_time,state,step_limiter,&jacobian[0]);
}

int CounterflowReactor::GetJacobianPattern(int row_id[],
                                           int col_id[])
{
  return impl_->GetJacobianPattern(&row_id[0],&col_id[0]);
}

int CounterflowReactor::GetNumStates() const
{
  return impl_->GetNumStates();
}

int CounterflowReactor::GetNumSpecies() const
{
  return impl_->GetNumSpecies();
}

int CounterflowReactor::GetNumReactions() const
{
  return impl_->GetNumReactions();
}

int CounterflowReactor::GetNumSteps() const
{
  return impl_->GetNumSteps();
}

int CounterflowReactor::GetJacobianSize() const
{
  return impl_->GetJacobianSize();
}

MatrixType CounterflowReactor::GetMatrixType() const
{
  return impl_->GetMatrixType();
}

const char * CounterflowReactor::GetMechanismName() const
{
  return impl_->GetMechanismName();
}

const char * CounterflowReactor::GetThermodynamicsName() const
{
  return impl_->GetThermodynamicsName();
}

const char * CounterflowReactor::GetParserLogName() const
{
  return impl_->GetParserLogName();
}

const char * CounterflowReactor::GetReactorInfo() const
{
  return impl_->GetReactorInfo();
}

zerork::mechanism* CounterflowReactor::GetMechanism()
{
  return impl_->GetMechanism();
}

// species/reaction accessors
int CounterflowReactor::GetIdOfState(const char *state_name) const
{
  return impl_->GetIdOfState(state_name);
}

const char * CounterflowReactor::GetNameOfStateId(const int state_id) const
{
  return impl_->GetNameOfStateId(state_id);
}


// A-Factor sensitivity utilities
double CounterflowReactor::GetAMultiplierOfForwardReactionId(const int reaction_id) const
{
  return impl_->GetAMultiplierOfForwardReactionId(reaction_id);
}

ReactorError CounterflowReactor::SetAMultiplierOfForwardReactionId(const int reaction_id, const double a_multiplier)
{
  return impl_->SetAMultiplierOfForwardReactionId(reaction_id,a_multiplier);
}

double CounterflowReactor::GetAMultiplierOfReverseReactionId(const int reaction_id) const
{
  return impl_->GetAMultiplierOfReverseReactionId(reaction_id);
}

ReactorError CounterflowReactor::SetAMultiplierOfReverseReactionId(const int reaction_id, const double a_multiplier)
{
  return impl_->SetAMultiplierOfReverseReactionId(reaction_id, a_multiplier);
}

double CounterflowReactor::GetAMultiplierOfStepId(const int step_id) const
{
  return impl_->GetAMultiplierOfStepId(step_id);
}

ReactorError CounterflowReactor::SetAMultiplierOfStepId(const int step_id, const double a_multiplier)
{
  return impl_->SetAMultiplierOfStepId(step_id,a_multiplier);
}

void CounterflowReactor::SetReferenceTemperature(const double ref_temperature)
{
  impl_->SetReferenceTemperature(ref_temperature);
}
double CounterflowReactor::GetReferenceTemperature() const
{
  return impl_->GetReferenceTemperature();
}

void CounterflowReactor::SetPressure(const double pressure)
{
  impl_->SetPressure(pressure);
}
double CounterflowReactor::GetPressure() const
{
  return impl_->GetPressure();
}

double
  CounterflowReactor::GetMixtureSpecificHeat_Cp(const double state[])
{
  return impl_->GetMixtureSpecificHeat_Cp(state);
}

double
  CounterflowReactor::GetMixtureSpecificHeat_Cp(const double state[],
                                                  double *species_cp)
{
  return impl_->GetMixtureSpecificHeat_Cp(state, species_cp);
}

double
  CounterflowReactor::GetMixtureSpecificHeat_Cp(const double temperature,
                                             const double mass_fractions[],
                                             double *species_cp)
{
  return impl_->GetMixtureSpecificHeat_Cp(temperature,
                                          &mass_fractions[0],
                                          species_cp);
}

void CounterflowReactor::GetSpeciesHydrogenCount(int num_atoms[]) const
{
  impl_->GetSpeciesHydrogenCount(num_atoms);
}
void CounterflowReactor::GetSpeciesNitrogenCount(int num_atoms[]) const
{
  impl_->GetSpeciesNitrogenCount(num_atoms);
}
void CounterflowReactor::GetSpeciesCarbonCount(int num_atoms[]) const
{
  impl_->GetSpeciesCarbonCount(num_atoms);
}
void CounterflowReactor::GetSpeciesOxygenCount(int num_atoms[]) const
{
  impl_->GetSpeciesOxygenCount(num_atoms);
}
void CounterflowReactor::GetSpeciesMolecularWeight(double molecular_weight[]) const
{
  impl_->GetSpeciesMolecularWeight(molecular_weight);
}

double CounterflowReactor::GetGasConstant() const
{
  return impl_->GetGasConstant();
}
