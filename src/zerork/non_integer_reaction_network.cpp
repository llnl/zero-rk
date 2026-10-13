#include <limits.h>
#include <assert.h>

#include <utility> // std::pair and std::make_pair


#include "non_integer_reaction_network.h"


namespace zerork
{

static std::string DoubleToString(const double d)
{
  char double_c_str[32];
  sprintf(double_c_str,"%.18g",d);
  return std::string(double_c_str);
}

static std::string IntToString(const int i)
{
  char int_c_str[32];
  sprintf(int_c_str,"%d",i);
  return std::string(int_c_str);
}

static int FindIntVectorId(const int search_elem, std::vector<int> &vec) {
  const int nelems = vec.size();
  for(int j=0; j<nelems; ++j) {

    if(search_elem == vec[j]) {
      return j;
    }
  }
  return -1; // search_elem not found
}

NonIntegerReactionNetwork::NonIntegerReactionNetwork()
{
  num_non_integer_steps_     = 0;
  last_jacobian_step_count_  = 0;
  num_non_integer_reactions_ = 0;
  num_jacobian_nonzeros_ = 0; //Added that?
  max_step_id_ = INT_MIN;
  list_id_of_step_.clear();
  reaction_names_.clear();
  reaction_id_of_step_.clear();
  reaction_dir_of_step_.clear();
  params_.clear();

  ford_correction_k_ = 5;
  ford_correction_c0_ = log(1.0e-16);
  ford_correction_derivative_ = false;
  if(getenv("ZERORK_FORD_CORRECTION_DERIVATIVE") != NULL) {
     int ford_correction_env = atoi(getenv("ZERORK_FORD_CORRECTION_DERIVATIVE"));
     if(ford_correction_env!=0) {
       ford_correction_derivative_ = true;
     }
  }

}

int NonIntegerReactionNetwork::AddStep(const ckr::Reaction &ckreader_reaction,
		      const std::map<std::string, int> &id_of_name,
                      const int reaction_id,
                      const ReactionDirection reaction_dir, // FORWARD/REVERSE
                      const int step_id)
{
  NonIntegerStepParams add_params;

  std::map<std::string, double> forward_rop_map, reverse_rop_map;
  std::map<std::string, double>::const_iterator rop_it;

  // extract the necessary parameters from the reaction data
  add_params.step_id_ = step_id;
  // process the reactants stoichiometric data
  for(size_t j=0; j<ckreader_reaction.reactants.size(); ++j) {

    double stoich_num = ckreader_reaction.reactants[j].number;
    std::string species_name = ckreader_reaction.reactants[j].name;
    std::map<std::string, int>::const_iterator map_iter;

    map_iter = id_of_name.find(species_name);

    if(map_iter == id_of_name.end()) {
      // species not found, do not add step and return negative flag
      printf("# WARNING: In NonIntegerReactionNetwork::AddStep(...),\n");
      printf("#          could not add step id = %d (reaction id = %d)\n",
             step_id, reaction_id);
      printf("#          because species index not found for reactant = %s\n",
             species_name.c_str());
      fflush(stdout);
      return -1;
    }

    add_params.reactant_species_ids_.push_back(map_iter->second);
    add_params.reactant_stoich_num_.push_back(stoich_num);
    
    if(forward_rop_map.find(species_name) == forward_rop_map.end()) {
      forward_rop_map[species_name] = stoich_num;      
    } else {
      // CKParser does not combine repeated stoichiometric coefficients
      // on the same side of the reaction
      forward_rop_map[species_name] += stoich_num;
    }

  }
  // process the products stoichiometric data
  for(size_t j=0; j<ckreader_reaction.products.size(); ++j) {

    double stoich_num = ckreader_reaction.products[j].number;
    std::string species_name = ckreader_reaction.products[j].name;
    std::map<std::string, int>::const_iterator map_iter;

    map_iter = id_of_name.find(species_name);

    if(map_iter == id_of_name.end()) {
      // species not found, do not add step and return negative flag
      printf("# WARNING: In NonIntegerReactionNetwork::AddStep(...),\n");
      printf("#          could not add step id = %d (reaction id = %d)\n",
             step_id, reaction_id);
      printf("#          because species index not found for product = %s\n",
             species_name.c_str());
      fflush(stdout);
      return -1;
    }

    add_params.product_species_ids_.push_back(map_iter->second);
    add_params.product_stoich_num_.push_back(stoich_num);

    if(reverse_rop_map.find(species_name) == reverse_rop_map.end()) {
      reverse_rop_map[species_name] = stoich_num;
    } else {
      // CKParser does not combine repeated stoichiometric coefficients
      // on the same side of the reaction
      reverse_rop_map[species_name] += stoich_num;
    }

  }

  // Process the rate-of-progress concentration exponents specified by
  // the FORD and RORD keywords. If the FORD (or RORD)  keyword isn't present, 
  // then the rate-of-progress concentration exponents are the same as the 
  // reactants (or products). 
  //
  // forward_rop_map and reverse_rop_map at this point are set to the
  // stoichiometric values 
    
  // Overwrite if the FORD or RORD keywords are found
  if(ckreader_reaction.isRealOrder) {
    //printf("# DEBUG: Add step found FORD/RORD\n"); fflush(stdout);

    for(rop_it=ckreader_reaction.fwdOrder.begin();
      rop_it != ckreader_reaction.fwdOrder.end(); rop_it++) {
      
      forward_rop_map[rop_it->first] = rop_it->second;
    }

    for(rop_it=ckreader_reaction.revOrder.begin();
      rop_it != ckreader_reaction.revOrder.end(); rop_it++) {
      
      reverse_rop_map[rop_it->first] = rop_it->second;
    }
  }
  // forward_rop_map and reverse_rop_map at this point are set to the
  // final real order values. 

  // Copy forward_rop_map info to add_params.rop_species_ids_ and
  // add_params.rop_concetration_powers_
  
  for(rop_it=forward_rop_map.begin(); 
    rop_it != forward_rop_map.end(); rop_it++) {

    std::map<std::string, int>::const_iterator map_iter;
    map_iter = id_of_name.find(rop_it->first);

    if(map_iter == id_of_name.end()) {
      // species not found, do not add step and return negative flag
      printf("# WARNING: In NonIntegerReactionNetwork::AddStep(...),\n");
      printf("#          could not add step id = %d (reaction id = %d)\n",
             step_id, reaction_id);
      printf("#          because species index not found for reactant = %s\n",
             rop_it->first.c_str());
      fflush(stdout);
      return -1;
    }
    add_params.rop_species_ids_.push_back(map_iter->second);
    add_params.rop_concentration_powers_.push_back(rop_it->second);  
  }

  // Copy reverse_rop_map info to add_params.reverse_rop_species_ids_ and
  // add_params.reverse_rop_concentration_powers_
  //
  // The reverse rate-of-progress concentration information is also stored to
  // facilitate the calculation of the equilibrium constant

  for(rop_it=reverse_rop_map.begin();
    rop_it != reverse_rop_map.end(); rop_it++) {

    std::map<std::string, int>::const_iterator map_iter;
    map_iter = id_of_name.find(rop_it->first);

    if(map_iter == id_of_name.end()) {
      // species not found, do not add step and return negative flag
      printf("# WARNING: In NonIntegerReactionNetwork::AddStep(...),\n");
      printf("#          could not add step id = %d (reaction id = %d)\n",
             step_id, reaction_id);
      printf("#          because species index not found for product = %s\n",
             rop_it->first.c_str());
      fflush(stdout);
      return -1;
    }
    add_params.reverse_rop_species_ids_.push_back(map_iter->second);
    add_params.reverse_rop_concentration_powers_.push_back(rop_it->second);  
  }

  // swap the reactants and products if the direction is reverse
  // seap forward and reverse ROP vectors if the direction is reverse
  if(reaction_dir == REVERSE) {
    
    add_params.reactant_species_ids_.swap(add_params.product_species_ids_);
    add_params.reactant_stoich_num_.swap(add_params.product_stoich_num_);    

    add_params.rop_species_ids_.swap(add_params.reverse_rop_species_ids_);
    add_params.rop_concentration_powers_.swap(
      add_params.reverse_rop_concentration_powers_);
  }


  // At this point the NonIntegerStepParams data is set

  if(!HasReaction(reaction_id)) {
    // add new reaction to the list
    std::string reaction_str = GetReactionString(ckreader_reaction);
    //printf("reaction = %s\n", reaction_str.c_str());
    reaction_names_[reaction_id] = reaction_str;
    ++num_non_integer_reactions_;
  }

  if(HasStep(step_id)) {
    printf("# WARNING: In NonIntegerReactionNetwork::AddStep(...),\n");
    printf("#          overwriting existing step id = %d.\n", step_id);
    fflush(stdout);

    int list_id = GetListIndexOfStep(step_id);

    if(list_id == -1) {
      printf("# WARNING: In NonIntegerReactionNetwork::AddStep(...),\n");
      printf("#          can not find step id = %d that has been recorded.\n",
             step_id);
      printf("#          Could not add step id = %d (reaction id = %d).\n",
             step_id, reaction_id);
      fflush(stdout);
      return -1;
    }
    // replace to the step_id map data
    reaction_id_of_step_[step_id]  = reaction_id;
    reaction_dir_of_step_[step_id] = reaction_dir;
    params_[list_id] = add_params;
  } else {
    // add to the step_id map data
    reaction_id_of_step_[step_id]  = reaction_id;
    reaction_dir_of_step_[step_id] = reaction_dir;
    params_.push_back(add_params);
    // add step_id to the
    if(step_id > max_step_id_) {
      // resize will store the value -1 for all new data elements
      list_id_of_step_.resize(step_id+1, -1);
      list_id_of_step_[step_id] = params_.size() - 1;
      max_step_id_ = step_id;
    }

    ++num_non_integer_steps_;
  }

  return 0;
}


int NonIntegerReactionNetwork::UpdateRatesOfProgress(const double concentrations[], double rates_of_progress[]) const
{
  const int num_non_integer_steps = num_non_integer_steps_;

  for(int j=0; j<num_non_integer_steps; ++j) {

    const int num_terms = static_cast<int>(params_[j].rop_species_ids_.size());
    bool negative_concentration = false;

    for(int k=0; k<num_terms; ++k) {

      const double species_concentration =
        concentrations[params_[j].rop_species_ids_[k]];
      const double multiplier =
        pow(fabs(species_concentration),
            params_[j].rop_concentration_powers_[k]);
      double correction = 1.0;

      if(species_concentration < 0.0) {
	negative_concentration = true;
      }
      if(ford_correction_derivative_) {
        const double kx = ford_correction_k_;
        const double x0 = ford_correction_c0_;
	const double x = log(max(1.0e-30,species_concentration));
        const double sigmoid = 1/(1+exp(-kx*(x-x0)));
        correction = sigmoid + (1-sigmoid)*pow(fabs(species_concentration),1-params_[j].rop_concentration_powers_[k]);
      }

      rates_of_progress[params_[j].step_id_] *=  multiplier*correction;
    } // k-loop over reactants

    // if one or more species has a negative concentration we
    if(negative_concentration) {
      rates_of_progress[params_[j].step_id_] *= -1.0;
    }

  } // j-loop over non-integer steps

  return 0;
}

int NonIntegerReactionNetwork::GetCreationRates(const double rates_of_progress[], double creation_rates[]) const
{
  const int num_non_integer_steps = num_non_integer_steps_;

  for(int j=0; j<num_non_integer_steps; ++j) {

    const int num_terms =
      static_cast<int>(params_[j].product_species_ids_.size());

    for(int k=0; k<num_terms; ++k) {
      creation_rates[params_[j].product_species_ids_[k]] +=
        params_[j].product_stoich_num_[k]*
        rates_of_progress[params_[j].step_id_];
    }
  }

  return 0;
}

int NonIntegerReactionNetwork::GetDestructionRates(const double rates_of_progress[], double destruction_rates[]) const
{
  const int num_non_integer_steps = num_non_integer_steps_;

  for(int j=0; j<num_non_integer_steps; ++j) {

    const int num_terms =
      static_cast<int>(params_[j].reactant_species_ids_.size());

    for(int k=0; k<num_terms; ++k) {
      destruction_rates[params_[j].reactant_species_ids_[k]] +=
        params_[j].reactant_stoich_num_[k]*
        rates_of_progress[params_[j].step_id_];
    }
  }

  return 0;
}

double NonIntegerReactionNetwork::GetThermoChangeOfStep(const int step_id, const double species_thermo[]) const
{
  int list_id = GetListIndexOfStep(step_id);
  if(list_id < 0) {
    printf("# WARNING: In NonIntegerReactionNetwork::GetThermoChangeOfStep(id),\n");
    printf("#          step id = %d not found in the NonIntegerReactionNetwork\n", step_id);
    printf("#          parameter list.  Returning zero.\n");
    fflush(stdout);
    return 0.0;
  }

  double thermo_sum = 0.0;
  const int num_products  = params_[list_id].reverse_rop_species_ids_.size();
  const int num_reactants = params_[list_id].rop_species_ids_.size();
  // add the products
  for(int j=0; j<num_products; ++j) {
    thermo_sum += params_[list_id].reverse_rop_concentration_powers_[j]*
      species_thermo[params_[list_id].reverse_rop_species_ids_[j]];
  }
  // subtract reactants
  for(int j=0; j<num_reactants; ++j) {
    thermo_sum -= params_[list_id].rop_concentration_powers_[j]*
      species_thermo[params_[list_id].rop_species_ids_[j]];
  }
  return thermo_sum;
}

int NonIntegerReactionNetwork::GetProductIndexOfStep(const int step_id, const int prod_id) const
{
  int list_id = GetListIndexOfStep(step_id);
  if(list_id < 0) {
    printf("# WARNING: In NonIntegerReactionNetwork::GetProductIndexOfStep(step_id, prod_id),\n");
    printf("#          step id = %d not found in the NonIntegerReactionNetwork\n", step_id);
    printf("#          parameter list.  Returning -1.\n");
    fflush(stdout);
    return -1;
  }

  const int num_products = params_[list_id].reverse_rop_species_ids_.size();
  if(prod_id >= num_products) {
    printf("# WARNING: In NonIntegerReactionNetwork::GetReactantIndexOfStep(step_id, prod_id),\n");
    printf("#          product id = %d is greater than number of products in reaction (%d)\n", prod_id, num_products);
    printf("#          Returning -1.\n");
    fflush(stdout);
    return -1;
  }
  return params_[list_id].reverse_rop_species_ids_[prod_id];
}

int NonIntegerReactionNetwork::GetReactantIndexOfStep(const int step_id, const int react_id) const
{
  int list_id = GetListIndexOfStep(step_id);
  if(list_id < 0) {
    printf("# WARNING: In NonIntegerReactionNetwork::GetReactantIndexOfStep(step_id, react_id),\n");
    printf("#          step id = %d not found in the NonIntegerReactionNetwork\n", step_id);
    printf("#          parameter list.  Returning -1.\n");
    fflush(stdout);
    return -1;
  }

  const int num_reactants = params_[list_id].rop_species_ids_.size();
  if(react_id >= num_reactants) {
    printf("# WARNING: In NonIntegerReactionNetwork::GetReactantIndexOfStep(step_id, react_id),\n");
    printf("#          reactant id = %d is greater than number of reactants in reaction (%d)\n", react_id, num_reactants);
    printf("#          Returning -1.\n");
    fflush(stdout);
    return -1;
  }
  return params_[list_id].rop_species_ids_[react_id];
}

double NonIntegerReactionNetwork::GetProductPowerOfStep(const int step_id, const int prod_id) const
{
  int list_id = GetListIndexOfStep(step_id);
  if(list_id < 0) {
    printf("# WARNING: In NonIntegerReactionNetwork::GetProductPowerOfStep(step_id, react_id),\n");
    printf("#          step id = %d not found in the NonIntegerReactionNetwork\n", step_id);
    printf("#          parameter list.  Returning 0.\n");
    fflush(stdout);
    return 0.0;
  }

  const int num_products = params_[list_id].reverse_rop_species_ids_.size();
  if(prod_id >= num_products) {
    printf("# WARNING: In NonIntegerReactionNetwork::GetReactantPowerOfStep(step_id, prod_id),\n");
    printf("#          product id = %d is greater than number of products in reaction (%d)\n", prod_id, num_products);
    printf("#          Returning 0.\n");
    fflush(stdout);
    return 0.0;
  }
  return params_[list_id].reverse_rop_concentration_powers_[prod_id];
}

double NonIntegerReactionNetwork::GetReactantPowerOfStep(const int step_id, const int react_id) const
{
  int list_id = GetListIndexOfStep(step_id);
  if(list_id < 0) {
    printf("# WARNING: In NonIntegerReactionNetwork::GetReactantPowerOfStep(step_id, react_id),\n");
    printf("#          step id = %d not found in the NonIntegerReactionNetwork\n", step_id);
    printf("#          parameter list.  Returning -1.\n");
    fflush(stdout);
    return 0.0;
  }

  const int num_reactants = params_[list_id].rop_species_ids_.size();
  if(react_id >= num_reactants) {
    printf("# WARNING: In NonIntegerReactionNetwork::GetReactantPowerOfStep(step_id, react_id),\n");
    printf("#          reactant id = %d is greater than number of reactants in reaction (%d)\n", react_id, num_reactants);
    printf("#          Returning -1.\n");
    fflush(stdout);
    return 0.0;
  }
  return params_[list_id].rop_concentration_powers_[react_id];
}

int NonIntegerReactionNetwork::GetNumProductsOfStep(const int step_id) const
{
  const int list_id = GetListIndexOfStep(step_id);
  if(list_id < 0) {
    printf("# WARNING: In NonIntegerReactionNetwork::GetNumProductsOfStep(step_id),\n");
    printf("#          step id = %d not found in the NonIntegerReactionNetwork\n", step_id);
    printf("#          parameter list.  Returning -1.\n");
    fflush(stdout);
    return -1;
  }

  return params_[list_id].reverse_rop_species_ids_.size();
}

int NonIntegerReactionNetwork::GetNumReactantsOfStep(const int step_id) const
{
  const int list_id = GetListIndexOfStep(step_id);
  if(list_id < 0) {
    printf("# WARNING: In NonIntegerReactionNetwork::GetNumReactantsOfStep(step_id),\n");
    printf("#          step id = %d not found in the NonIntegerReactionNetwork\n", step_id);
    printf("#          parameter list.  Returning -1.\n");
    fflush(stdout);
    return -1;
  }

  return params_[list_id].rop_species_ids_.size();
}

int NonIntegerReactionNetwork::GetJacobianParameters(int jacobian_term_indexes[],
                                                     int jacobian_concentration_indexes[],
                                                     int jacobian_step_indexes[],
                                                     double jacobian_multipliers[]) const
{
  const int num_terms = process_step_id_.size();
  for(int j=0; j<num_terms; ++j) {
    jacobian_term_indexes[j] = process_jacobian_id_[j];
    jacobian_concentration_indexes[j]   = process_concentration_id_[j];
    jacobian_step_indexes[j]   = process_step_id_[j];
    jacobian_multipliers[j]    = process_multiplier_[j];
  }
  return 0;
}

int NonIntegerReactionNetwork::GetSpeciesJacobian(
                                           const double inv_concentration[],
                                           const double rates_of_progress[],
                                           double jacobian[]) const
{
  const int num_terms = process_step_id_.size();
  for(int j=0; j<num_terms; ++j) {

    jacobian[process_jacobian_id_[j]] +=
      process_multiplier_[j]*rates_of_progress[process_step_id_[j]]*
      inv_concentration[process_concentration_id_[j]];

  }
  return 0;
}

bool NonIntegerReactionNetwork::HasStep(const int step_id) const
{
  std::map<int, int>::const_iterator map_iter;

  map_iter = reaction_id_of_step_.find(step_id);
  if(map_iter != reaction_id_of_step_.end()) {
    // found step id
    return true;
  }
  return false;
}

bool NonIntegerReactionNetwork::HasReaction(const int reaction_id) const
{
  std::map<int, std::string>::const_iterator map_iter;

  map_iter = reaction_names_.find(reaction_id);
  if(map_iter != reaction_names_.end()) {
    // found step id
    return true;
  }
  return false;
}


std::string NonIntegerReactionNetwork::GetNameOfReaction(const int reaction_id) const
{
  std::string str;
  std::map<int, std::string>::const_iterator map_iter;
  if(HasReaction(reaction_id)) {
    map_iter = reaction_names_.find(reaction_id);
    str = map_iter->second;
  } else {
  str = std::string("Reaction id = ") +
    IntToString(reaction_id) + " not found in the NonIntegerReactionNetwork";
  }
  return str;
}


std::string NonIntegerReactionNetwork::GetReactionString(const ckr::Reaction &ckreader_reaction) const
{
  std::string reaction_str;

  reaction_str.clear();
  // process the reactants
  for(size_t j=0; j<ckreader_reaction.reactants.size(); ++j) {

    if(ckreader_reaction.reactants[j].number != 1.0) {
      reaction_str += DoubleToString(ckreader_reaction.reactants[j].number) +
        std::string(" ");
    }
    reaction_str += ckreader_reaction.reactants[j].name;
    if(j+1 != ckreader_reaction.reactants.size()) {
      reaction_str += std::string(" + ");
    }
  }
  // check for third bodies or falloff
  if(ckreader_reaction.isThreeBodyRxn) {
    reaction_str += std::string(" + M");
  } else if(ckreader_reaction.isFalloffRxn) {
    reaction_str += std::string(" (+") + ckreader_reaction.thirdBody +
      std::string(")");
  }

  // add the equal sign
  if(ckreader_reaction.isReversible &&
     ckreader_reaction.krev.A != 0.0) {
    // reversible reaction
    reaction_str += std::string(" <=> ");
  } else {
    // irreversible reaction
    reaction_str += std::string(" => ");
  }

  // process the products
  for(size_t j=0; j<ckreader_reaction.products.size(); ++j) {

    if(ckreader_reaction.products[j].number != 1.0) {
      reaction_str += DoubleToString(ckreader_reaction.products[j].number) +
        std::string(" ");
    }
    reaction_str += ckreader_reaction.products[j].name;
    if(j+1 != ckreader_reaction.products.size()) {
      reaction_str += std::string(" + ");
    }
  }
  // check for third bodies or falloff
  if(ckreader_reaction.isThreeBodyRxn) {
    reaction_str += std::string(" + M");
  } else if(ckreader_reaction.isFalloffRxn) {
    reaction_str += std::string(" (+") + ckreader_reaction.thirdBody +
      std::string(")");
  }
  return reaction_str;
}

int NonIntegerReactionNetwork::GetListIndexOfStep(const int step_id) const
{
  if(step_id > max_step_id_) {
    return -1;
  }

  return list_id_of_step_[step_id];
}

double NonIntegerReactionNetwork::GetOrderOfStep(const int step_id) const
{
  int list_id = GetListIndexOfStep(step_id);

  if(list_id < 0) {
    printf("# WARNING: In NonIntegerReactionNetwork::GetOrderOfStep(id),\n");
    printf("#          step id = %d not found in the NonIntegerReactionNetwork\n", step_id);
    printf("#          parameter list.  Returning zero.\n");
    fflush(stdout);
    return 0.0;
  }

  double step_order = 0.0;
  for(size_t j=0; j<params_[list_id].rop_concentration_powers_.size(); ++j) {
    step_order += params_[list_id].rop_concentration_powers_[j];
  }
  return step_order;
}

double NonIntegerReactionNetwork::GetNumProductMolesOfStep(const int step_id) const
{
  int list_id = GetListIndexOfStep(step_id);

  if(list_id < 0) {
    printf("# WARNING: In NonIntegerReactionNetwork:::GetNumProductMolesOfStep(id),\n");
    printf("#          step id = %d not found in the NonIntegerReactionNetwork\n", step_id);
    printf("#          parameter list.  Returning zero.\n");
    fflush(stdout);
    return 0.0;
  }

  double stoich_sum = 0.0;
  for(size_t j=0; j<params_[list_id].product_stoich_num_.size(); ++j) {
    stoich_sum += params_[list_id].product_stoich_num_[j];
  }
  return stoich_sum;
}

double NonIntegerReactionNetwork::GetNumReactantMolesOfStep(const int step_id) const
{
  int list_id = GetListIndexOfStep(step_id);

  if(list_id < 0) {
    printf("# WARNING: In NonIntegerReactionNetwork:::GetNumReactantMolesOfStep(id),\n");
    printf("#          step id = %d not found in the NonIntegerReactionNetwork\n", step_id);
    printf("#          parameter list.  Returning zero.\n");
    fflush(stdout);
    return 0.0;
  }

  double stoich_sum = 0.0;
  for(size_t j=0; j<params_[list_id].reactant_stoich_num_.size(); ++j) {
    stoich_sum += params_[list_id].reactant_stoich_num_[j];
  }
  return stoich_sum;
}

std::vector<double> NonIntegerReactionNetwork::GetRateOfProgressConcentrationPowersOfStep(const int step_id) const
{
  int list_id = GetListIndexOfStep(step_id);

  if(list_id < 0) {
    printf("# WARNING: In NonIntegerReactionNetwork:::GetRateOfProgressConcentrationPowersOfStep(id),\n");
    printf("#          step id = %d not found in the NonIntegerReactionNetwork\n", step_id);
    printf("#          parameter list.  Returning empty vector.\n");
    fflush(stdout);
    return std::vector<double>();
  }
  return params_[list_id].rop_concentration_powers_;
}

std::vector<int> NonIntegerReactionNetwork::GetReactantIndexesOfStep(const int step_id) const
{
  int list_id = GetListIndexOfStep(step_id);

  if(list_id < 0) {
    printf("# WARNING: In NonIntegerReactionNetwork:::GetReactantIndexesOfStep(id),\n");
    printf("#          step id = %d not found in the NonIntegerReactionNetwork\n", step_id);
    printf("#          parameter list.  Returning empty vector.\n");
    fflush(stdout);
    return std::vector<int>();
  }
  return params_[list_id].reactant_species_ids_;
}

std::vector<int> NonIntegerReactionNetwork::GetProductIndexesOfStep(const int step_id) const
{
  int list_id = GetListIndexOfStep(step_id);

  if(list_id < 0) {
    printf("# WARNING: In NonIntegerReactionNetwork:::GetProductStoichNumsOfStep(id),\n");
    printf("#          step id = %d not found in the NonIntegerReactionNetwork\n", step_id);
    printf("#          parameter list.  Returning empty vector.\n");
    fflush(stdout);
    return std::vector<int>();
  }
  return params_[list_id].product_species_ids_;
}

std::vector<double> NonIntegerReactionNetwork::GetReactantStoichNumsOfStep(const int step_id) const
{
  int list_id = GetListIndexOfStep(step_id);

  if(list_id < 0) {
    printf("# WARNING: In NonIntegerReactionNetwork:::GetReactantStoichNumsOfStep(id),\n");
    printf("#          step id = %d not found in the NonIntegerReactionNetwork\n", step_id);
    printf("#          parameter list.  Returning empty vector.\n");
    fflush(stdout);
    return std::vector<double>();
  }
  return params_[list_id].reactant_stoich_num_;
}

std::vector<double> NonIntegerReactionNetwork::GetProductStoichNumsOfStep(const int step_id) const
{
  int list_id = GetListIndexOfStep(step_id);

  if(list_id < 0) {
    printf("# WARNING: In NonIntegerReactionNetwork:::GetProductStoichNumsOfStep(id),\n");
    printf("#          step id = %d not found in the NonIntegerReactionNetwork\n", step_id);
    printf("#          parameter list.  Returning empty vector.\n");
    fflush(stdout);
    return std::vector<double>();
  }
  return params_[list_id].product_stoich_num_;
}

int NonIntegerReactionNetwork::GetNumJacobianNonzeros()
{
  if(last_jacobian_step_count_ != num_non_integer_steps_) {
    BuildJacobian();
  }
  return num_jacobian_nonzeros_;
}

int NonIntegerReactionNetwork::GetNumJacobianTerms()
{
  if(last_jacobian_step_count_ != num_non_integer_steps_) {
    BuildJacobian();
  }
  const int num_terms = process_step_id_.size();
  return num_terms;
}

int NonIntegerReactionNetwork::GetJacobianPattern(int row_id[],
                                                  int column_id[])
{
  if(last_jacobian_step_count_ != num_non_integer_steps_) {
    BuildJacobian();
  }

  const int num_nonzeros = jacobian_row_id_.size();
  for(int j=0; j<num_nonzeros; ++j) {
    row_id[j]    = jacobian_row_id_[j];
    column_id[j] = jacobian_column_id_[j];
  }
  return num_nonzeros;
}

bool NonIntegerReactionNetwork::GetFORDCorrectionDerivative() const
{
  return ford_correction_derivative_;
}

double NonIntegerReactionNetwork::GetFORDCorrectionDecay() const
{
  return ford_correction_k_;
}

double NonIntegerReactionNetwork::GetFORDCorrectionOffset() const
{
  return ford_correction_c0_;
}

void NonIntegerReactionNetwork::BuildJacobian()
{
  if(last_jacobian_step_count_ != num_non_integer_steps_) {

    // In comparison between two pairs, the relationship operator is applied
    // first to compare the first element of each pair.  If the relationship
    // operator evaluates true for the first elements, then the overall
    // operator evaluates true.  If the relationship operator evaluates
    // false for the first elements and true for the second, it still
    // evaluates true.  Since we want to sort the matrix elements in
    // a compressed column order, the first element of the pair needs
    // to be the column index.
    //
    // column_index = column_row_pair.first;  // or ROP perturbation species id
    // row_index    = column_row_pair.second;
    //
    std::pair<int,int> column_row_pair;
    std::map<std::pair<int,int>,int> matrix_element_count;
    std::map<std::pair<int,int>,int>::iterator iter;

    matrix_element_count.clear();
    process_step_id_.clear();
    process_concentration_id_.clear();
    process_jacobian_id_.clear();
    process_multiplier_.clear();

    for(size_t j=0; j<params_.size(); ++j) {

      for(size_t k=0; k<params_[j].rop_species_ids_.size(); ++k) {

        // column index
        int perturbed_species_id = params_[j].rop_species_ids_[k];

        if(params_[j].rop_concentration_powers_[k] != 0.0) {

          // Add the row index for each reactant
          for(size_t m=0; m<params_[j].reactant_species_ids_.size(); ++m) {

            int row_index = params_[j].reactant_species_ids_[m];
            column_row_pair = std::make_pair(perturbed_species_id, row_index);

            // Record non-zero Jacobian element
            iter = matrix_element_count.find(column_row_pair);
            if(iter == matrix_element_count.end()) {
              // Add new row and column index pair
              matrix_element_count[column_row_pair] = 1;
            } else {
              ++matrix_element_count[column_row_pair];
            }

            // Setup the Jacobian processing arrays
            double multiplier = -params_[j].reactant_stoich_num_[m]*
              params_[j].rop_concentration_powers_[k];

            process_step_id_.push_back(params_[j].step_id_);
            process_concentration_id_.push_back(perturbed_species_id);
            // Temporarily store the row index to use to look up the
            // sparse jacobian index after all the terms are recorded
            process_jacobian_id_.push_back(row_index);
            process_multiplier_.push_back(multiplier);

          } // end for m-loop over params_[j].reactant_species_ids_

          // Add the row index for each product
          for(size_t m=0; m<params_[j].product_species_ids_.size(); ++m) {

            int row_index = params_[j].product_species_ids_[m];
            column_row_pair = std::make_pair(perturbed_species_id, row_index);

            // Record non-zero Jacobian element
            iter = matrix_element_count.find(column_row_pair);
            if(iter == matrix_element_count.end()) {
              // Add new row and column index pair
              matrix_element_count[column_row_pair] = 1;
            } else {
              ++matrix_element_count[column_row_pair];
            }

            // Setup the Jacobian processing arrays
            double multiplier = params_[j].product_stoich_num_[m]*
              params_[j].rop_concentration_powers_[k];

            process_step_id_.push_back(params_[j].step_id_);
            process_concentration_id_.push_back(perturbed_species_id);
            // Temporarily store the row index to use to look up the
            // sparse jacobian index after all the terms are recorded
            process_jacobian_id_.push_back(row_index);
            process_multiplier_.push_back(multiplier);

          } // end for loop-m over params_[j].product_species_ids_ vector

        } // end if (params_[j].rop_concentration_powers_[k] != 0.0)

      } // end for loop-k over params_[j].rop_species_ids_ vector

    } // end for loop-j over params_ vector

    // Update the class members
    last_jacobian_step_count_ = num_non_integer_steps_;
    num_jacobian_nonzeros_ = matrix_element_count.size();

    jacobian_row_id_.clear();
    jacobian_column_id_.clear();

    int sparse_id = 0;
    // recompute the Jacobian non zero structure information
    for(iter = matrix_element_count.begin();
        iter != matrix_element_count.end();
        ++iter) {

      // note that the matrix element count map uses a pair for a key where
      // the first element of the pair is the column index and the second
      // element is the row index
      jacobian_column_id_.push_back(iter->first.first);
      jacobian_row_id_.push_back(iter->first.second);

      // reset the mapped value to the sparse_id
      iter->second = sparse_id;
      ++sparse_id;
    }

    // Use the temporarily stored row index to look up the Jacobian
    // index and update the process array
    for(size_t j=0; j<process_step_id_.size(); ++j) {

      int row_index            = process_jacobian_id_[j];
      int perturbed_species_id = process_concentration_id_[j];

      column_row_pair = std::make_pair(perturbed_species_id, row_index);
      iter = matrix_element_count.find(column_row_pair);
//      if(iter == matrix_element_count.end()) {
//        printf("# ERROR: In NonIntegerReactionNetwork::BuildJacobian(),\n"
//               "#        can not find Jacobian sparse index for\n"
//               "#        element (row id = %d, column id =%d)\n",
//               row_index,
//               perturbed_species_id);
//      }
      assert(iter != matrix_element_count.end());
      process_jacobian_id_[j] = iter->second; // set sparse index
    }


  } // end if (last_jacobian_step_count_ != num_non_integer_steps_)

}


} // namespace zerork
