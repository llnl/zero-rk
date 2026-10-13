#ifndef NON_INTEGER_REACTION_NETWORK_H_
#define NON_INTEGER_REACTION_NETWORK_H_

#include <string>
#include <vector>
#include <map>

#include "../CKconverter/CKReader.h"

namespace zerork{

enum ReactionDirection {FORWARD, REVERSE};

class NonIntegerReactionNetwork
{
 public:
  NonIntegerReactionNetwork();
  //~NonIntegerReactionNetwork();


  // Core setup and calculation functions.  All return zero if succssful.
  //
  // AddStep does not perform any consistency check with respect to the
  // reaction definition, species name to index map, reaction index, reaction
  // direction, and step index.  These are assumed to be created consistently
  // from the parser object accessed in info_net.cpp.
  int AddStep(const ckr::Reaction &ckreader_reaction,
	      const std::map<std::string, int> &id_of_name,
              const int reaction_id,
              const ReactionDirection reaction_dir, // FORWARD/REVERSE
              const int step_id);

  int UpdateRatesOfProgress(const double concentrations[],
                            double rates_of_progress[]) const;
  int GetCreationRates(const double rates_of_progress[], 
                       double creation_rates[]) const;
  int GetDestructionRates(const double rates_of_progress[], 
                          double destruction_rates[]) const;

  double GetThermoChangeOfStep(const int step_id,
                               const double species_thermo[]) const;
  int GetSpeciesJacobian(const double inv_concentration[],
                         const double rates_of_progress[],
                         double jacobian[]) const;

  // -------------------------------------------------------------------------
  // Accessors 
  //
  // id refers to the overall mechanism reaction index or step index specified
  // by reaction_id or step_id in the AddStep method
  bool HasStep(const int step_id) const;
  bool HasReaction(const int reaction_id) const;
  // If reaction index is not in the reaction network, then a message stating
  // the index is not found is returned
  std::string GetNameOfReaction(const int reaction_id) const;
  
  int GetNumNonIntegerSteps() const {return num_non_integer_steps_;}
  int GetNumNonIntegerReactions() const {return num_non_integer_reactions_;}

  // GetOrderOfStep is the sum of the rate-of-progress concentrtion powers.
  // If the step index does not exist, zero is returned.
  double GetOrderOfStep(const int step_id) const;

  // GetNumProductMolesOfStep is the sum of the products' stoichiometric 
  // coefficients.  If the step index does not exist, zero is returned.
  double GetNumProductMolesOfStep(const int step_id) const;

  // GetNumReactantMolesOfStep is the sum of the reactants' stoichiometric 
  // coefficients.  If the step index does not exist, zero is returned.
  double GetNumReactantMolesOfStep(const int step_id) const;

  // Accessor functions so GPU rates can build it's own data structures
  int GetNumReactantsOfStep(const int step_id) const;
  int GetNumProductsOfStep(const int step_id) const;
  double GetReactantPowerOfStep(const int step_id, const int react_id) const;
  double GetProductPowerOfStep(const int step_id, const int prod_id) const;
  int GetReactantIndexOfStep(const int step_id, const int react_id) const;
  int GetProductIndexOfStep(const int step_id, const int prod_id) const;

  // Accessor functions so GPU perf_net can build it's own data structures
  std::vector<double> GetRateOfProgressConcentrationPowersOfStep(const int step_id) const;
  std::vector<int> GetReactantIndexesOfStep(const int step_id) const;
  std::vector<int> GetProductIndexesOfStep(const int step_id) const;
  std::vector<double> GetReactantStoichNumsOfStep(const int step_id) const;
  std::vector<double> GetProductStoichNumsOfStep(const int step_id) const;

  // Accessor function so GPU reactors can build their own data structures
  int GetNumJacobianTerms();
  int GetJacobianParameters(int jacobian_term_indexes[],
                            int jacobian_concentration_indexes[],
                            int jacobian_step_indexes[],
                            double jacobian_multipliers[]) const;

  // TODO: make constant by making the underlying structures mutable
  int GetNumJacobianNonzeros();
  int GetJacobianPattern(int row_id[], 
                         int column_id[]);

  bool GetFORDCorrectionDerivative() const;
  double GetFORDCorrectionDecay() const;
  double GetFORDCorrectionOffset() const;
 private:

  std::string GetReactionString(const ckr::Reaction &ckreader_reaction) const;
  int GetListIndexOfStep(const int step_id) const;

  typedef struct {
    // stoichiometric data
    std::vector<int>    reactant_species_ids_;
    std::vector<double> reactant_stoich_num_;
    std::vector<int>    product_species_ids_;
    std::vector<double> product_stoich_num_;
    // The stoichiometric data is used to compute the concentration product
    // for the rate of progress, unless this behavior is over-ridden by
    // the FORD or RORD keywords.  That is, the default non-integer reaction
    // sets rop_species_ids                  <- reactant_species_ids
    //      rop_concentration_powers         <- reactant_stoich_num;
    //      reverse_rop_species_ids          <- product_species_ids
    //      reverse_rop_concentration_powers <- product_stoich_num;
    std::vector<int>    rop_species_ids_;
    std::vector<double> rop_concentration_powers_;
    std::vector<int>    reverse_rop_species_ids_;
    std::vector<double> reverse_rop_concentration_powers_;

    int step_id_;
    
  } NonIntegerStepParams;

  int num_non_integer_steps_;
  int num_non_integer_reactions_;
  int max_step_id_;
  std::vector<int> list_id_of_step_;
  std::vector<int> list_id_of_reverse_step_;
  std::map<int, std::string> reaction_names_;             // key is reaction_id
  std::map<int, int> reaction_id_of_step_;                // key is step_id
  std::map<int, ReactionDirection> reaction_dir_of_step_; // key is step_id
  std::vector<NonIntegerStepParams> params_;

  // Jacobian information
  void BuildJacobian();

  int last_jacobian_step_count_; // used to determine if Jacobian data needs
                                 // to be updated
  int num_jacobian_nonzeros_;
  // jacobian structure information
  std::vector<int> jacobian_row_id_;
  std::vector<int> jacobian_column_id_;

  // jacobian processing information
  std::vector<int> process_step_id_;
  std::vector<int> process_concentration_id_;
  std::vector<int> process_jacobian_id_;
  std::vector<double> process_multiplier_;

  bool ford_correction_derivative_;
  double ford_correction_k_;
  double ford_correction_c0_;
};

} // namespace zerork
#endif

