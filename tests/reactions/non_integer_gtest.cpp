#include <math.h>
#include <vector>
#include <cstdlib>

#include <zerork/mechanism.h>

#include <gtest/gtest.h>

// ---------------------------------------------------------------------------
// test constants
// ---------------------------------------------------------------------------
static const double OK_DOUBLE = 1.0e-13; // acceptable relative tolerance

static const char MECH_FILENAME[]  = "mechanisms/ideal/non_integer_test.mech";
static const char THERM_FILENAME[] = "mechanisms/ideal/const_specific_heat.therm";
static const char PARSER_LOGNAME[] = "parser.log";

static const int NUM_SPECIES = 5;
static const int NUM_REACTIONS = 2;
static const int NUM_STEPS = 3;

static const double LOCAL_GAS_CONST = 8.314462618e3;

static double GetEquilibriumConstReaction01(const double temperature);

static double GetRateConstant(zerork::mechanism *mech,
                              const double temperature,
                              const double pressure,
                              const std::vector<double> &mole_fractions,
                              const int reaction_id,
                              const int reaction_dir);
//static void GetUniformComposition(zerork::mechanism *mech,
//                                 std::vector<double> *composition);

static bool NearScalar(const double a,
                      const double b);
//static bool NearVector(size_t n,
//                       const double a[],
//                      const double b[]);

// ---------------------------------------------------------------------------
// test fixture for the Zero-RK mechanism 
class MechanismTestFixture: public ::testing::Test 
{ 
 public: 
  MechanismTestFixture( ) { 
    // initialization code here
    mechanism_    = NULL;
    const char * ZERORK_DATA_DIR = std::getenv("ZERORK_DATA_DIR");
    std::string load_mech(MECH_FILENAME);
    std::string load_therm(THERM_FILENAME);
    if(ZERORK_DATA_DIR == nullptr) {
      load_mech = std::string("../../") + load_mech;
      load_therm = std::string("../../") + load_therm;
    } else {
      load_mech = std::string(ZERORK_DATA_DIR) + "/" + load_mech;
      load_therm = std::string(ZERORK_DATA_DIR) + "/" + load_therm;
    }
    mechanism_ = new zerork::mechanism(load_mech.c_str(),
                                       load_therm.c_str(),
                                       PARSER_LOGNAME);
  } 

  void SetUp( ) { 
    // code here will execute just before the test ensues 
  }

  void TearDown( ) { 
    // code here will be called just after the test completes
    // ok to through exceptions from here if need be
  }

  ~MechanismTestFixture( )  { 
    // cleanup any pending stuff, but no exceptions allowed
    if(mechanism_ != NULL) {
      delete mechanism_;
    }
  }

  // put in any custom data members that you need 
  zerork::mechanism    *mechanism_;
};

TEST_F (MechanismTestFixture, Allocation)
{
  // Note assert failures halts test fixture, but not all tests
  ASSERT_TRUE(mechanism_ != NULL) << 
    "mechanism_ = new zerork::mechanism";
  
  const int num_species   = mechanism_->getNumSpecies();
  const int num_reactions = mechanism_->getNumReactions();
  const int num_steps     = mechanism_->getNumSteps();

  ASSERT_TRUE(NUM_SPECIES == num_species) <<
    "mechanism_->getNumSpecies()";

  ASSERT_TRUE(NUM_REACTIONS == num_reactions) <<
    "mechanism_->getNumReactions()";

  ASSERT_TRUE(NUM_STEPS == num_steps) <<
    "mechanism_->getNumSteps()";
}
TEST_F (MechanismTestFixture, Step01)
{
  // Note assert failures halts test fixture, but not all tests
  ASSERT_TRUE(mechanism_ != NULL) << 
    "mechanism_ = new zerork::mechanism";

  double pressure    = 1.0e6; // [Pa]
  double temperature = 850.0; // [K]
  double total_concentration = pressure/(LOCAL_GAS_CONST*temperature);
  double expected_rate;
  std::vector<double> concentrations;
  std::vector<double> net_rates, creation_rates, destruction_rates, step_rates;
  concentrations.assign(NUM_SPECIES,0.0);
  net_rates.assign(NUM_SPECIES, 0.0);
  creation_rates.assign(NUM_SPECIES, 0.0);
  destruction_rates.assign(NUM_SPECIES, 0.0);
  step_rates.assign(NUM_STEPS, 0.0);
  
  concentrations[0] = 0.0; // o
  concentrations[1] = 0.3; // o2
  concentrations[2] = 0.0; // o3
  concentrations[3] = 0.7; // co
  concentrations[4] = 0.0; // co2

  for(size_t j=0; j<concentrations.size(); ++j) {
    concentrations[j] *= total_concentration;
  }
  mechanism_->getReactionRates(temperature, 
                               &concentrations[0],
                               &net_rates[0],
                               &creation_rates[0],
                               &destruction_rates[0],
                               &step_rates[0]);
  expected_rate = 2.0e4*sqrt(0.001)*pow(total_concentration,1.5)*0.7*sqrt(0.3);
  EXPECT_TRUE(NearScalar(step_rates[0],expected_rate)) << "Reaction 1 - forward step rate";

  // check the net, creation and destruction rate
  EXPECT_TRUE(NearScalar(net_rates[0], 0.0)) << "Reaction 1 - net rate o";
  EXPECT_TRUE(NearScalar(net_rates[1], -0.5*expected_rate)) << "Reaction 1 - net rate o2";
  EXPECT_TRUE(NearScalar(net_rates[2], 0.0)) << "Reaction 1 - net rate o3";
  EXPECT_TRUE(NearScalar(net_rates[3], -expected_rate)) << "Reaction 1 - net rate co";
  EXPECT_TRUE(NearScalar(net_rates[4], expected_rate)) << "Reaction 1 - net rate co2";
  EXPECT_TRUE(NearScalar(creation_rates[0], 0.0)) << "Reaction 1 - creation rate o";
  EXPECT_TRUE(NearScalar(creation_rates[1], 0.0)) << "Reaction 1 - creation rate o2";
  EXPECT_TRUE(NearScalar(creation_rates[2], 0.0)) << "Reaction 1 - creation rate o3";
  EXPECT_TRUE(NearScalar(creation_rates[3], 0.0)) << "Reaction 1 - creation rate co";
  EXPECT_TRUE(NearScalar(creation_rates[4], expected_rate)) << "Reaction 1 - creation rate co2";
  EXPECT_TRUE(NearScalar(destruction_rates[0], 0.0)) << "Reaction 1 - destruction rate o";
  EXPECT_TRUE(NearScalar(destruction_rates[1], 0.5*expected_rate)) << "Reaction 1 - destruction rate o2";
  EXPECT_TRUE(NearScalar(destruction_rates[2], 0.0)) << "Reaction 1 - destruction rate o3";
  EXPECT_TRUE(NearScalar(destruction_rates[3], expected_rate)) << "Reaction 1 - destruction rate co";
  EXPECT_TRUE(NearScalar(destruction_rates[4], 0.0)) << "Reaction 1 - destruction rate co2";

  // check to see if the reaction can be processed when the concentration
  // is negative
  concentrations[1] = -concentrations[1];
  mechanism_->getReactionRates(temperature, 
                               &concentrations[0],
                               &net_rates[0],
                               &creation_rates[0],
                               &destruction_rates[0],
                               &step_rates[0]);
  EXPECT_TRUE(NearScalar(step_rates[0],-expected_rate)) << "Reaction 1 - forward step rate with one negative concentration";   
  // check to see if the reaction can be processed when the concentration
  // is negative, two or more negative concentrations should still have
  // a negative ROP to force the negative species to zero
  concentrations[3] = -concentrations[3];
  mechanism_->getReactionRates(temperature, 
                               &concentrations[0],
                               &net_rates[0],
                               &creation_rates[0],
                               &destruction_rates[0],
                               &step_rates[0]);
  EXPECT_TRUE(NearScalar(step_rates[0],-expected_rate)) << "Reaction 1 - forward step rate with two negative concentrations";   
}

TEST_F (MechanismTestFixture, Step02)
{
  // Note assert failures halts test fixture, but not all tests
  ASSERT_TRUE(mechanism_ != NULL) << 
    "mechanism_ = new zerork::mechanism";
  
  double pressure    = 2.0e6; // [Pa]
  double temperature = 1200.0; // [K]
  double total_concentration = pressure/(LOCAL_GAS_CONST*temperature);
  double expected_fwd_rate, expected_rev_rate;
  double K_fwd, K_rev, K_eqm_expected;
  std::vector<double> concentrations;
  std::vector<double> net_rates, creation_rates, destruction_rates, step_rates;
  concentrations.assign(NUM_SPECIES,0.0);
  net_rates.assign(NUM_SPECIES, 0.0);
  creation_rates.assign(NUM_SPECIES, 0.0);
  destruction_rates.assign(NUM_SPECIES, 0.0);
  step_rates.assign(NUM_STEPS, 0.0);
  
  concentrations[0] = 0.0; // o
  concentrations[1] = 0.2; // o2
  concentrations[2] = 0.0; // o3
  concentrations[3] = 0.3; // co
  concentrations[4] = 0.5; // co2
 
  K_eqm_expected = GetEquilibriumConstReaction01(temperature);
  K_fwd = GetRateConstant(mechanism_,
                          temperature,
                          pressure,
                          concentrations, // mole fractions
                          0,   // reaction index
                          1);  // reaction direction
  K_rev = GetRateConstant(mechanism_,
                          temperature,
                          pressure,
                          concentrations, // mole fractions
                          0,   // reaction index
                          -1); // reaction direction
   
  EXPECT_TRUE(NearScalar(K_fwd, 2.0e4*sqrt(0.001))) << "Reaction 1 - forward rate coefficient";
  EXPECT_TRUE(NearScalar(K_rev, 2.0e4*sqrt(0.001)/K_eqm_expected)) << "Reaction 1 - reverse rate coefficient";

  for(size_t j=0; j<concentrations.size(); ++j) {
    concentrations[j] *= total_concentration;
  }

  mechanism_->getReactionRates(temperature, 
                               &concentrations[0],
                               &net_rates[0],
                               &creation_rates[0],
                               &destruction_rates[0],
                               &step_rates[0]);

  expected_fwd_rate = 2.0e4*sqrt(0.001)*pow(total_concentration,1.5)*0.3*sqrt(0.2);
  EXPECT_TRUE(NearScalar(step_rates[0],expected_fwd_rate)) << "Reaction 1 - forward step rate";

  expected_rev_rate = 2.0e4*sqrt(0.001)/K_eqm_expected*total_concentration*0.5;
  EXPECT_TRUE(NearScalar(step_rates[1],expected_rev_rate)) << "Reaction 1 - reverse step rate";

    // check the net, creation and destruction rate
  EXPECT_TRUE(NearScalar(net_rates[0], 0.0)) << "Reaction 1 - net rate o";
  EXPECT_TRUE(NearScalar(net_rates[1], 0.5*(expected_rev_rate-expected_fwd_rate))) << "Reaction 1 - net rate o2";
  EXPECT_TRUE(NearScalar(net_rates[2], 0.0)) << "Reaction 1 - net rate o3";
  EXPECT_TRUE(NearScalar(net_rates[3], expected_rev_rate-expected_fwd_rate)) << "Reaction 1 - net rate co";
  EXPECT_TRUE(NearScalar(net_rates[4], expected_fwd_rate-expected_rev_rate)) << "Reaction 1 - net rate co2";
  EXPECT_TRUE(NearScalar(creation_rates[0], 0.0)) << "Reaction 1 - creation rate o";
  EXPECT_TRUE(NearScalar(creation_rates[1], 0.5*expected_rev_rate)) << "Reaction 1 - creation rate o2";
  EXPECT_TRUE(NearScalar(creation_rates[2], 0.0)) << "Reaction 1 - creation rate o3";
  EXPECT_TRUE(NearScalar(creation_rates[3], expected_rev_rate)) << "Reaction 1 - creation rate co";
  EXPECT_TRUE(NearScalar(creation_rates[4], expected_fwd_rate)) << "Reaction 1 - creation rate co2";
  EXPECT_TRUE(NearScalar(destruction_rates[0], 0.0)) << "Reaction 1 - destruction rate o";
  EXPECT_TRUE(NearScalar(destruction_rates[1], 0.5*expected_fwd_rate)) << "Reaction 1 - destruction rate o2";
  EXPECT_TRUE(NearScalar(destruction_rates[2], 0.0)) << "Reaction 1 - destruction rate o3";
  EXPECT_TRUE(NearScalar(destruction_rates[3], expected_fwd_rate)) << "Reaction 1 - destruction rate co";
  EXPECT_TRUE(NearScalar(destruction_rates[4], expected_rev_rate)) << "Reaction 1 - destruction rate co2";
}

TEST_F (MechanismTestFixture, Step03)
{
  // Note assert failures halts test fixture, but not all tests
  ASSERT_TRUE(mechanism_ != NULL) << 
    "mechanism_ = new zerork::mechanism";
  
  double pressure    = 2.0e6; // [Pa]
  double temperature = 1200.0; // [K]
  double total_concentration = pressure/(LOCAL_GAS_CONST*temperature);
  std::vector<double> concentrations;
  std::vector<double> net_rates, creation_rates, destruction_rates, step_rates;
  concentrations.assign(NUM_SPECIES,0.0);
  net_rates.assign(NUM_SPECIES, 0.0);
  creation_rates.assign(NUM_SPECIES, 0.0);
  destruction_rates.assign(NUM_SPECIES, 0.0);
  step_rates.assign(NUM_STEPS, 0.0);

  concentrations[0] = 0.1; // o
  concentrations[1] = 0.3; // o2
  concentrations[2] = 0.6; // o3
  concentrations[3] = 0.0; // co
  concentrations[4] = 0.0; // co2

  for(size_t j=0; j<concentrations.size(); ++j) {
    concentrations[j] *= total_concentration;
  }

  mechanism_->getReactionRates(temperature, 
                               &concentrations[0],
                               &net_rates[0],
                               &creation_rates[0],
                               &destruction_rates[0],
                               &step_rates[0]);
  double expected_fwd_rate = 3.0e5*total_concentration*0.6;
  EXPECT_TRUE(NearScalar(step_rates[2],expected_fwd_rate)) << "Reaction 2 - forward step rate";

  // check the net, creation and destruction rate
  EXPECT_TRUE(NearScalar(net_rates[0], 1.4*expected_fwd_rate)) << "Reaction 2 - net rate o";
  EXPECT_TRUE(NearScalar(net_rates[1], 0.8*expected_fwd_rate)) << "Reaction 2 - net rate o2";
  EXPECT_TRUE(NearScalar(net_rates[2], -expected_fwd_rate)) << "Reaction 2 - net rate o3";
  EXPECT_TRUE(NearScalar(net_rates[3], 0.0)) << "Reaction 2 - net rate co";
  EXPECT_TRUE(NearScalar(net_rates[4], 0.0)) << "Reaction 2 - net rate co2";
  EXPECT_TRUE(NearScalar(creation_rates[0], 1.4*expected_fwd_rate)) << "Reaction 2 - creation rate o";
  EXPECT_TRUE(NearScalar(creation_rates[1], 0.8*expected_fwd_rate)) << "Reaction 2 - creation rate o2";
  EXPECT_TRUE(NearScalar(creation_rates[2], 0.0)) << "Reaction 2 - creation rate o3";
  EXPECT_TRUE(NearScalar(creation_rates[3], 0.0)) << "Reaction 2 - creation rate co";
  EXPECT_TRUE(NearScalar(creation_rates[4], 0.0)) << "Reaction 2 - creation rate co2";
  EXPECT_TRUE(NearScalar(destruction_rates[0], 0.0)) << "Reaction 2 - destruction rate o";
  EXPECT_TRUE(NearScalar(destruction_rates[1], 0.0)) << "Reaction 2 - destruction rate o2";
  EXPECT_TRUE(NearScalar(destruction_rates[2], expected_fwd_rate)) << "Reaction 2 - destruction rate o3";
  EXPECT_TRUE(NearScalar(destruction_rates[3], 0.0)) << "Reaction 2 - destruction rate co";
  EXPECT_TRUE(NearScalar(destruction_rates[4], 0.0)) << "Reaction 2 - destruction rate co2";


  
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

static double GetEquilibriumConstReaction01(const double temperature)
{
  // reaction 1: co + 0.5 o2 <=> co2
  // S/Ru = a_0*log(T) + a_6
  double co_entropy   = ( 3.50000000e+00)*log(temperature) + 3.83145325e+00;
  double o2_entropy   = ( 3.50000000e+00)*log(temperature) + 4.73253404e+00;
  double co2_entropy  = ( 4.46541338e+00)*log(temperature) + 2.70309557e-01;
  // H/RuT = a_0 + a_5/T
  double co_enthalpy  = 3.50000000e+00 - 1.43372329e+04/temperature;
  double o2_enthalpy  = 3.50000000e+00 - 1.04352500e+03/temperature;
  double co2_enthalpy = 4.46541338e+00 - 4.86597535e+04/temperature;

  double delta_entropy  = -co_entropy  - 0.5*o2_entropy  + co2_entropy;
  double delta_enthalpy = -co_enthalpy - 0.5*o2_enthalpy + co2_enthalpy;

  double Keq = pow(1.01325e5/(LOCAL_GAS_CONST*temperature), -0.5)*
    exp(delta_entropy-delta_enthalpy);

  return Keq;
}

static double GetRateConstant(zerork::mechanism *mech,
                              const double temperature,
                              const double pressure,
                              const std::vector<double> &mole_fractions,
                              const int reaction_id,
                              const int reaction_dir)
{
  const double gas_constant = mech->getGasConstant();
  const int num_species = mech->getNumSpecies();
  const int num_reactions = mech->getNumReactions();
  double Krxn;

  std::vector<double> concentrations;
  std::vector<double> forward_rate_coefficients;
  std::vector<double> reverse_rate_coefficients;

  concentrations.clear();
  concentrations.assign(num_species, 0.0);
  forward_rate_coefficients.clear();
  forward_rate_coefficients.assign(num_reactions, 0.0);
  reverse_rate_coefficients.clear();
  reverse_rate_coefficients.assign(num_reactions, 0.0);

  for(int j=0; j<num_species; ++j) {
    concentrations[j] = mole_fractions[j]*pressure/(gas_constant*temperature);
  }
  mech->getKrxnFromTC(temperature,
                      &concentrations[0],
                      &forward_rate_coefficients[0],
                      &reverse_rate_coefficients[0]);
  Krxn = ((reaction_dir < 0) ? reverse_rate_coefficients[reaction_id] :
          forward_rate_coefficients[reaction_id]);

  return Krxn;

}
static void GetUniformComposition(zerork::mechanism *mech,
                                  std::vector<double> *composition)
{
  const int num_species = mech->getNumSpecies();
  composition->clear();
  composition->assign(num_species,1.0/(double)num_species);
}

// static double GetEquilibriumConst(const double temperature,
//                                   const double pressure)
// {
//   // A + A <=> A2 (diatomic)
//   // S/Ru
//   double   atomic_entropy =   ATOMIC_A0*log(temperature) +   ATOMIC_A6;
//   double diatomic_entropy = DIATOMIC_A0*log(temperature) + DIATOMIC_A6;
//   // H/RuT
//   double   atomic_enthalpy =   ATOMIC_A0 +   ATOMIC_A5/temperature;
//   double diatomic_enthalpy = DIATOMIC_A0 + DIATOMIC_A5/temperature;

//   double delta_entropy  = -2.0*atomic_entropy  + diatomic_entropy;
//   double delta_enthalpy = -2.0*atomic_enthalpy + diatomic_enthalpy;

//   double Keq = pow(1.01325e5/(LOCAL_GAS_CONST*temperature), -1.0)*
//     exp(delta_entropy-delta_enthalpy);

//   return Keq;
// }





static bool NearScalar(const double a,
                       const double b)
{
  double weight = 0.5*(fabs(a)+fabs(b));

  if(weight >= OK_DOUBLE*OK_DOUBLE) {
    // check the normalized difference
    if(fabs(a-b)/weight > OK_DOUBLE) {

      printf("#         %24.18e != %24.18e\n",a,b);
      return false;
    } 
  }
  return true;
}

// static bool NearVector(size_t n,
//                        const double a[],
//                        const double b[])
// {
//   bool near_scalar;
//   for(size_t j=0; j<n; ++j) {
//     near_scalar = NearScalar(a[j],b[j]);
//     if(near_scalar == false) {
//       return false;
//     }
//   }
//   return true;
// }
