#include <stdlib.h>
#include <stdio.h>

#include <vector>
#include <string>
#include <map>

#include "utilities.h"
#include <utilities/file_utilities.h>
#include <utilities/string_utilities.h>

#include <mpi.h>

#include "flame_params.h"
#include "kinsol_functions.h"
#include "set_initial_conditions.h"

#include "restart_writer.h"


#include <nvector/nvector_parallel.h> // serial N_Vector types, fcts., and macros
#include <kinsol/kinsol.h>

#ifdef SUNDIALS2
#include <kinsol/kinsol_spgmr.h>
#elif SUNDIALS3
#include <kinsol/kinsol_spils.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#elif SUNDIALS4
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sunnonlinsol/sunnonlinsol_newton.h>
#endif

using zerork::getHighResolutionTime;

//#include <kinsol/kinsol_bbdpre.h>
//#include <sundials/sundials_dense.h>
//#include <sundials/sundials_types.h>
//#include <sundials/sundials_math.h>


const bool RUN_DEBUG=false;
const int NUM_STDOUT_PARAMS = 8; // number of non species parameters to write
                                  // to standard out
const int NUM_LOG_PARAMS = 10;    // number of non species parameters to write
                                  // LogGridPointState
static double FindMaximumParallel(const size_t num_points,
				  const double x[],
				  const size_t x_stride,
				  const double f[],
				  const size_t f_stride,
				  const bool use_quadratic,
				  double *x_at_max);

static void WriteFieldParallel(double t,
			       const double state[],
			       const FlameParams &params);

static double GetMixtureMolecularMass(const int grid_id,
                                      const double t,
                                      const double state[],
                                      const FlameParams &params);
static double minSumMassFractions(const double state[],
                                  const FlameParams &params);
static double maxSumMassFractions(const double state[],
                                  const FlameParams &params);

static int GetFuelSpeciesId(const FlameParams &params,
                            std::vector<int> *species_id);

static double StoichiometricMixtureFraction(const FlameParams &params);

static int GetTrackMaxStateId(const FlameParams &params,
                              std::vector<int> *state_id);
static int GetStateMaxima(const std::vector<int> &state_id,
                          const double state[],
                          const FlameParams &params,
                          std::vector<double> *max_value,
                          std::vector<double> *max_position);

typedef struct
{
  int rxnId;
  double relSens;
  double relSensAbs;
} rxnSens_t;

int compare_rxnSens_t(const void *A, const void *B);


int main(int argc, char *argv[])
{
  double clock_time = getHighResolutionTime();
  double setup_time, loop_time, sensanal_time, uq_time;

  if(argc < 2) {
    printf("# ERROR: Incorrect command line usage.\n");
    printf("#        use %s <input parameters>\n",argv[0]);
    exit(-1);
  }

  // MPI Stuff
  MPI_Comm comm;
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;

  // KINSOL memory pointer and linear solver
  void *kinsol_ptr = NULL;
#if defined SUNDIALS3 || defined SUNDIALS4
  SUNLinearSolver LS;
  LS = NULL;
#endif

  // Initialize flame params
  FlameParams flame_params(argv[1],comm);

  // Initialize state variables/scalers vectors and pointers
  N_Vector flame_state, scaler, constraints;
  flame_state = NULL;
  scaler = constraints = NULL;
  double *flame_state_ptr, *scaler_ptr;
  flame_state_ptr = NULL;
  scaler_ptr = NULL;

  // Declare variables
  int flag=0;
  realtype dq_rel_uu;
  int mudq, mldq, mukeep, mlkeep;
  int Nlocal;
  int maxl, maxlrst;
  int maxiter, mset;

  // Get constant from flame_params
  int my_pe = flame_params.my_pe_;
  const int num_grid_points = flame_params.z_.size();
  const int num_local_points = flame_params.num_local_points_;
  const int num_states = flame_params.reactor_->GetNumStates();
  const int num_species = flame_params.reactor_->GetNumSpecies();
  const int num_steps = flame_params.reactor_->GetNumSteps();
  const int num_reactions = flame_params.reactor_->GetNumReactions();
  long int num_local_states = num_local_points*num_states;
  long int total_states = num_grid_points*num_states;
  const double ref_temperature = flame_params.ref_temperature_;
  int num_prints = 0;
  double current_time = 0.0;
  double time_offset;
  double fuel_density, fuel_molecular_mass;
  double stoichiometric_mixture_fraction;
  double min_sum_mass_fraction, max_sum_mass_fraction;

  Nlocal = num_local_states;

  // KINSOL integrator Stats
  long int nsteps, nfevals, nliniters, njacsetups, njacsolves;

  // Tracking
  std::vector<double> temperature_jump;
  std::vector<int> track_max_state_id;
  std::vector<double> state_maxima, state_maxima_positions;


  // allocate KINSOL data structures
  flame_state          = N_VNew_Parallel(comm, num_local_states, total_states);
  flame_state_ptr      = NV_DATA_P(flame_state);
  scaler          = N_VNew_Parallel(comm, Nlocal, total_states);
  scaler_ptr      = NV_DATA_P(scaler);
  constraints     = N_VNew_Parallel(comm, Nlocal, total_states);
  N_VConst(0.0, constraints);// 0 for no constraint, 1.0 for >= 0.0, 2.0 for > 0.0

  // Initialize state vector
  time_offset = 0.0;
  SetInitialCompositionAndWallTemp(flame_params, flame_state_ptr, &time_offset);

  // Set scaling to 1
  for (int j=0; j<num_local_points; ++j) {
    for (int k=0; k<num_states; ++k){
      scaler_ptr[j*num_states + k] = 1.0;
    }
  }

  // Initialize soot
  if(flame_params.soot_)
    InitializePAH(&flame_params);

  temperature_jump.assign(num_local_points,0.0);
  GetTrackMaxStateId(flame_params, &track_max_state_id);

  //----------------------------------------------------------------------------
  // Setup KINSOL solver
  // Create pointer
  kinsol_ptr = KINCreate();

  // Initialize KINSOL module with RHS function and state vector
  flag = KINInit(kinsol_ptr, ConstPressureFlame, flame_state);

  // Set function to handle errors and exit cleanly
  flag = KINSetErrHandlerFn(kinsol_ptr, ErrorFunction, &flame_params);

  // Set user data
  flag = KINSetUserData(kinsol_ptr, &flame_params);

  // Here would be KINSetConstraints but since we have none we just destroy them
  N_VDestroy_Parallel(constraints);

  // Set tolerances
  // RHS(y) < fnormtol
  flag = KINSetFuncNormTol(kinsol_ptr, flame_params.parser_->rel_tol());
  // Step tolerance: dy < steptol
  flag = KINSetScaledStepTol(kinsol_ptr, flame_params.parser_->abs_tol());

  // max Krylov dimension
  maxl = flame_params.krylov_subspace_;
  maxlrst = 2;

#ifdef SUNDIALS2
  // Initialize Linear Solver
  flag = KINSpgmr(kinsol_ptr, maxl);
  flag = KINSpilsSetMaxRestarts(kinsol_ptr, maxlrst);
  if(flame_params.parser_->integrator_type() == 2) {
    // Option 2 is user-defined dense block-tridiagonal preconditioner
    flag = KINSpilsSetPreconditioner(kinsol_ptr, ReactorBBDSetup, ReactorBBDSolve);
  } else if(flame_params.parser_->integrator_type() == 3) {
    // Option 3 is user-defined approximate sparse preconditioner
    flag = KINSpilsSetPreconditioner(kinsol_ptr, ReactorAFSetup, ReactorAFSolve);
  } else  {
    printf("integrator_type == %d not supported\n",
           flame_params.parser_->integrator_type());
    flag = -1;
  }
#elif SUNDIALS3
  // Initialize Linear Solver
  LS = SUNSPGMR(flame_state, PREC_RIGHT, maxl);
  flag = KINSpilsSetLinearSolver(kinsol_ptr, LS);
  // Set max restarts
  flag = SUNSPGMRSetMaxRestarts(LS, maxlrst);

  if(flame_params.parser_->integrator_type() == 2) {
    // Option 2 is user-defined dense block-tridiagonal preconditioner
    flag = KINSpilsSetPreconditioner(kinsol_ptr, ReactorBBDSetup, ReactorBBDSolve);
  } else if(flame_params.parser_->integrator_type() == 3) {
    // Option 3 is user-defined approximate sparse preconditioner
    flag = KINSpilsSetPreconditioner(kinsol_ptr, ReactorAFSetup, ReactorAFSolve);
  } else  {
    printf("integrator_type == %d not supported\n",
           flame_params.parser_->integrator_type());
    flag = -1;
  }
#elif SUNDIALS4
  // Initialize Linear Solver
  LS = SUNLinSol_SPGMR(flame_state, PREC_RIGHT, maxl);
  flag = KINSetLinearSolver(kinsol_ptr, LS, NULL);
  flag = SUNLinSol_SPGMRSetMaxRestarts(LS, maxlrst);
    if(flame_params.parser_->integrator_type() == 2) {
    // Option 2 is user-defined dense block-tridiagonal preconditioner
    flag = KINSetPreconditioner(kinsol_ptr, ReactorBBDSetup, ReactorBBDSolve);
  } else if(flame_params.parser_->integrator_type() == 3) {
    // Option 3 is user-defined approximate sparse preconditioner
    flag = KINSetPreconditioner(kinsol_ptr, ReactorAFSetup, ReactorAFSolve);
  } else  {
    printf("integrator_type == %d not supported\n",
           flame_params.parser_->integrator_type());
    flag = -1;
  }
#endif

  maxiter = flame_params.parser_->max_iter();
  flag = KINSetNumMaxIters(kinsol_ptr, maxiter);

  // Set maximum number of Newton iterations
  //0 for default, 1 for exact Newton, > 1 for modified Newton
  mset = flame_params.parser_->max_subiter();
  flag = KINSetMaxSetupCalls(kinsol_ptr, mset);

  // Set print level
  //Print: 0 for no info, 1 for scaled l2 norm, 3 for additional linear solver info
  flag = KINSetPrintLevel(kinsol_ptr, 1);


  // Write data file at time 0
  if (time_offset == 0.0) {
    WriteFieldParallel(time_offset,
		       &flame_state_ptr[0],
		       flame_params);
  }

  // Report simulation info
  if(my_pe == 0) {
    fuel_molecular_mass = GetMixtureMolecularMass(
	   0,
	   0.0,
	   &flame_params.fuel_mass_fractions_[0],
	   flame_params);
    fuel_density = flame_params.parser_->pressure()*fuel_molecular_mass/
      (flame_params.fuel_temperature_*flame_params.ref_temperature_*
       flame_params.reactor_->GetGasConstant());
    stoichiometric_mixture_fraction = StoichiometricMixtureFraction(flame_params);
    printf("# Number of states     : %d\n",num_states);
    printf("# Number of grid points: %d\n", num_grid_points);
    printf("# Stoichiometric Mixture fraction: %.18g\n",
	   stoichiometric_mixture_fraction);
    printf("# Fuel temperature  [K]: %.18g\n",
	   flame_params.fuel_temperature_*flame_params.ref_temperature_);
    printf("# Fuel density [kg/m^3]: %.18g\n", fuel_density);
    printf("# Initial upstream temperature  (next to inlet)     [K]: %.18g\n",
	   flame_state_ptr[num_states-1]*ref_temperature);
    printf("# Initial upstream density      (next to inlet)[kg/m^3]: %.18g\n",
	   1.0/flame_state_ptr[num_states-2]);
    printf("#------------------------------------------------------------------------------\n");
    printf("# Column  1: [#] time steps printed\n");
    printf("# Column  2: [s] ODE system time\n");
    printf("# Column  3: [#] number of steps taken by KINSOL\n");
    printf("# Column  4: [#] number of calls to the user's (RHS) function\n");
    printf("# Column  5: [#] number of calls to the linear solver setup function\n");
    printf("# Column  6: [#] number of linear iterations\n");
    printf("# Column  7: min sum of mass fractions\n");
    printf("# Column  8: max sum of mass fractions\n");
    for(int j=0; j<(int)track_max_state_id.size(); ++j) {
      printf("# Column %2d: maximum %s in the domain\n",
	     NUM_STDOUT_PARAMS+1+2*j,
	     flame_params.reactor_->GetNameOfStateId(track_max_state_id[j]));
      printf("# Column %2d: [m] location of the maximum %s\n",
	     NUM_STDOUT_PARAMS+2+2*j,
	     flame_params.reactor_->GetNameOfStateId(track_max_state_id[j]));
    }
  }// if(my_pe==0)

  // Get time
  setup_time = getHighResolutionTime() - clock_time;
  clock_time = getHighResolutionTime();

  double fnorm, stepnorm;
  // Pseudo-unsteady
  if(flame_params.pseudo_unsteady_) {
    flag = KINSetPrintLevel(kinsol_ptr, 0);
    flag = KINSetNumMaxIters(kinsol_ptr, 50);
    if(my_pe==0) {
      printf("# begin pseudo time-stepping\n");
      printf("# time      timestep     maxT    nfevals  cputime\n");
    }
    double pseudo_time = 0.0;
    double kinstart_time, kinend_time;
    double max_temperature_jump, z_max_temperature_jump;
    flame_params.dt_ = 1.0e-3; //seems to be a good starting guess

    while(pseudo_time < 0.05) {
      for(int j=0; j<num_local_states; j++) {
        flame_params.y_old_[j] = flame_state_ptr[j];
      }

      // Solve system
      kinstart_time = getHighResolutionTime();
      flag = KINSol(kinsol_ptr,
                    flame_state,
                    KIN_NONE,
                    scaler,
                    scaler);
      kinend_time = getHighResolutionTime();

      KINGetNumFuncEvals(kinsol_ptr,&nfevals);
      KINGetFuncNorm(kinsol_ptr, &fnorm);

      if((flag==0 || flag==1) && fnorm != 0.0){
        // Success
        // Compute T
        for(int j=0; j<num_local_points; ++j) {
          temperature_jump[j] =
            flame_state_ptr[j*num_states+num_species+1]*flame_params.ref_temperature_;
        }
        // Find maximum T-Twall and its location
        max_temperature_jump = FindMaximumParallel(num_local_points,
                                                   &flame_params.z_[0],
                                                   1,
                                                   &temperature_jump[0],
                                                   1,
                                                   true, // use quadratic
                                                   &z_max_temperature_jump);

        pseudo_time += flame_params.dt_;
        if(my_pe==0) {printf("%5.3e   %5.3e   %5.3e  %d   %5.3e\n",
                             pseudo_time,
                             flame_params.dt_,
                             max_temperature_jump,
                             nfevals,
                             kinend_time-kinstart_time);}

        // Increase time step if it's converging quickly
        if(nfevals < 6)
          flame_params.dt_ *= 2.0;

      } else {
        // Failure, reset state and decrease timestep
        for(int j=0; j<num_local_states; j++) {
          flame_state_ptr[j] = flame_params.y_old_[j];
        }
        flame_params.dt_ *= 0.5;
        if(flame_params.dt_ < 1.0e-5){
          if(my_pe==0) {printf("# Error, pseudo-unsteady solver is not converging\n");}
          exit(-1);
        }
        if(my_pe==0){printf("# time step failed, decreasing time step to: %5.3e\n", flame_params.dt_);}
      }
    }
    if(my_pe==0){printf("# end pseudo time-stepping\n");}
  }


  // Solve system
  flame_params.pseudo_unsteady_ = false;
  flag = KINSetPrintLevel(kinsol_ptr, 1);
  flag = KINSol(kinsol_ptr,
                flame_state,
                KIN_NONE, // basic Newton
                scaler,
                scaler);
  KINGetFuncNorm(kinsol_ptr, &fnorm);

  if((flag==0 || flag==1) && fnorm != 0) {
    // Get KINSOL stats
    flag = KINGetNumFuncEvals(kinsol_ptr,&nfevals);
    flag = KINGetNumNonlinSolvIters(kinsol_ptr, &nsteps);
#if defined SUNDIALS2 || defined SUNDIALS3
    flag = KINSpilsGetNumPrecEvals(kinsol_ptr,&njacsetups);
    flag = KINSpilsGetNumPrecSolves(kinsol_ptr,&njacsolves);
    flag = KINSpilsGetNumLinIters(kinsol_ptr,&nliniters);
#elif defined SUNDIALS4
    flag = KINGetNumPrecEvals(kinsol_ptr,&njacsetups);
    flag = KINGetNumPrecSolves(kinsol_ptr,&njacsolves);
    flag = KINGetNumLinIters(kinsol_ptr,&nliniters);
#endif
    flag = KINGetFuncNorm(kinsol_ptr, &fnorm);
    flag = KINGetStepLength(kinsol_ptr, &stepnorm);

    if (my_pe==0) {
      printf("Scaled norm of F: %14.7e\n",fnorm);
      printf("Scaled norm of the step: %14.7e\n", stepnorm);
      printf("Number of function evaluations: %ld\n", nfevals);
      printf("Number of nonlinear iterations: %ld\n", nsteps);
      printf("Number of linear iterations: %ld\n", nliniters);
      printf("Number of preconditioner evaluations: %ld\n", njacsetups);
      printf("Number of preconditioner solves: %ld\n", njacsolves);
    }

    // Get min/max of sum(Y_i)
    min_sum_mass_fraction = minSumMassFractions(flame_state_ptr,flame_params);
    max_sum_mass_fraction = maxSumMassFractions(flame_state_ptr,flame_params);

    // Get max of tracked variables
    GetStateMaxima(track_max_state_id,
                   flame_state_ptr,
                   flame_params,
                   &state_maxima,
                   &state_maxima_positions);

    if(my_pe == 0) { //only root prints
      printf("Final  %14.7e  %6d  %6d  %6d  %6d  %14.7e  %14.7e",
	       current_time,
	       (int)nsteps,
	       (int)nfevals,
	       (int)njacsetups,
	       (int)nliniters,
	       min_sum_mass_fraction,
	       max_sum_mass_fraction);
	for(size_t j=0; j<track_max_state_id.size(); ++j) {
	  printf("  %14.7e  %14.7e",state_maxima[j], state_maxima_positions[j]);
	}
	printf("\n");
      } // if(my_pe==0)

      if(num_prints % flame_params.parser_->field_dt_multiplier() == 0) {
	WriteFieldParallel(1.0,
			   &flame_state_ptr[0],
			   flame_params);
      }
  }// if(flag==0)

  // For error cases
  if(flag<0) {
    if(my_pe == 0) { //only root prints
      printf("Final  0  0  0\n");
    }
  }

  loop_time = getHighResolutionTime() - clock_time;

  // Clear before sens analysis?
  KINFree(&kinsol_ptr);
#if defined SUNDIALS3 || defined SUNDIALS4
  SUNLinSolFree(LS);
#endif

  // Write soot
  if (flame_params.soot_)
    WriteDimerProdRate(&flame_params,&flame_state_ptr[0]);

  /*------------------------------------------------------------------------*/
  if (flame_params.soot_ && flame_params.sensitivity_analysis_) {
    // Perform sensitivity analysis
    if(my_pe==0) printf("Performing soot sensitivity analysis\n");

    clock_time = getHighResolutionTime();
    double multiplier = flame_params.parser_->sensitivity_multiplier();
    rxnSens_t *rxnSensList;
    rxnSensList = new rxnSens_t[num_reactions];

    // 1) Save current/original flame state
    std::vector<double> flame_state_orig;
    flame_state_orig.assign(num_local_states, 0.0); // DUMB
    for(int j=0; j<num_local_states; j++)
      flame_state_orig[j] = flame_state_ptr[j];

    std::vector<double> dimer_prod_rate;
    dimer_prod_rate.assign(num_local_points, 0.0);
    double dimer_prod_rate_local[3];
    double max_dimer_orig, max_dimer_orig_position, max_dimer_position;
    double sum_dimer_orig = 0.0;
    double local_sum = 0.0;
    const int nover = flame_params.nover_;
    for(int j=0; j<num_local_points; j++) {
      ComputeDimerProdRate(&flame_params,
                           &flame_state_ptr[j*num_states],
                           &dimer_prod_rate_local[0]);
      dimer_prod_rate[j] = dimer_prod_rate_local[0];
    }
    max_dimer_orig = FindMaximumParallel(num_local_points,
                                         &flame_params.z_[0], 1,
                                         &dimer_prod_rate[0], 1,
                                         true, &max_dimer_orig_position);

    // Compute Y_soot = int(omega_nucl/rhou_z dZ)
    local_sum = 0.0;
    for(int j=0; j<num_local_points; ++j) {
      int jext = j + nover;
      local_sum += dimer_prod_rate[j]*flame_state_ptr[j*num_states + num_species]/
        flame_params.convection_velocity_[jext]*flame_params.dzm_local_[jext];
    }
    MPI_Allreduce(&local_sum,&sum_dimer_orig,1,PVEC_REAL_MPI_TYPE,MPI_SUM,comm);

    // 2) Loop over reactions
    for(int k=0; k<num_reactions; k++) {
      // 2.1) Perturb A factor
      int fwdId = flame_params.mechanism_->getStepIdxOfRxn(k,1);
      int revId = flame_params.mechanism_->getStepIdxOfRxn(k,-1);

      flame_params.reactor_->SetAMultiplierOfStepId(fwdId, multiplier);
      if(revId >= 0 && revId < num_steps)
        flame_params.reactor_->SetAMultiplierOfStepId(revId, multiplier);

      // 2.2) Compute solution
      kinsol_ptr = KINCreate();
      flag = KINInit(kinsol_ptr, ConstPressureFlame, flame_state);
      flag = KINSetErrHandlerFn(kinsol_ptr, ErrorFunction, &flame_params);
      flag = KINSetUserData(kinsol_ptr, &flame_params);
      flag = KINSetFuncNormTol(kinsol_ptr, flame_params.parser_->rel_tol());
      flag = KINSetScaledStepTol(kinsol_ptr, flame_params.parser_->abs_tol());
#ifdef SUNDIALS2
      flag = KINSpgmr(kinsol_ptr, maxl);
      flag = KINSpilsSetMaxRestarts(kinsol_ptr, maxlrst);
      if(flame_params.parser_->integrator_type() == 2) {
        flag = KINSpilsSetPreconditioner(kinsol_ptr, ReactorBBDSetup, ReactorBBDSolve);
      } else if(flame_params.parser_->integrator_type() == 3) {
          flag = KINSpilsSetPreconditioner(kinsol_ptr, ReactorAFSetup, ReactorAFSolve);
      } else  {
        printf("integrator_type == %d not supported\n",
               flame_params.parser_->integrator_type());
        flag = -1;
      }
#elif SUNDIALS3
      LS = SUNSPGMR(flame_state, PREC_RIGHT, maxl);
      flag = KINSpilsSetLinearSolver(kinsol_ptr, LS);
      flag = SUNSPGMRSetMaxRestarts(LS, maxlrst);
      if(flame_params.parser_->integrator_type() == 2) {
        flag = KINSpilsSetPreconditioner(kinsol_ptr, ReactorBBDSetup, ReactorBBDSolve);
      } else if(flame_params.parser_->integrator_type() == 3) {
          flag = KINSpilsSetPreconditioner(kinsol_ptr, ReactorAFSetup, ReactorAFSolve);
      } else  {
        printf("integrator_type == %d not supported\n",
               flame_params.parser_->integrator_type());
        flag = -1;
      }
#elif SUNDIALS4
      LS = SUNLinSol_SPGMR(flame_state, PREC_RIGHT, maxl);
      flag = KINSetLinearSolver(kinsol_ptr, LS, NULL);
      flag = SUNLinSol_SPGMRSetMaxRestarts(LS, maxlrst);
      if(flame_params.parser_->integrator_type() == 2) {
        flag = KINSetPreconditioner(kinsol_ptr, ReactorBBDSetup, ReactorBBDSolve);
      } else if(flame_params.parser_->integrator_type() == 3) {
          flag = KINSetPreconditioner(kinsol_ptr, ReactorAFSetup, ReactorAFSolve);
      } else  {
        printf("integrator_type == %d not supported\n",
               flame_params.parser_->integrator_type());
        flag = -1;
      }
#endif
      flag = KINSetNumMaxIters(kinsol_ptr, maxiter);
      flag = KINSetMaxSetupCalls(kinsol_ptr, mset);
      flag = KINSetPrintLevel(kinsol_ptr, 0);

      flag = KINSol(kinsol_ptr,
                    flame_state,
                    KIN_NONE,
                    scaler,
                    scaler);

      KINFree(&kinsol_ptr);
#if defined SUNDIALS3 || defined SUNDIALS4
      SUNLinSolFree(LS);
#endif
      // 2.3) Compute max dimer prod rate
      for(int j=0; j<num_local_points; j++) {
        ComputeDimerProdRate(&flame_params,
                             &flame_state_ptr[j*num_states],
                             &dimer_prod_rate_local[0]);
        dimer_prod_rate[j] = dimer_prod_rate_local[0];
      }

      double max_dimer = FindMaximumParallel(num_local_points,
                                             &flame_params.z_[0], 1,
                                             &dimer_prod_rate[0], 1,
                                             true, &max_dimer_position);

      // Compute Y_soot = int(omega_nucl/rhou_z dZ)
      double sum_dimer = 0.0;
      local_sum = 0.0;
      for(int j=0; j<num_local_points; ++j) {
        int jext = j + nover;
        local_sum += dimer_prod_rate[j]*flame_state_ptr[j*num_states + num_species]/
          flame_params.convection_velocity_[jext]*flame_params.dzm_local_[jext];
      }
      MPI_Allreduce(&local_sum,&sum_dimer,1,PVEC_REAL_MPI_TYPE,MPI_SUM,comm);

      // 2.3) Compute sensitivity coefficient
      int reacId = k;
      rxnSensList[reacId].rxnId = k;
      rxnSensList[reacId].relSens = (sum_dimer - sum_dimer_orig)/sum_dimer_orig;
      rxnSensList[reacId].relSensAbs = fabs(sum_dimer - sum_dimer_orig)/fabs(sum_dimer_orig);
      if(my_pe==0) printf("reaction %d / %d, %s, S: %14.7e\n",
                          k, num_reactions,
                          flame_params.mechanism_->getReactionName(k),
                          rxnSensList[reacId].relSens);

      // 2.4) Set A and solution back to original
      flame_params.reactor_->SetAMultiplierOfStepId(fwdId, 1.0);
      if(revId >= 0 && revId < num_steps)
        flame_params.reactor_->SetAMultiplierOfStepId(revId, 1.0);

      for(int j=0; j<num_local_states; j++)
        flame_state_ptr[j] = flame_state_orig[j];

    } // for k<num_reactions

    // Sort sensitivity coefficients in descending order
    qsort((void *)&rxnSensList[0], num_reactions, sizeof(rxnSens_t), compare_rxnSens_t);

    if(my_pe == 0) {
      FILE * sensFile;
      sensFile = fopen("sensAnal","w");
      fprintf(sensFile,"#reac_id  reaction         Srel\n");
      for(int j=0; j<num_reactions; j++) {
        fprintf(sensFile,"%d  %s  %14.7e\n",
                rxnSensList[j].rxnId,
                flame_params.mechanism_->getReactionName(rxnSensList[j].rxnId),
                rxnSensList[j].relSens);
      }
      fclose(sensFile);
    }
    sensanal_time = getHighResolutionTime() - clock_time;

  } // if soot && sensanal
  /*------------------------------------------------------------------------*/

  if (flame_params.soot_ && flame_params.uncertainty_quantification_) {
    // Uncertainty quantification
    if(my_pe==0) printf("Performing uncertainty quantification\n");
    clock_time = getHighResolutionTime();

    double dimer_prod_rate_local[3];

    // Compute dimer prod rate on doped flame
    std::vector<double> dimer_prod_rate_doped;
    dimer_prod_rate_doped.assign(num_local_points, 0.0);
    for(int j=0; j<num_local_points; j++) {
      ComputeDimerProdRate(&flame_params,
                           &flame_state_ptr[j*num_states],
                           &dimer_prod_rate_local[0]);
      dimer_prod_rate_doped[j] = dimer_prod_rate_local[0];
    }

    // Save original state vectors for doped flames
    std::vector<double> flame_state_doped_orig;
    flame_state_doped_orig.assign(num_local_states, 0.0);
    for(int j=0; j<num_local_states; j++) {
      flame_state_doped_orig[j] = flame_state_ptr[j];
    }

    // Randomly perturb reactions
    int num_random = flame_params.parser_->num_random_samples();
    std::vector<int> uncertain_reactions;
    int num_uncertain_reactions;
    uncertain_reactions = flame_params.parser_->uncertain_reactions();
    num_uncertain_reactions = uncertain_reactions.size();
    double uncertainty_factor;
    uncertainty_factor = flame_params.parser_->uncertainty_factor();

    std::vector<double> dimer_error_doped, dimer_error_BL, dimer_error_diff;
    dimer_error_doped.assign(num_local_points*num_random, 0.0);
    dimer_error_BL.assign(num_local_points*num_random, 0.0);
    dimer_error_diff.assign(num_local_points*num_random, 0.0);

    std::vector<double> dimer_mean;
    dimer_mean.assign(num_local_points, 0.0);

    std::vector<double> dimer_doped;
    dimer_doped.assign(num_local_points*num_random, 0.0);

    for(int l=0; l<num_random; ++l) {
      if(my_pe==0) printf("Random iteration %d of %d\n", l+1, num_random);
      // Perturb reactions
      double rand_multiplier, r;
      for(int k=0; k<num_uncertain_reactions; k++) {
        int rxnId = uncertain_reactions[k];
        int fwdId = flame_params.mechanism_->getStepIdxOfRxn(rxnId,1);
        int revId = flame_params.mechanism_->getStepIdxOfRxn(rxnId,-1);

        r = -1.0 + 2.0* ((double) rand() / (RAND_MAX)); // number -1<r<1

        rand_multiplier = pow(uncertainty_factor, r);

        flame_params.reactor_->SetAMultiplierOfStepId(fwdId, rand_multiplier);
        if(revId >= 0 && revId < num_steps)
          flame_params.reactor_->SetAMultiplierOfStepId(revId, rand_multiplier);
      }

      // Compute solution and dimer production rate on doped flame
      // Set state vector to unperturbed solution
      for(int j=0; j<num_local_states; j++) {
        flame_state_ptr[j] = flame_state_doped_orig[j];}
      flame_params.SetInlet();

      if(my_pe==0) printf("Computing doped flame\n");
      kinsol_ptr = KINCreate();
      flag = KINInit(kinsol_ptr, ConstPressureFlame, flame_state);
      flag = KINSetErrHandlerFn(kinsol_ptr, ErrorFunction, &flame_params);
      flag = KINSetUserData(kinsol_ptr, &flame_params);
      flag = KINSetFuncNormTol(kinsol_ptr, flame_params.parser_->rel_tol());
      flag = KINSetScaledStepTol(kinsol_ptr, flame_params.parser_->abs_tol());
#ifdef SUNDIALS2
      flag = KINSpgmr(kinsol_ptr, maxl);
      flag = KINSpilsSetMaxRestarts(kinsol_ptr, maxlrst);
      if(flame_params.parser_->integrator_type() == 2) {
        flag = KINSpilsSetPreconditioner(kinsol_ptr, ReactorBBDSetup, ReactorBBDSolve);
      } else if(flame_params.parser_->integrator_type() == 3) {
          flag = KINSpilsSetPreconditioner(kinsol_ptr, ReactorAFSetup, ReactorAFSolve);
      } else  {
        printf("integrator_type == %d not supported\n",
               flame_params.parser_->integrator_type());
        flag = -1;
      }
#elif SUNDIALS3
      LS = SUNSPGMR(flame_state, PREC_RIGHT, maxl);
      flag = KINSpilsSetLinearSolver(kinsol_ptr, LS);
      flag = SUNSPGMRSetMaxRestarts(LS, maxlrst);
      if(flame_params.parser_->integrator_type() == 2) {
        flag = KINSpilsSetPreconditioner(kinsol_ptr, ReactorBBDSetup, ReactorBBDSolve);
      } else if(flame_params.parser_->integrator_type() == 3) {
          flag = KINSpilsSetPreconditioner(kinsol_ptr, ReactorAFSetup, ReactorAFSolve);
      } else  {
        printf("integrator_type == %d not supported\n",
               flame_params.parser_->integrator_type());
        flag = -1;
      }
#elif SUNDIALS4
      LS = SUNLinSol_SPGMR(flame_state, PREC_RIGHT, maxl);
      flag = KINSetLinearSolver(kinsol_ptr, LS, NULL);
      flag = SUNLinSol_SPGMRSetMaxRestarts(LS, maxlrst);
      if(flame_params.parser_->integrator_type() == 2) {
        flag = KINSetPreconditioner(kinsol_ptr, ReactorBBDSetup, ReactorBBDSolve);
      } else if(flame_params.parser_->integrator_type() == 3) {
          flag = KINSetPreconditioner(kinsol_ptr, ReactorAFSetup, ReactorAFSolve);
      } else  {
        printf("integrator_type == %d not supported\n",
               flame_params.parser_->integrator_type());
        flag = -1;
      }
#endif
      flag = KINSetNumMaxIters(kinsol_ptr, maxiter);
      flag = KINSetMaxSetupCalls(kinsol_ptr, mset);
      flag = KINSetPrintLevel(kinsol_ptr, 1);

      flag = KINSol(kinsol_ptr,
                    flame_state,
                    KIN_NONE,
                    scaler,
                    scaler);

      KINFree(&kinsol_ptr);
#if defined SUNDIALS3 || defined SUNDIALS4
      SUNLinSolFree(LS);
#endif
      for(int j=0; j<num_local_points; j++) {
        ComputeDimerProdRate(&flame_params,
                             &flame_state_ptr[j*num_states],
                             &dimer_prod_rate_local[0]);
        dimer_error_doped[j*num_random + l] = fabs(dimer_prod_rate_local[0] -
                                                   dimer_prod_rate_doped[j]);
        dimer_mean[j] += dimer_prod_rate_local[0];
        dimer_doped[j*num_random + l] = dimer_prod_rate_local[0];
      }

      // Could compute error difference between doped and baseline flames
      // but here only looking at doped flame
      for(int j=0; j<num_local_points; j++) {
        dimer_error_diff[j*num_random + l] = dimer_error_doped[j*num_random+l];
      }

      // Reset rates to their original values
      for(int k=0; k<num_uncertain_reactions; k++) {
        int rxnId = uncertain_reactions[k];
        int fwdId = flame_params.mechanism_->getStepIdxOfRxn(rxnId,1);
        int revId = flame_params.mechanism_->getStepIdxOfRxn(rxnId,-1);

        flame_params.reactor_->SetAMultiplierOfStepId(fwdId, 1.0);
        if(revId >= 0 && revId < num_steps)
          flame_params.reactor_->SetAMultiplierOfStepId(revId, 1.0);
      }

    } // for l<num_random

    // reset state vector
    for(int j=0; j<num_local_states; j++)
      flame_state_ptr[j] = flame_state_doped_orig[j];

    // Compute statistics and write to file
    if(my_pe==0) printf("Computing dimer statistics\n");
    std::vector<double> dimer_error_diff_all;
    dimer_error_diff_all.assign(num_grid_points*num_random, 0.0);
    long int dsize = num_local_points*num_random;
    int nodeDest = 0;

    MPI_Gather(&dimer_error_diff[0],
               dsize,
               PVEC_REAL_MPI_TYPE,
               &dimer_error_diff_all[0],
               dsize,
               PVEC_REAL_MPI_TYPE,
               nodeDest,
               comm);

    std::vector<double> dimer_doped_all;
    dimer_doped_all.assign(num_grid_points*num_random, 0.0);
    MPI_Gather(&dimer_doped[0],
               dsize,
               PVEC_REAL_MPI_TYPE,
               &dimer_doped_all[0],
               dsize,
	       PVEC_REAL_MPI_TYPE,
               nodeDest,
               comm);

    std::vector<double> dimer_mean_all;
    dimer_mean_all.assign(num_grid_points, 0.0);
    dsize = num_local_points;
    MPI_Gather(&dimer_mean[0],
               dsize,
               PVEC_REAL_MPI_TYPE,
               &dimer_mean_all[0],
               dsize,
               PVEC_REAL_MPI_TYPE,
               nodeDest,
               comm);

    // Print UQ results
    if(my_pe == 0) {
      FILE * dimerFile;
      dimerFile = fopen("dimerError","w");
      fprintf(dimerFile, "#Z meanDimerErr stdDimerErr minDimerErr maxDimerErr meanDimer stdDimer\n");

      for(int j=0; j<num_grid_points; ++j) {
        double meanDimerErr = 0.0;
        double stdDimerErr = 0.0;
        double maxDimerErr = -10000000.0;
        double minDimerErr =  10000000.0;
        double stdDimer = 0.0;
        for(int l=0; l<num_random; ++l) {
          meanDimerErr += dimer_error_diff_all[j*num_random + l];
          // Get min/max
          if(dimer_error_diff_all[j*num_random + l] < minDimerErr)
            minDimerErr = dimer_error_diff_all[j*num_random + l];
          if(dimer_error_diff_all[j*num_random + l] > maxDimerErr)
            maxDimerErr = dimer_error_diff_all[j*num_random + l];
        }
        meanDimerErr /= (double) num_random;
        dimer_mean_all[j] /= (double) num_random;

        for(int l=0; l<num_random; ++l) {
          stdDimerErr += pow(dimer_error_diff_all[j*num_random + l] - meanDimerErr, 2.0);
          stdDimer += pow(dimer_doped_all[j*num_random + l] - dimer_mean_all[j], 2.0);
        }
        stdDimerErr = sqrt(stdDimerErr/(double) num_random);
        stdDimer = sqrt(stdDimer/(double) num_random);

        fprintf(dimerFile,"%12.6e  %12.6e  %12.6e  %12.6e  %12.6e  %12.6e  %12.6e\n",
                flame_params.z_[j],
                meanDimerErr, stdDimerErr, minDimerErr, maxDimerErr,
                dimer_mean_all[j], stdDimer);
      }
      fclose(dimerFile);

    } //my_pe ==0
    uq_time = getHighResolutionTime() - clock_time;

  } // if uncertainty_quantification
  /*------------------------------------------------------------------------*/

  N_VDestroy_Parallel(flame_state);
  N_VDestroy_Parallel(scaler);

  if(my_pe == 0) {
    printf("# Simulation setup time   [s]: %12.5e\n",setup_time);
    printf("# Time in integrator loop [s]: %12.5e\n",loop_time);
    if(flame_params.sensitivity_analysis_) printf("# Time in sensitivity loop [s]: %12.5e\n",sensanal_time);
    if(flame_params.uncertainty_quantification_) printf("# Time in UQ loop [s]: %12.5e\n",uq_time);
  }
  flame_params.logger_->PrintF(
    "# Simulation setup time   [s]: %12.5e\n",setup_time);
  flame_params.logger_->PrintF(
    "# Time in integrator loop [s]: %12.5e\n",loop_time);

  if(flame_params.integrator_type_ == 2) {
    flame_params.sparse_matrix_dist_->SparseMatrixClean_dist();
  }
  MPI_Finalize();
  return 0;
}

// Find maximum and its location
static double FindMaximumParallel(const size_t num_points,
                          const double x[],
                          const size_t x_stride,
                          const double f[],
                          const size_t f_stride,
                          const bool use_quadratic,
                          double *x_at_max)
{
  int myrank;
  struct {
    double value;
    int index;
  } in, out;

  // Compute local maximum
  in.value = f[0];
  in.index = 0;
  for(int j=1; j<(int)num_points; ++j) {
    if(in.value < f[j*f_stride]) {
      in.value = f[j*f_stride];
      in.index = j;
    }
  }

  // Compute global maximum
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  in.index += myrank*num_points;

  MPI_Allreduce(&in,&out,1,MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

  *x_at_max = x[out.index];

  return out.value;
}

// Write state variables to binary file
static void WriteFieldParallel(double t,
			       const double state[],
			       const FlameParams &params)
{
  const int num_grid_points = (int)params.z_.size();
  const int num_grid_points_ext = (int)params.z_.size() + 2; //with BC
  const int num_local_points = (int)params.num_local_points_;
  const int num_reactor_states = params.reactor_->GetNumStates();
  int num_species = params.reactor_->GetNumSpecies();
  int num_steps = params.reactor_->GetNumSteps();
  int npes = params.npes_;
  int my_pe = params.my_pe_;
  bool dump_mole_fractions = params.parser_->write_mole_fractions_to_field_files();
  bool dump_step_rates = params.parser_->write_step_rates_to_field_files();

  int field_count = num_local_points;
  if(my_pe == 0) field_count +=1; //left BC
  if(my_pe == npes-1) field_count +=1; //right BC
 
  std::string filename = "data_" + std::to_string(t);

  // Count Variables
  int num_vars = num_reactor_states; //excludes mass flux
  if(dump_mole_fractions) num_vars += num_species;
  if(dump_step_rates) num_vars += num_steps + num_species;

  // Collect state names
  std::vector<std::string> state_names(num_vars);
  for(int j=0; j<num_reactor_states; ++j) {
    state_names[j] = params.reactor_->GetNameOfStateId(j);
  }
  int names_offset = num_reactor_states;
  if(dump_mole_fractions) {
    for(int j=0; j<num_reactor_states; ++j) {
      std::string state_name = params.reactor_->GetNameOfStateId(j);
      size_t found = state_name.find("MassFraction");
      if(found != std::string::npos) {
        state_name.replace(0, 4, "Mole");
        state_names[names_offset+j] = state_name;
      }
    }
    names_offset += num_species;
  }
  if(dump_step_rates) {
    for(int j=0; j<num_steps; ++j) {
      std::string step_name;
      params.reactor_->GetMechanism()->getReactionNameDirOfStep(j, &step_name);
      state_names[names_offset+j] = step_name + " ROP";
    }
    names_offset += num_steps;
    for(int j=0; j<num_species; ++j) {
      std::string state_name = params.reactor_->GetNameOfStateId(j);
      size_t found = state_name.find("MassFraction");
      if(found != std::string::npos) {
        state_name.replace(0, 12, "NetProductionRate");
        state_names[names_offset+j] = state_name;
      }
    }
  }

  //Instantiate Writer
  zerork::utilities::RestartWriter writer(filename, field_count, num_vars, t, state_names, params.comm_);


  // Write data for each variable
  std::vector<double> buffer(field_count,0);
  for(int j=0; j<num_reactor_states; ++j) {
    int offset = 0;
    if(my_pe==0) {
      offset = 1;
      // Write left (oxidizer) BC data
      if(j==num_reactor_states-1){
        buffer[0] = params.oxidizer_temperature_*params.ref_temperature_;
      } else if (j==num_reactor_states-2) {
        buffer[0] = params.oxidizer_relative_volume_;
      } else {
        buffer[0] = params.oxidizer_mass_fractions_[j];
      }
    }

    // Write interior data
    for (int k=0; k<num_local_points; ++k) {
      buffer[k+offset] = state[k*num_reactor_states + j];
      if(j==num_reactor_states-1){
	buffer[k+offset] *= params.ref_temperature_;
      }
    }

    if(my_pe==npes-1) {
      // Write right (fuel) BC data
      if(j==num_reactor_states-1) {
        buffer[num_local_points+offset] = params.fuel_temperature_*params.ref_temperature_;
      } else if (j==num_reactor_states-2) {
        buffer[num_local_points+offset] = params.fuel_relative_volume_;
      } else {
        buffer[num_local_points+offset] = params.fuel_mass_fractions_[j];
      }
    }
    writer.WriteField(buffer);
  }

  if(dump_mole_fractions) {
    std::vector<double> fuel_mole_fractions(num_species);
    std::vector<double> oxidizer_mole_fractions(num_species);
    std::vector<double> state_mole_fractions(num_species*num_local_points);
    params.reactor_->GetMechanism()->getXfromY(&params.fuel_mass_fractions_[0],&fuel_mole_fractions[0]);
    params.reactor_->GetMechanism()->getXfromY(&params.oxidizer_mass_fractions_[0],&oxidizer_mole_fractions[0]);
    for (int k=0; k<num_local_points; ++k) {
      params.reactor_->GetMechanism()->getXfromY(&state[k*num_reactor_states],&state_mole_fractions[k*num_species]);
    }

    for(int j=0; j<num_species; ++j) {
      // Write left (fuel) BC data
      int offset = 0;
      if(my_pe == 0) {
        offset+=1;
        buffer[0] = fuel_mole_fractions[j];
      }

      // Write interior data
      for (int k=0; k<num_local_points; ++k) {
        buffer[k+offset] = state_mole_fractions[k*num_species + j];
      }

      if(my_pe == npes-1) {
        // Write right (oxidizer) BC data
        buffer[num_local_points+offset] = oxidizer_mole_fractions[j];
      }

      writer.WriteField(buffer);
    }
  }

  if(dump_step_rates) {
    std::vector<double> creation_rates(num_steps);
    std::vector<double> destruction_rates(num_steps);
    std::vector<double> concentrations(num_species);

    std::vector<double> fuel_step_rates(num_steps);
    std::vector<double> fuel_net_rates(num_species);
    double relative_volume = params.fuel_relative_volume_;
    double density         = 1.0/relative_volume;
    double temperature     = params.fuel_temperature_*params.ref_temperature_;
    params.reactor_->GetMechanism()->getCfromVY(relative_volume,&params.fuel_mass_fractions_[0],&concentrations[0]);
    params.reactor_->GetMechanism()->getReactionRates(temperature,&concentrations[0],
		                                      &fuel_net_rates[0],
		                                      &creation_rates[0],
		                                      &destruction_rates[0],
		                                      &fuel_step_rates[0]);

    std::vector<double> oxidizer_step_rates(num_steps);
    std::vector<double> oxidizer_net_rates(num_species);
    relative_volume = params.oxidizer_relative_volume_;
    density         = 1.0/relative_volume;
    temperature     = params.oxidizer_temperature_*params.ref_temperature_;
    params.reactor_->GetMechanism()->getCfromVY(relative_volume,&params.oxidizer_mass_fractions_[0],&concentrations[0]);
    params.reactor_->GetMechanism()->getReactionRates(temperature,&concentrations[0],
		                                      &oxidizer_net_rates[0],
		                                      &creation_rates[0],
		                                      &destruction_rates[0],
		                                      &oxidizer_step_rates[0]);

    std::vector<double> state_step_rates(num_steps*num_local_points, 0.0);
    std::vector<double> state_net_rates(num_species*num_local_points, 0.0);
    for (int k=0; k<num_local_points; ++k) {
      relative_volume = state[k*num_reactor_states+num_species];
      density         = 1.0/relative_volume;
      temperature     = state[k*num_reactor_states+num_species+1]*params.ref_temperature_;
      params.reactor_->GetMechanism()->getCfromVY(relative_volume,&state[k*num_reactor_states],&concentrations[0]);
      params.reactor_->GetMechanism()->getReactionRates(temperature,&concentrations[0],
                                                        &state_net_rates[k*num_species],
                                                        &creation_rates[0],
                                                        &destruction_rates[0],
                                                        &state_step_rates[k*num_steps]);
    }

    for(int j=0; j<num_steps; ++j) {
      int offset = 0;
      if(my_pe == 0) {
        offset += 1;
        // Write left (fuel) BC data
        buffer[0] = fuel_step_rates[j];
      }

      // Write interior data
      for (int k=0; k<num_local_points; ++k) {
        buffer[k+offset] = state_step_rates[k*num_steps + j];
      }

      // Write right (oxidizer) BC data
      if(my_pe == npes-1) {
        buffer[num_local_points+offset] = oxidizer_step_rates[j];
      }

      writer.WriteField(buffer);
    }

    for(int j=0; j<num_species; ++j) {
      int offset = 0;
      if(my_pe == 0) {
        offset += 1;
        // Write left (fuel) BC data
        buffer[0] = fuel_net_rates[j];
      }

      // Write interior data
      for (int k=0; k<num_local_points; ++k) {
        buffer[k+offset] = state_net_rates[k*num_species + j];
      }


      if(my_pe == npes-1) {
        // Write right (oxidizer) BC data
        buffer[num_local_points+offset] = oxidizer_net_rates[j];
      }
      writer.WriteField(buffer);
    }
  }
}

static double GetMixtureMolecularMass(const int grid_id,
                                      const double t,
                                      const double state[],
                                      const FlameParams &params)
{
  const int num_reactor_states = params.reactor_->GetNumStates();
  const int num_species = params.reactor_->GetNumSpecies();
  double molecular_mass=0.0;

  if(0 <= grid_id && grid_id < (int)params.z_.size()) {

    molecular_mass=0.0;
    for(int j=0; j<num_species; ++j) {
      molecular_mass += state[grid_id*num_reactor_states+j]*
        params.inv_molecular_mass_[j];
    }
    molecular_mass = 1.0/molecular_mass;
  }
  return molecular_mass;
}
static double minSumMassFractions(const double state[],
                                  const FlameParams &params)
{
  const int num_reactor_states = params.reactor_->GetNumStates();
  const int num_species = params.reactor_->GetNumSpecies();
  const int num_local_points = (int)params.num_local_points_;
  double sum_mass_fraction;
  double local_min;
  double global_min;

  local_min = 100;
  for(int k=0; k<num_local_points; ++k) {
    sum_mass_fraction=0.0;
    for(int j=0; j<num_species; ++j) {
      sum_mass_fraction += state[k*num_reactor_states+j];
    }
    if (sum_mass_fraction < local_min) {
      local_min = sum_mass_fraction;
    }
  }

  MPI_Allreduce(&local_min,&global_min,1,PVEC_REAL_MPI_TYPE,MPI_MIN,MPI_COMM_WORLD);
  return global_min;
}
static double maxSumMassFractions(const double state[],
                                  const FlameParams &params)
{
  const int num_reactor_states = params.reactor_->GetNumStates();
  const int num_species = params.reactor_->GetNumSpecies();
  const int num_local_points = (int)params.num_local_points_;
  double sum_mass_fraction;
  double local_max, global_max;

  local_max = -100.0;

  for(int k=0; k<num_local_points; ++k) {
    sum_mass_fraction=0.0;
    for(int j=0; j<num_species; ++j) {
      sum_mass_fraction += state[k*num_reactor_states+j];
    }
    if (sum_mass_fraction > local_max) {
      local_max = sum_mass_fraction;
    }
  }

  MPI_Allreduce(&local_max,&global_max,1,PVEC_REAL_MPI_TYPE,MPI_MAX,MPI_COMM_WORLD);
  return global_max;
}

static int GetTrackMaxStateId(const FlameParams &params,
                              std::vector<int> *state_id)
{
  std::string prefix="MassFraction_";
  std::string state_name;
  const int num_reactor_states = params.reactor_->GetNumStates();
  const int num_species = params.reactor_->GetNumSpecies();
  int species_id;
  state_id->clear();
  state_id->push_back(num_reactor_states-1); // set the temperature id

  for(size_t j=0; j<params.parser_->track_max().size(); ++j) {
    state_name.clear();
    state_name = prefix + params.parser_->track_max()[j];
    species_id = params.reactor_->GetIdOfState(state_name.c_str());
    if(0 <= species_id && species_id < num_species) {
      state_id->push_back(species_id);
    } else {
      params.logger_->PrintF(
        "# WARNING: In GetTrackMaxStateId,\n"
        "#          input file species given by track_max[%lu] = '%s'\n"
        "#          is not found in the mechanism. Skipping species.\n",
        j,
        params.parser_->track_max()[j].c_str());
      params.logger_->FFlush();
    }
  }
  return (int)state_id->size();
}

static int GetStateMaxima(const std::vector<int> &state_id,
                          const double state[],
                          const FlameParams &params,
                          std::vector<double> *max_value,
                          std::vector<double> *max_position)
{
  const int num_local_points = (int)params.num_local_points_;
  const int num_reactor_states = params.reactor_->GetNumStates();

  double state_max, state_max_position;
  max_value->clear();
  max_position->clear();

  for(size_t j=0; j<state_id.size(); ++j) {

    state_max = FindMaximumParallel(num_local_points,
				    &params.z_[0],
				    1,
				    &state[state_id[j]],
				    num_reactor_states,
				    true, // use quadratic
				    &state_max_position);
    if(state_id[j] == num_reactor_states-1) {
      // temperature
      state_max*=params.ref_temperature_;
    }
    max_value->push_back(state_max);
    max_position->push_back(state_max_position);
  }

  return 0;
}

static int GetFuelSpeciesId(const FlameParams &params,
                            std::vector<int> *species_id)
{
  const int num_species = params.reactor_->GetNumSpecies();
  std::map<std::string, double>::const_iterator iter;
  species_id->clear();
  for(iter=params.parser_->inlet_fuel_comp().begin();
      iter != params.parser_->inlet_fuel_comp().end();
      ++iter) {

    std::string species_name=iter->first;
    std::string state_name = std::string("MassFraction_")+species_name;
    double mole_fraction = iter->second;
    int id = params.reactor_->GetIdOfState(state_name.c_str());
    if(0 <= id && id < num_species && mole_fraction > 0.0) {
      species_id->push_back(id);
    }
  }
  return (int)species_id->size();
}

static double StoichiometricMixtureFraction(const FlameParams &params)
{
  double fuel_sum = 0.0;
  std::vector<int> fuel_species_id;
  GetFuelSpeciesId(params,&fuel_species_id);
  for(size_t j=0; j<fuel_species_id.size(); ++j) {
    fuel_sum += params.stoichiometric_mass_fractions_[fuel_species_id[j]];
  }
  return fuel_sum;
}

int compare_rxnSens_t(const void *A, const void *B)
{
  rxnSens_t *Aptr =(rxnSens_t *)A;
  rxnSens_t *Bptr =(rxnSens_t *)B;
  if(Aptr->relSensAbs < Bptr->relSensAbs) {
    return 1;
  }
  else if (Aptr->relSensAbs > Bptr->relSensAbs) {
    return -1;
  }
  return 0;
}
