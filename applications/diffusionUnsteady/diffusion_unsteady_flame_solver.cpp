#include <stdlib.h>
#include <stdio.h>

#include <vector>
#include <string>
#include <map>

#include <cvode/cvode.h>            // prototypes for CVODE fcts. and consts.
#include <nvector/nvector_parallel.h> // parallel N_Vector types, fcts., and macros

#ifdef SUNDIALS2
#include <cvode/cvode_spgmr.h>
#elif SUNDIALS3
#include <cvode/cvode_spils.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#elif SUNDIALS4
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sunnonlinsol/sunnonlinsol_newton.h>
#endif

#include "utilities.h"
#include <utilities/file_utilities.h>

#include <mpi.h>

#include "flame_params.h"
#include "cvode_functions.h"
#include "set_initial_conditions.h"

#include "restart_writer.h"

using zerork::getHighResolutionTime;

const bool RUN_DEBUG=false;
const int NUM_STDOUT_PARAMS = 12; // number of non species parameters to write
                                  // to standard out
const int NUM_LOG_PARAMS = 10;    // number of non species parameters to write
                                  // LogGridPointState

static int check_flag(void *flagvalue, const char *funcname, int opt);

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

static double Min(double a, double b)
{
  return ((a < b) ? a : b);
}

int main(int argc, char *argv[])
{
  double clock_time = getHighResolutionTime();
  double setup_time, loop_time;

  if(argc < 2) {
    printf("# ERROR: Incorrect command line usage.\n");
    printf("#        use %s <input parameters>\n",argv[0]);
    exit(-1);
  }

  // MPI
  MPI_Comm comm;
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;

  // CVode memory pointer and linear solver
  void *cvode_ptr = NULL;
#ifdef SUNDIALS3
  SUNLinearSolver LS;
  LS = NULL;
#elif SUNDIALS4
  SUNLinearSolver LS;
  SUNNonlinearSolver NLS;
  LS = NULL;
  NLS = NULL;
#endif

  // Initialize flame params
  FlameParams flame_params(argv[1],comm);

  // Initialize stte variables vector and pointer
  N_Vector flame_state;
  flame_state = NULL;
  double *flame_state_ptr;
  flame_state_ptr = NULL;

  // Get constants from flame params
  const int num_grid_points = flame_params.z_.size();
  const int num_local_points = flame_params.num_local_points_;
  const int num_states = flame_params.reactor_->GetNumStates();
  long int num_local_states = num_local_points*num_states;
  long int total_states = num_grid_points*num_states;
  double max_time = flame_params.parser_->max_time();
  const double step_dt = flame_params.parser_->stats_dt();
  const double ref_temperature = flame_params.ref_temperature_;
  const double dz = flame_params.dz_[(int)num_grid_points/2]; //should be min(dz_)
  int my_pe = flame_params.my_pe_;

  // Declare variables
  int flag=0;
  int num_prints = 0;
  double current_time = 0.0;
  double time_offset;
  double next_time;
  double fuel_density, fuel_molecular_mass;
  double stoichiometric_mixture_fraction;
  double min_sum_mass_fraction, max_sum_mass_fraction;

  // CVode integrator Stats
  long int nsteps, nfevals, nlinsetups, netfails;
  int qlast, qcur;
  double hinused, hlast, hcur, tcur;

  std::vector<double> temperature_jump;
  std::vector<int> track_max_state_id;
  std::vector<double> state_maxima, state_maxima_positions;

  // allocate CVode data structures
  flame_state          = N_VNew_Parallel(comm, num_local_states, total_states);
  flame_state_ptr      = NV_DATA_P(flame_state);

  // Initialize state vector
  time_offset = 0.0;
  SetInitialCompositionAndWallTemp(flame_params, flame_state_ptr, &time_offset);
  max_time -= time_offset;

  // Initialize Soot
  if(flame_params.soot_)
    InitializePAH(&flame_params);

  temperature_jump.assign(num_local_points,0.0);
  GetTrackMaxStateId(flame_params, &track_max_state_id);

  // Setup CVode
    /* Call CVodeCreate to create the solver memory and specify the
   * Backward Differentiation Formula and the use of a Newton iteration */
#ifdef SUNDIALS4
  cvode_ptr = CVodeCreate(CV_BDF);
#else
  cvode_ptr = CVodeCreate(CV_BDF, CV_NEWTON);
#endif
  if (check_flag((void *)cvode_ptr, "CVodeCreate", 0)) exit(-1);

  /* Call CVodeInit to initialize the integrator memory and specify the
   * user's right hand side function in y'=f(t,y), the inital time T0, and
   * the initial dependent variable vector y. */
  flag=CVodeInit(cvode_ptr,
                 ConstPressureFlame,
                 0.0,
                 flame_state);
  if (check_flag(&flag, "CVodeInit", 1)) exit(-1);

  /* Call CVodeSStolerances to specify the scalar relative tolerance
   * and scalar absolute tolerances */
  flag = CVodeSStolerances(cvode_ptr,
                           flame_params.parser_->rel_tol(),
                           flame_params.parser_->abs_tol());
  if (check_flag(&flag, "CVodeSStolerances", 1)) exit(-1);

  /* Set the pointer to user-defined data */
  flag = CVodeSetUserData(cvode_ptr,
                          &flame_params);
  if(check_flag(&flag, "CVodeSetUserData", 1)) exit(-1);

#ifdef SUNDIALS2
  /* Setup the linear solver method */
  flag = CVSpgmr(cvode_ptr, PREC_LEFT, 5);
  if(check_flag(&flag, "CVSpgmr", 1)) exit(-1);

  flag = CVSpilsSetGSType(cvode_ptr, MODIFIED_GS);
  if(check_flag(&flag, "CVSpilsSetGSType", 1)) exit(-1);

   /* Set the preconditioner setup and solve functions */
  flag = CVSpilsSetPreconditioner(cvode_ptr,
                                  ReactorPreconditionerSetup,
                                  ReactorPreconditionerSolve);
  if(check_flag(&flag, "CVSpilsSetPreconditioner", 1)) exit(-1);

#elif SUNDIALS3
  /* Setup the linear solver method */
  // TODO: add SPGMR solver parameters to input file for user control
  LS = SUNSPGMR(flame_state,
                PREC_LEFT,
                5); //max krylov dimension

  flag = CVSpilsSetLinearSolver(cvode_ptr, LS);
  if(check_flag(&flag, "CVSpilsSetLinearSolver", 1)) exit(-1);

  flag = SUNSPGMRSetGSType(LS, MODIFIED_GS);
  if(check_flag(&flag, "SUNSPGMRSetGSType", 1)) exit(-1);

  /* Set the preconditioner setup and solve functions */
  flag = CVSpilsSetPreconditioner(cvode_ptr,
                                  ReactorPreconditionerSetup,
                                  ReactorPreconditionerSolve);
  if(check_flag(&flag, "CVSpilsSetPreconditioner", 1)) exit(-1);
#elif SUNDIALS4
  /* Setup the nonlinear solver method */
  NLS = SUNNonlinSol_Newton(flame_state);
  flag = CVodeSetNonlinearSolver(cvode_ptr, NLS);
  if(check_flag(&flag, "CVodeSetNonlinearSolver", 1)) exit(-1);

  /* Setup the linear solver method */
  // TODO: add SPGMR solver parameters to input file for user control
  LS = SUNLinSol_SPGMR(flame_state,
                       PREC_LEFT,
                       5); //max krylov dimension

  flag = CVodeSetLinearSolver(cvode_ptr, LS, NULL);
  if(check_flag(&flag, "CVodeSetLinearSolver", 1)) exit(-1);

  /* Set the preconditioner setup and solve functions */
  flag = CVodeSetPreconditioner(cvode_ptr,
                                ReactorPreconditionerSetup,
                                ReactorPreconditionerSolve);
  if(check_flag(&flag, "CVodeSetPreconditioner", 1)) exit(-1);
#endif

  if(flame_params.parser_->integrator_type() != 3) {
    printf("integrator_type == %d not currently supported\n",
           flame_params.parser_->integrator_type());
    flag = -1;
  }
  if(flag != 0) {
    N_VDestroy_Parallel(flame_state);
    if(cvode_ptr != NULL) {
      CVodeFree(&cvode_ptr);
    }
#if defined SUNDIALS3 || defined SUNDIALS4
    SUNLinSolFree(LS);
#endif
#if defined SUNDIALS4
  SUNNonlinSolFree(NLS);
#endif
    return flag;
  }

  /* Set the maximum number of internal steps per CVode call and the maximum
   * allowable internal steps. */
  flag = CVodeSetMaxNumSteps(cvode_ptr,
                             flame_params.parser_->max_num_steps());
  if (flag < 0) {
    printf("#        CVodeSetMaxNumSteps failed with error flag = %d.\n",
           flag);
    CVodeFree(&cvode_ptr);
#if defined SUNDIALS3 || defined SUNDIALS4
    SUNLinSolFree(LS);
#endif
#if defined SUNDIALS4
  SUNNonlinSolFree(NLS);
#endif
    return flag;
  }

  flag = CVodeSetMaxStep(cvode_ptr,
                         flame_params.parser_->max_internal_dt());
  if (flag < 0) {
    printf("#        CVodeSetMaxStep failed with error flag = %d.\n",
           flag);
    CVodeFree(&cvode_ptr);
#if defined SUNDIALS3 || defined SUNDIALS4
    SUNLinSolFree(LS);
#endif
#if defined SUNDIALS4
  SUNNonlinSolFree(NLS);
#endif
    return flag;
  }

  // Write initial field
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
    printf("# Column  3: [-] Ysootmax\n");
    printf("# Column  4: [#] number of steps taken by CVode\n");
    printf("# Column  5: [#] number of calls to the user's time derivative (RHS) function\n");
    printf("# Column  6: [#] number of calls to the linear solver setup function\n");
    printf("# Column  7: [#] number of error test failures\n");
    printf("# Column  8: [#] current method order of time integrator\n");
    printf("# Column  9: [s] current size of internal time step\n");
    printf("# Column 10: [s] characteristic time step for diffusion stability control 0.5*dx*dx/max(thermal diffusivity)\n");
    printf("# Column 11: min sum of mass fractions\n");
    printf("# Column 12: max sum of mass fractions\n");
    for(int j=0; j<(int)track_max_state_id.size(); ++j) {
      printf("# Column %2d: maximum %s in the domain\n",
	     NUM_STDOUT_PARAMS+1+2*j,
	     flame_params.reactor_->GetNameOfStateId(track_max_state_id[j]));
      printf("# Column %2d: [m] location of the maximum %s\n",
	     NUM_STDOUT_PARAMS+2+2*j,
	     flame_params.reactor_->GetNameOfStateId(track_max_state_id[j]));
    }
  }// if(my_pe==0)

  // integrate ODE system
  next_time = Min(step_dt, max_time);

  // Save time for CPU time evaluation
  setup_time = getHighResolutionTime() - clock_time;
  clock_time = getHighResolutionTime();

  // While simulation loop
  while(current_time < max_time) {

    flag = CVode(cvode_ptr,
                 next_time,
                 flame_state,
                 &current_time,
                 CV_NORMAL);

    if(flag == CV_SUCCESS || current_time >= next_time) {
      next_time += step_dt;
      ++num_prints;

      // Get min/max sum(Yi) and velocity
      min_sum_mass_fraction = minSumMassFractions(flame_state_ptr,flame_params);
      max_sum_mass_fraction = maxSumMassFractions(flame_state_ptr,flame_params);

      // Get max and locations of tracked variables
      GetStateMaxima(track_max_state_id,
                     flame_state_ptr,
                     flame_params,
                     &state_maxima,
                     &state_maxima_positions);

      // Get integrator stats
      CVodeGetIntegratorStats(cvode_ptr, &nsteps, &nfevals,
                              &nlinsetups, &netfails, &qlast, &qcur,
                              &hinused, &hlast, &hcur, &tcur);

      // Track integrated soot yield
      if (flame_params.soot_) {UpdateDimerProdRate(&flame_params,&flame_state_ptr[0]);}

      // Print to screen/log
      if(my_pe == 0) { //only root prints
	printf("%7d  %14.7e  %14.7e  %6d  %6d  %6d  %6d  %6d  %14.7e  %14.7e  %14.7e  %14.7e",
	       num_prints,
	       current_time+time_offset,
               flame_params.Y_sootmax,
	       (int)nsteps,
	       (int)nfevals,
	       (int)nlinsetups,
	       (int)netfails,
	       qcur,
	       hcur,
	       0.5*dz*dz/flame_params.max_thermal_diffusivity_,
	       min_sum_mass_fraction,
	       max_sum_mass_fraction);
	for(size_t j=0; j<track_max_state_id.size(); ++j) {
	  printf("  %14.7e  %14.7e",state_maxima[j], state_maxima_positions[j]);
	}
	printf("\n");
      } // if(my_pe==0)

      if(num_prints % flame_params.parser_->field_dt_multiplier() == 0) {
	WriteFieldParallel(current_time+time_offset,
			   &flame_state_ptr[0],
			   flame_params);
        // Write soot
        if (flame_params.soot_) {WriteDimerProdRate(&flame_params,&flame_state_ptr[0]);}
      }

    } else {
      printf("# ERROR: In time marching loop,\n");
      printf("#        CVode returned flag = %d.\n", flag);
      printf("#        Halting integration.\n");
      break;
    }

  }
  loop_time = getHighResolutionTime() - clock_time;

  N_VDestroy_Parallel(flame_state);
  if(cvode_ptr != NULL) {
    CVodeFree(&cvode_ptr);
  }
#if defined SUNDIALS3 || defined SUNDIALS4
  SUNLinSolFree(LS);
#endif
#if defined SUNDIALS4
  SUNNonlinSolFree(NLS);
#endif
  if(my_pe == 0) {
    printf("# Simulation setup time   [s]: %12.5e\n",setup_time);
    printf("# Time in integrator loop [s]: %12.5e\n",loop_time);
  }
  flame_params.logger_->PrintF(
    "# Simulation setup time   [s]: %12.5e\n",setup_time);
  flame_params.logger_->PrintF(
    "# Time in integrator loop [s]: %12.5e\n",loop_time);

  MPI_Finalize();
  return 0;
}

// Find maximum and its location in parallel
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

// Write field to binary file for restart/postprocessing
static void WriteFieldParallel(double t,
			       const double state[],
			       const FlameParams &params)
{
  const int num_grid_points = (int)params.z_.size();
  const int num_grid_points_ext = num_grid_points + 2; //with BC
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
    if(my_pe == 0) {
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

    if(my_pe == npes-1) {
      // Write right (fuel) BC data
      if(j==num_reactor_states-1){
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

/*
 * Check function return value...
 *   opt == 0 means SUNDIALS function allocates memory so check if
 *            returned NULL pointer
 *   opt == 1 means SUNDIALS function returns a flag so check if
 *            flag >= 0
 *   opt == 2 means function allocates memory so check if returned
 *            NULL pointer
*/

static int check_flag(void *flagvalue, const char *funcname, int opt)
{
  int *errflag;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && flagvalue == NULL) {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return(1); }

  /* Check if flag < 0 */
  else if (opt == 1) {
    errflag = (int *) flagvalue;
    if (*errflag < 0) {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
              funcname, *errflag);
      return(1); }}

  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && flagvalue == NULL) {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return(1); }

  return(0);
}
