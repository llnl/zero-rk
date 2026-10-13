///////////////////////////////////////////////////////////////////////
//
//     This demonstration program builds an object representing a
//     reacting gas mixture, and uses it to compute thermodynamic
//     properties, chemical equilibrium, and transport properties.
//
///////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

const int MAX_THIST_LINE=1000000; //20000;

#ifdef ZERORK_MPI
#include <mpi.h>
#include "utilities/mpi_utilities.h"
#endif

#include <cvode/cvode.h>            // prototypes for CVODE fcts. and consts.
#include <nvector/nvector_serial.h> // serial N_Vector types, fcts., and macros

#ifdef SUNDIALS2
#include <cvode/cvode_spgmr.h>      // prototypes & constants for CVSPGMR
#elif SUNDIALS3
#include <cvode/cvode_spils.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#elif SUNDIALS4
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sunnonlinsol/sunnonlinsol_newton.h>
#endif

#include <zerork/constants.h>
#include <zerork/constants_api.h>

#include "matrix_funcs.h"
#include "cv_param_sparse.h"
#include "ode_funcs.h"
#include "sweep_util_spify.h"

#include "equil/zerork_equilibrium_solver.h"

int setMoleFracFromFile(zerork::mechanism &mechInp, const char *fileName,
                        double moleFrac[]);

static int check_flag(void *flagvalue, const char *funcname, int opt);

void getTimeHistLine_full(const double currTime,
                          const double *sysState,
                          const idt_sweep_params *idt_ctrl,
                          const cv_param *cvp,
                          const double wallTime,
                          const int particle,
                          char *thistLine);

void getTimeHistHeader_full(const idt_sweep_params *idt_ctrl,
                            const cv_param *cvp,
                            char *thistHead);

int getIdtLine(const double initMoleFrac[],
               const idt_sweep_params *idt_ctrl,
               const double idt,
               const double runTime,
               const int nCvodeFails,
               const int maxBuffer,
               char *idtLine);
int getIdtHeader(const idt_sweep_params *idt_ctrl,
                 const cv_param *cvp,
                 const int maxBuffer,
                 char *idtHeader);


int getSimHeader_full(const int inp_argc,
                      char **inp_argv,
                      const idt_sweep_params *idt_ctrl,
                      const cv_param *cvp,
                      const int max,
                      char *simHead);

// PASR
int inflow(const int num_streams,
           const double flow_rates[],
           double running_flow_rates[]);

void mix(const int num_particles,
        const int nState,
        const double dt,
        const double tau_mix,
        double *state);

void select_pairs(const int num_particles,
                  const int nstate,
                  const int num_pairs,
                  const int num_skip,
                  double *state);

void cvReactor(int inp_argc, char **inp_argv)
{

#ifdef ZERORK_MPI
  int rank, nranks;
  MPI_Init(&inp_argc, &inp_argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
#else
  int rank = 0;
#endif

  double idt,tmax;
  int isIdtFound;
  double nRunsPerThresh;

  FILE *idtFilePtr,*thistFilePtr;
  //FILE *filePtr;
  //filePtr = fopen("pasr_states","w");
  int i,j,k,l;
  int nTrackSpec,nSpc,nState,nStep;
  double tnext,tcurr;
  // cvode variables
  N_Vector systemState, systemState_local, equiState;
  cv_param systemParam;
  void *cvode_mem;
#ifdef SUNDIALS3
  SUNLinearSolver LS;
#elif SUNDIALS4
  SUNLinearSolver LS;
  SUNNonlinearSolver NLS;
#endif

  int flag;

  char thistLine[MAX_THIST_LINE];
  int did_cvodeFail;
  int isBadStep;

  // timing data
  double startTime,stopTime, simTime, commTime,rank0Time,lastIterTime;
  commTime = 0;
  rank0Time = 0;

  // PaSR
  int nParticles;

  if(inp_argc != 2)
  {
    if(rank == 0) {
      printf("ERROR: incorrect command line usage\n");
      printf("       use instead %s <idt sweep input>\n",inp_argv[0]);
      fflush(stdout);
    }
    exit(-1);
  }

  startTime=zerork::getHighResolutionTime();
  // sweep controls
  idt_sweep_params idt_ctrl(inp_argv[1]);

  if(idt_ctrl.random_seed() != 0) {
    srand(idt_ctrl.random_seed());
  }

  stopTime=zerork::getHighResolutionTime();

  nSpc=idt_ctrl.getNumSpecies();
  nState=idt_ctrl.getNumSpecies()+1;
  nStep=idt_ctrl.getNumSteps();
  nParticles=idt_ctrl.getNumParticles();

  std::vector<double> avgTime(idt_ctrl.getNumThreshRuns(), 0.0);
  std::vector<double> minTime(idt_ctrl.getNumThreshRuns(), 1.0e300);
  std::vector<double> maxTime(idt_ctrl.getNumThreshRuns(), -1.0e300);

  systemState = N_VNew_Serial(nState*nParticles);//
  double* massFracPtr=NV_DATA_S(systemState); // caution: assumes realtype == double
  if(idt_ctrl.fully_coupled()) {
    systemState_local = N_VNew_Serial(2*nState);
  } else {
    systemState_local = N_VNew_Serial(nState);
  }
  equiState = N_VNew_Serial(nState);
  double* equiStatePtr=NV_DATA_S(equiState);
  std::vector<double> moleFracCurr(nSpc);
  std::vector<double> moleFracInit(nSpc);

  // set up the system parameters
  systemParam.nSpc=idt_ctrl.getNumSpecies();
  systemParam.nStep=idt_ctrl.getNumSteps();
  systemParam.minMassFrac=1.0e-30;
  systemParam.sqrtUnitRnd=sqrt(UNIT_ROUNDOFF);
  systemParam.Tref=idt_ctrl.getRefTemp();
  systemParam.mech=idt_ctrl.getMechPtr();
  systemParam.htxModel = zerork::HeatTransferModelFactory(systemParam.mech,  idt_ctrl);

  //
  systemParam.netProd    =(double *)malloc(sizeof(double)*nSpc);
  systemParam.Energy     =(double *)malloc(sizeof(double)*nSpc);
  systemParam.CpMass     =(double *)malloc(sizeof(double)*nSpc);
  systemParam.molWt      =(double *)malloc(sizeof(double)*nSpc);
  systemParam.invMolWt   =(double *)malloc(sizeof(double)*nSpc);
  systemParam.fwdROP     =(double *)malloc(sizeof(double)*nStep);
  systemParam.createRate =(double *)malloc(sizeof(double)*nSpc);
  systemParam.destroyRate=(double *)malloc(sizeof(double)*nSpc);
  systemParam.conc       =(double *)malloc(sizeof(double)*nSpc);

  systemParam.step_limiter = (double *)malloc(sizeof(double)*nStep);
  for (j=0; j<nStep; j++)
    {systemParam.step_limiter[j] = idt_ctrl.getStepLimiter();}

  // PaSR
  systemParam.nParticles = nParticles;
  systemParam.nState = nState;

  // set constant parameters
  systemParam.mech->getMolWtSpc(systemParam.molWt);
  for(j=0; j<nSpc; j++)
    systemParam.invMolWt[j]=1.0/systemParam.molWt[j];

  // set up sparse jacobian matrix class
  startTime=zerork::getHighResolutionTime();
  systemParam.sparseMtx =
    (Jsparse *)alloc_Jsparse(*systemParam.mech,0.0,idt_ctrl.getILU(),
                             idt_ctrl.getUpdate(),idt_ctrl.getThreshType(),
                             idt_ctrl.getPartialPivotThresh(), idt_ctrl.getPermutationType());

  stopTime=zerork::getHighResolutionTime();

  /* Call CVodeCreate to create the solver memory and specify the
   * Backward Differentiation Formula and the use of a Newton iteration */
  startTime=zerork::getHighResolutionTime();
#ifdef SUNDIALS4
  cvode_mem = CVodeCreate(CV_BDF);
#else
  cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
#endif
  if (check_flag((void *)cvode_mem, "CVodeCreate", 0)) exit(-1);
  systemParam.cvodeMemPtr=cvode_mem;

  /* Call CVodeInit to initialize the integrator memory and specify the
   * user's right hand side function in y'=f(t,y), the inital time T0, and
   * the initial dependent variable vector y. */
  if(idt_ctrl.fully_coupled()) {
    flag=CVodeInit(cvode_mem, const_vol_wsr_mix, 0.0, systemState_local);
  } else {
    flag=CVodeInit(cvode_mem, const_vol_wsr, 0.0, systemState_local);
  }
  if (check_flag(&flag, "CVodeInit", 1)) exit(-1);

  /* Call CVodeSStolerances to specify the scalar relative tolerance
   * and scalar absolute tolerances */
  flag = CVodeSStolerances(cvode_mem,idt_ctrl.getRTol(),idt_ctrl.getATol());
  if (check_flag(&flag, "CVodeSStolerances", 1)) exit(-1);

  /* Call CVodeRootInit to specify the root function with 1 component */
  //flag = CVodeRootInit(cvode_mem, 1, tempRootFunc);
  //if (check_flag(&flag, "CVodeRootInit", 1)) exit(-1);

  /* Set the pointer to user-defined data */
  flag = CVodeSetUserData(cvode_mem, &systemParam);
  if(check_flag(&flag, "CVodeSetUserData", 1)) exit(-1);

  /* Call CVSpgmr to specify the linear solver CVSPGMR
     with left preconditioning and the maximum Krylov dimension maxl */
#ifdef SUNDIALS2
  flag = CVSpgmr(cvode_mem, PREC_LEFT, idt_ctrl.getKrylovDim());
  if(check_flag(&flag, "CVSpgmr", 1)) exit(-1);

  flag = CVSpilsSetGSType(cvode_mem, MODIFIED_GS);
  if(check_flag(&flag, "CVSpilsSetGSType", 1)) exit(-1);

  /* Set preconditioner setup and solve routines Precond and PSolve,
     and the pointer to the user-defined block data */
  if(idt_ctrl.fully_coupled()) {
    flag = CVodeSetPreconditioner(cvode_mem, jac_full_prec_setup_mix,
                                  jac_full_prec_solveV3_mix);
  } else {
    flag = CVodeSetPreconditioner(cvode_mem, jac_full_prec_setup,
                                  jac_full_prec_solveV3);
  }
  if(check_flag(&flag, "CVSpilsSetPreconditioner", 1)) exit(-1);
#elif SUNDIALS3
  LS = SUNSPGMR(systemState_local, PREC_LEFT, idt_ctrl.getKrylovDim());

  flag = CVSpilsSetLinearSolver(cvode_mem, LS);
  if(check_flag(&flag, "CVSpilsSetLinearSolver", 1)) exit(-1);

  flag = SUNSPGMRSetGSType(LS, MODIFIED_GS);
  if(check_flag(&flag, "SUNSPGMRSetGSType", 1)) exit(-1);
    /* Set preconditioner setup and solve routines Precond and PSolve,
     and the pointer to the user-defined block data */
  if(idt_ctrl.fully_coupled()) {
    flag = CVodeSetPreconditioner(cvode_mem, jac_full_prec_setup_mix,
                                  jac_full_prec_solveV3_mix);
  } else {
    flag = CVodeSetPreconditioner(cvode_mem, jac_full_prec_setup,
                                  jac_full_prec_solveV3);
  }
  if(check_flag(&flag, "CVSpilsSetPreconditioner", 1)) exit(-1);

#elif SUNDIALS4
  NLS = SUNNonlinSol_Newton(systemState_local);
  flag = CVodeSetNonlinearSolver(cvode_mem, NLS);
  if(check_flag(&flag, "CVodeSetNonlinearSolver", 1)) exit(-1);
  LS = SUNLinSol_SPGMR(systemState_local, PREC_LEFT, idt_ctrl.getKrylovDim());
  flag = CVodeSetLinearSolver(cvode_mem, LS, NULL);
  if(check_flag(&flag, "CVodeSetLinearSolver", 1)) exit(-1);

  /* Set preconditioner setup and solve routines Precond and PSolve,
     and the pointer to the user-defined block data */
  if(idt_ctrl.fully_coupled()) {
    flag = CVodeSetPreconditioner(cvode_mem, jac_full_prec_setup_mix,
                                  jac_full_prec_solveV3_mix);
  } else {
    flag = CVodeSetPreconditioner(cvode_mem, jac_full_prec_setup,
                                  jac_full_prec_solveV3);
  }
  if(check_flag(&flag, "CVodeSetPreconditioner", 1)) exit(-1);
#endif

  /* Set the maximum number of internal steps per CVode call and the maximum
   * allowable internal steps. */
  flag = CVodeSetMaxNumSteps(cvode_mem, idt_ctrl.getMaxInternalSteps());
  if (check_flag(&flag, "CVodeSetMaxNumSteps", 1)) exit(-1);
  flag = CVodeSetMaxStep(cvode_mem, idt_ctrl.getMaxInternalDt());
  if (check_flag(&flag, "CVodeSetMaxStep", 1)) exit(-1);

  flag = CVodeSetNonlinConvCoef(cvode_mem, idt_ctrl.getNlConvCoeff()); // Default [0.1]
#if defined SUNDIALS2 || defined SUNDIALS3
  flag = CVSpilsSetEpsLin(cvode_mem, idt_ctrl.getEpsLin());    // Default [0.05]
#else
  flag = CVodeSetEpsLin(cvode_mem, idt_ctrl.getEpsLin());    // Default [0.05]
#endif

  stopTime=zerork::getHighResolutionTime();

  nTrackSpec = idt_ctrl.getNumTrackSpecies();

  // ready for integration

  // open ignition delay and time history files for write
  if(rank == 0) {
    idtFilePtr=fopen(idt_ctrl.getIdtFileName(),"w");
    if(idtFilePtr==NULL) {
      printf("ERROR: could not open output file %s for write\n",
             idt_ctrl.getIdtFileName());
      exit(-1);
    }
    thistFilePtr=fopen(idt_ctrl.getTHistFileName(),"w");
    if(thistFilePtr==NULL) {
      printf("ERROR: could not open output file %s for write\n",
             idt_ctrl.getTHistFileName());
      exit(-1);
    }
    // create the simulation header stored in a character array
    getSimHeader_full(inp_argc,
                      &inp_argv[0],
                      &idt_ctrl,
                      &systemParam,
                      MAX_THIST_LINE,
                      thistLine);
    // write the simulation  header to stdout, idt file and thist file
    printf("%s",thistLine);
    fflush(stdout);
    fprintf(thistFilePtr,"%s",thistLine);
    fflush(thistFilePtr);
    fprintf(idtFilePtr,"%s",thistLine);
    fflush(idtFilePtr);

    // create the time history header stored in a character array
    getTimeHistHeader_full(&idt_ctrl, &systemParam, thistLine);
    // write the time history header to stdout and the time history file
    printf("%s",thistLine);
    fflush(stdout);
    fprintf(thistFilePtr,"%s",thistLine);
    fflush(thistFilePtr);

    // create the idt file header stored in a character array
    getIdtHeader(&idt_ctrl, &systemParam, MAX_THIST_LINE, thistLine);
    // write the idt header to the idt file
    fprintf(idtFilePtr,"%s",thistLine);
    fflush(idtFilePtr);
  }

  tmax    = idt_ctrl.getResidenceTime()*idt_ctrl.getNumResidenceTime();

  for(j=0; j<idt_ctrl.getRunTotal(); j++)
  {
    // initialize the run counters
    systemParam.sparseMtx->reduceNNZ=0; // won't be set until the first J
    systemParam.sparseMtx->reduceNNZMix=0; // won't be set until the first J
    systemParam.sparseMtx->LUnnz = 0; // won't be set until the first J
    systemParam.sparseMtx->fillFactor= 100.; // won't be set until the first J
    systemParam.prevNumErrTestFails = 0;
    systemParam.nFunc=0;
    systemParam.nJacSetup=0;
    systemParam.nJacFactor=0;
    systemParam.nBackSolve=0;
    systemParam.nJacRescale=0;
    systemParam.nColPerm=0;
    systemParam.colPermTime   = 0;
    systemParam.jacFactorTime = 0;
    systemParam.backsolveTime = 0;
    systemParam.jacSetupTime = 0;
    systemParam.funcTime = 0;
    did_cvodeFail=0;
    isIdtFound=0;

    // reset the preconditioner threshold
    change_JsparseThresh(systemParam.sparseMtx,idt_ctrl.getThresh());

    // reset the time
    tcurr=0.0;
    tnext=tmax;
    idt=tmax;

    // Set state to initial conditions
    idt_ctrl.getInitMassFrac(&equiStatePtr[0]);
    equiStatePtr[nSpc]=
        idt_ctrl.getInitTemp()/idt_ctrl.getRefTemp();
    systemParam.pressure=idt_ctrl.getInitPres();

    // ------------ BEGIN Constrained Equibrium calc  -----------//
    zerork::equilibrium_solver zcm(*systemParam.mech);
    double Teq = equiStatePtr[systemParam.nSpc]*idt_ctrl.getRefTemp();
    zcm.equilibrate(systemParam.pressure, Teq,
                    &equiStatePtr[0]);
    equiStatePtr[systemParam.nSpc] = Teq/idt_ctrl.getRefTemp();

    //printf("# Equilibrium temperature: %5.3e K\n", equiStatePtr[num_species]*idt_ctrl.getRefTemp());

    std::vector<double> input_flow_rates = idt_ctrl.getInletRatios();
    int num_streams;
    std::vector<double> flow_rates;
    if(input_flow_rates.size() > 1) {
        if(idt_ctrl.getCombustionRegime() == 1) {
          //premixed
          num_streams = 2;
          if(input_flow_rates.size() != num_streams) {
            if(rank == 0) {
              printf("WARNING: inlet_ratios size does not match combustionRegime expectation (%d, %d).  Using default ratios [0.95, 0.05]",input_flow_rates.size(), num_streams);
	    }
            flow_rates.assign(num_streams, 0.0);
            flow_rates[0] = 0.95; flow_rates[1] = 0.05;
          } else {
            flow_rates = input_flow_rates;
          }
        } else if (idt_ctrl.getCombustionRegime() == 2) {
          //non-premixed
          num_streams = 3;
          if(input_flow_rates.size() != num_streams) {
            if(rank == 0) {
              printf("WARNING: inlet_ratios size does not match combustionRegime expectation (%d, %d).  Using default ratios [0.85, 0.05, 0.1]",input_flow_rates.size(), num_streams);
	    }
            flow_rates.assign(num_streams ,0.0);
            flow_rates[0] = 0.85; flow_rates[1] = 0.05; flow_rates[2] = 0.1;
          } else {
            flow_rates = input_flow_rates;
          }
        }
    } else {
        if(idt_ctrl.getCombustionRegime() == 1) {
          //premixed
          num_streams = 2;
          flow_rates.assign(num_streams, 0.0);
          flow_rates[0] = 0.95; flow_rates[1] = 0.05;
        } else if (idt_ctrl.getCombustionRegime() == 2) {
          //non-premixed
          num_streams = 3;
          flow_rates.assign(num_streams ,0.0);
          flow_rates[0] = 0.85; flow_rates[1] = 0.05; flow_rates[2] = 0.1;
        }
    }
    std::vector<double> running_flow_rates;
    running_flow_rates.assign(num_streams, 0.0);

    // set the initial mass fraction, and temperature
    // based on input streams
    int i_str;
    for(k=0; k<nParticles; k++) {
      i_str = inflow(num_streams, &flow_rates[0], &running_flow_rates[0]);
      if(idt_ctrl.getCombustionRegime() == 1) {
        if (i_str == 0) {
          // set to fuel_air stream
          idt_ctrl.getInitMassFrac(&massFracPtr[k*nState]);
          massFracPtr[k*nState+nSpc]= idt_ctrl.getInitTemp()/idt_ctrl.getRefTemp();
        } else if (i_str == 1) {
          // set to pilot stream
          for(l=0; l<nState; l++)
            massFracPtr[k*nState+l] = equiStatePtr[l];
        }
      } else if (idt_ctrl.getCombustionRegime() == 2) {
        if (i_str == 0) {
          // set to air stream
          idt_ctrl.getOxidMassFrac(&massFracPtr[k*nState]);
          massFracPtr[k*nState+nSpc]= idt_ctrl.getInitTemp()/idt_ctrl.getRefTemp();
         } else if (i_str == 1) {
           // set to fuel stream
           idt_ctrl.getFuelMassFrac(&massFracPtr[k*nState]);
           massFracPtr[k*nState+nSpc]= idt_ctrl.getInitTemp()/idt_ctrl.getRefTemp();
         } else if (i_str == 2) {
           // set to pilot stream
           for(l=0; l<nState; l++)
             massFracPtr[k*nState+l] = equiStatePtr[l];
         }
      }
    }

    if(rank==0) {
      for(i=0; i<nParticles; i++) {
        getTimeHistLine_full(tcurr,&massFracPtr[0],&idt_ctrl,
                             &systemParam,0.0,i,thistLine);
        printf("%s",thistLine);
        fflush(stdout);
        fprintf(thistFilePtr,"%s",thistLine);
        fflush(thistFilePtr);
      }
    }
    startTime=zerork::getHighResolutionTime();
    lastIterTime = startTime;

    // PaSR stuff -- move elsewere and rename appropriately
    double dt_max, dt_sub, tau_mix, tau_pair, tau_res, dt;
    int num_substeps, num_steps;
    tau_res = idt_ctrl.getResidenceTime();
    tau_pair = idt_ctrl.getPairingTime();
    tau_mix = idt_ctrl.getMixingTime();
    systemParam.tau_mix = tau_mix;
    systemParam.ymix = NULL;
    dt_max = 0.1 * min(tau_res, tau_pair);
    dt_sub = 0.04 * tau_mix;
    num_substeps = 1 + int(dt_max/dt_sub);
    num_steps = int(tmax/dt_max);

    double part_out, part_pair;
    part_out = 0.0;
    part_pair = 0.0;
    int npart_out, num_fl_pairs, num_pairs;

    int step_counter = 0;

#ifdef ZERORK_MPI    
    zerork::utilities::mpi_counter particle_counter(MPI_COMM_WORLD, 0);
    std::vector<int> responsible_ranks_root(nParticles, 0);
#endif

    //inflow/outflow
    while(tcurr<tmax)
    {
      // Adjust timestep not to overstep tmax
      if(tcurr + dt_max > tmax) {
        dt = tmax - tcurr;
      } else {
        dt = dt_max;
      }

      double startRank0=zerork::getHighResolutionTime();
      if(rank==0) {
        // Adjust inflow for limited pilot stream
        if(idt_ctrl.getPilotDuration() >= 0.0) {
          if((tcurr > idt_ctrl.getPilotDuration()) && (flow_rates[num_streams-1] > 0.)) {
            double pilot_rate = flow_rates[num_streams-1];
            double pilot_running_rate = running_flow_rates[num_streams-1];
            double sum_rates = 0.0;
            double sum_running_rates = 0.0;
            for(int i = 0; i < (num_streams-1); ++i) {
              flow_rates[i] = flow_rates[i]/(1.0-pilot_rate);
              running_flow_rates[i] = running_flow_rates[i]/(1.0-pilot_running_rate);
              sum_rates += flow_rates[i];
              sum_running_rates += running_flow_rates[i];
            }
            for(int i = 0; i < (num_streams-1); ++i) {
              flow_rates[i] = flow_rates[i]/sum_rates;
              running_flow_rates[i] = running_flow_rates[i]/sum_running_rates;
            }
            flow_rates[num_streams-1] = 0.0;
            running_flow_rates[num_streams-1] = 0.0;
          }
        }

        // Compute particles leaving
        part_out += nParticles*dt/tau_res;
        npart_out = min(int(round(part_out)),nParticles);
        part_out -= npart_out;

        std::vector<int> stream_particle_counts(num_streams,0);
        if(npart_out > 0) {
          // Select pairs of particles for each inflow/outflow
          num_fl_pairs = (npart_out+1)/2;
          select_pairs(nParticles, nState, num_fl_pairs, 0, &massFracPtr[0]);

          // Set alternate particles to inflow properties
          i = nParticles+1;
          for(k=0; k<npart_out; k++) {
            i -= 2;
            i_str = inflow(num_streams, &flow_rates[0], &running_flow_rates[0]);
            stream_particle_counts[i_str] += 1;
            if(idt_ctrl.getCombustionRegime() == 1) {
              if (i_str == 0) {
                // set to fuel_air stream
                idt_ctrl.getInitMassFrac(&massFracPtr[i*nState]);
                massFracPtr[i*nState+nSpc]=
                  idt_ctrl.getInitTemp()/idt_ctrl.getRefTemp();
              } else if (i_str == 1) {
                // set to pilot stream
                for(l=0; l<nState; l++)
                  massFracPtr[i*nState+l] = equiStatePtr[l];
              }
            } else if (idt_ctrl.getCombustionRegime() == 2) {
              if (i_str == 0) {
                // set to air stream
                idt_ctrl.getOxidMassFrac(&massFracPtr[i*nState]);
                massFracPtr[i*nState+nSpc]=
                  idt_ctrl.getInitTemp()/idt_ctrl.getRefTemp();
              } else if (i_str == 1) {
                // set to fuel stream
                idt_ctrl.getFuelMassFrac(&massFracPtr[i*nState]);
                massFracPtr[i*nState+nSpc]=
                  idt_ctrl.getInitTemp()/idt_ctrl.getRefTemp();
              } else if (i_str == 2) {
                // set to pilot stream
                for(l=0; l<nState; l++)
                  massFracPtr[i*nState+l] = equiStatePtr[l];
              }
            }
          }
        } //if npart_out > 0

        printf("# Inflow particle counts [");
        for(int i = 0; i < num_streams; ++i) {
          printf("%d", stream_particle_counts[i]);
          if(i < num_streams-1) {
            printf(", ");
          } else {
            printf("]\n");
          }
        }

        // Perform pairing
        part_pair += 0.5*nParticles*dt/tau_pair;
        num_pairs = min(int(round(part_pair)),nParticles/2);
        part_pair -= num_pairs;

        if(num_pairs>0) {
          select_pairs(nParticles, nState, num_pairs, num_fl_pairs, &massFracPtr[0]);

          // Rotate particles
          int ifst = nParticles - 2*num_pairs + 1;
          double temp_comp;
          for (k=0; k<nState; k++) {
            temp_comp = massFracPtr[(nParticles-1)*nState+k];

            for(i=nParticles-1; i>ifst+2; i -= 2) {
              massFracPtr[i*nState+k] = massFracPtr[(i-2)*nState+k];
            }

            massFracPtr[ifst*nState + k] = temp_comp;
          }
        } //if num_pairs > 1? 0?
      }
      rank0Time+=zerork::getHighResolutionTime() - startRank0;

      double dt_react = dt;
      int state_size = 2*nState;

      if(!idt_ctrl.fully_coupled()) { 
        dt_react = max(dt/num_substeps,1.0e-10);
	state_size = nState;
      }
      std::vector<double> particle_times(nParticles,0);
#ifdef ZERORK_MPI
      double startComm=zerork::getHighResolutionTime();
      MPI_Bcast(&massFracPtr[0], nState*nParticles, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
      std::vector<int> responsible_ranks(nParticles,0);

      if(idt_ctrl.load_balance()) {
        particle_counter.reset();
        i = particle_counter.increment(2);
      } else {
	i = 2*rank;
      }
      commTime += zerork::getHighResolutionTime() - startComm;
      while(i<nParticles-1) {
        responsible_ranks[i] = rank;
        responsible_ranks[i+1] = rank;
#else
      for(i=0; i<nParticles-1; i+=2) {
#endif
	int kloop = 1;
	if(!idt_ctrl.fully_coupled()) {
          kloop = num_substeps;
	}
        for (k=0; k<kloop; k++) {
          //mix-reaction substep
          //for(l=0; l<2*nState; l++)
          //  massFracPtr_local[l] = massFracPtr[i*nState+l];
          double startPart=zerork::getHighResolutionTime();
	  if(!idt_ctrl.fully_coupled()) {
            const double decay = 0.5*(1.0-exp(-2.0*dt_react/tau_mix));
            for(l=0; l<nState; l++) {
              const double delt = decay*(massFracPtr[i*nState+l] - massFracPtr[(i+1)*nState+l]);
              massFracPtr[i*nState+l] -= delt;
              massFracPtr[(i+1)*nState+l] += delt;
            }
	  }

          N_Vector particle_states;
          particle_states = N_VMake_Serial(state_size,&massFracPtr[i*nState]);
          flag = CVodeReInit(cvode_mem, 0.0, particle_states);
          flag = CVode(cvode_mem, dt_react, particle_states, &tnext, CV_NORMAL);
          N_VDestroy_Serial(particle_states);

	  if(!idt_ctrl.fully_coupled()) {
            //N_Vector particle_states;
            particle_states = N_VMake_Serial(state_size,&massFracPtr[(i+1)*nState]);
            flag = CVodeReInit(cvode_mem, 0.0, particle_states);
            flag = CVode(cvode_mem, dt_react, particle_states, &tnext, CV_NORMAL);
            N_VDestroy_Serial(particle_states);
	  }

          double endPart=zerork::getHighResolutionTime();
          particle_times[i] += endPart - startPart;
          particle_times[i+1] += endPart - startPart;
	}
#ifdef ZERORK_MPI
        startComm=zerork::getHighResolutionTime();
        if(idt_ctrl.load_balance()) {
          i = particle_counter.increment(2);
        } else {
          i += 2*nranks;
        }
        commTime += zerork::getHighResolutionTime() - startComm;
#endif
      } // for i<nParticles
#ifdef ZERORK_MPI
      //Send ours back to root
      startComm=zerork::getHighResolutionTime();
      std::vector<double> rank_times(nranks, 0);
      responsible_ranks_root.assign(nParticles, 0);
      MPI_Reduce(&responsible_ranks[0], &responsible_ranks_root[0], nParticles, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
      for(i=0; i<nParticles-1; i+=2) {
        if(rank==0) {
          MPI_Status status;
          int responsible_rank = responsible_ranks_root[i];
          if(responsible_rank > 0) {
            MPI_Recv(&massFracPtr[i*nState], 2*nState, MPI_DOUBLE, responsible_rank, i, MPI_COMM_WORLD, &status);
            MPI_Recv(&particle_times[i], 2, MPI_DOUBLE, responsible_rank, i, MPI_COMM_WORLD, &status);
          }
	  rank_times[responsible_rank] += particle_times[i];
        } else {
          int responsible_rank = responsible_ranks[i];
          if(responsible_rank > 0) {
            MPI_Send(&massFracPtr[i*nState], 2*nState, MPI_DOUBLE, 0, i, MPI_COMM_WORLD);
            MPI_Send(&particle_times[i], 2, MPI_DOUBLE, 0, i, MPI_COMM_WORLD);
          }
        }
      }
      double avg_time = 0.0;
      double max_time = 0.0;
      for(i=0; i<nranks; ++i) {
        avg_time += rank_times[i];
        max_time = std::max(max_time, rank_times[i]);
      }
      avg_time /= nranks;
      commTime += zerork::getHighResolutionTime() - startComm;
#else
      double avg_time = 0.0;
      for(i=0; i<nParticles-1; i+=2) {
        avg_time += particle_times[i];
      }
      double max_time = avg_time;
#endif
      tcurr += dt;

      // if error
      if (check_flag(&flag, "CVODE ERROR", 1)) {
        isBadStep = 1;
        did_cvodeFail++;
        cerr << "EVERYTHING FAILED. IT'S OVER.\n";
        exit(-1);
      } //if cvode error

      step_counter++;

      if(rank == 0) {
        if(step_counter % 1 ==0) {
          for(i=0; i<nParticles; i++) {
            getTimeHistLine_full(tcurr,
                                 &massFracPtr[0],
                                 &idt_ctrl,
                                 &systemParam,
                                 //zerork::getHighResolutionTime()-startTime,
                                 particle_times[i],
                                 i,
                                 thistLine);
            printf("%s",thistLine);
            fflush(stdout);
            fprintf(thistFilePtr,"%s",thistLine);
            fflush(thistFilePtr);
          }

	  printf("#Times: %g, %g, %g, %g, %g\n", zerork::getHighResolutionTime()-startTime, zerork::getHighResolutionTime()-lastIterTime, avg_time, max_time, commTime);
        }
      }
      lastIterTime = zerork::getHighResolutionTime();
    } //while tcurr<tmax

    stopTime=zerork::getHighResolutionTime();
    simTime=stopTime-startTime;
    // add new-line to separate the time histories
    if(rank == 0) {
      printf("\n");
      fprintf(thistFilePtr,"\n");

      idt_ctrl.getInitMoleFrac(&moleFracInit[0]);
      getIdtLine(&moleFracInit[0],&idt_ctrl,idt,simTime,did_cvodeFail,
                 MAX_THIST_LINE,thistLine);
      fprintf(idtFilePtr,"%s",thistLine);
      fflush(idtFilePtr);

      avgTime[idt_ctrl.getThreshId()]+=simTime;
      if(simTime > maxTime[idt_ctrl.getThreshId()])
      {maxTime[idt_ctrl.getThreshId()]=simTime;}
      if(simTime < minTime[idt_ctrl.getThreshId()])
      {minTime[idt_ctrl.getThreshId()]=simTime;}
    }

    idt_ctrl.incrementRunId();
  }

  if(rank ==0) {
    // report the threshold statistics
    /**/
    fprintf(idtFilePtr,"\n\n");
    fprintf(idtFilePtr,"# -----------------------------------------------------------------------------\n");
    fprintf(idtFilePtr,"# Column 1: [-] preconditioner threshold\n");
    fprintf(idtFilePtr,"# Column 2: [s] average simulation time\n");
    fprintf(idtFilePtr,"# Column 3: [s] minimum simulation time\n");
    fprintf(idtFilePtr,"# Column 4: [s] maximum simulation time\n");
    nRunsPerThresh=(double)(idt_ctrl.getRunTotal()/
                            idt_ctrl.getNumThreshRuns());
    for(j=0; j<idt_ctrl.getNumThreshRuns(); j++)
    {
      avgTime[j]/=nRunsPerThresh;
      fprintf(idtFilePtr,"%14.7e  %14.7e  %14.7e  %14.7e\n",
              idt_ctrl.getThreshFromId(j),avgTime[j],minTime[j],maxTime[j]);
    }
    fflush(idtFilePtr);
    /**/

    // clean up procedures
    fclose(idtFilePtr);
    fclose(thistFilePtr);
  }

  //fclose(filePtr);

 // Free ode system parameter memory
  free(systemParam.molWt);
  free(systemParam.invMolWt);
  free(systemParam.netProd);
  free(systemParam.fwdROP);
  free(systemParam.createRate);
  free(systemParam.destroyRate);
  free(systemParam.conc);
  free(systemParam.Energy);
  free(systemParam.CpMass);

  free_Jsparse(systemParam.sparseMtx);

  N_VDestroy_Serial(systemState);
  N_VDestroy_Serial(systemState_local);
  N_VDestroy_Serial(equiState);

  // Free integrator memory

  CVodeFree(&cvode_mem);
#if defined SUNDIALS3 || defined SUNDIALS4
  SUNLinSolFree(LS);
#endif
#if defined SUNDIALS4
  SUNNonlinSolFree(NLS);
#endif


//  printf("Begin.......delete of idt_sweep_params class\n"); fflush(stdout);
//  printf("Finished....delete of idt_sweep_params class\n"); fflush(stdout);
}

int main(int argc, char **argv)
{
  cvReactor(argc, argv);
  exit(0);
}

// getTimeHistLine_full(...)
//
// The purpose of this function is to construct a data line stored in a
// character pointer *thistLine.  The data line is considered 'full' because
// it contains all the ODE solver statistics and complete timing information.
// The data recorded per line is as follows: run id, ode time, temperature,
// pressure, density,  mole fraction of tracked species (i.e. those specified
// in the input file), ODE solver procedure statistics, SuperLU statistics,
// solver timings.
//
// Note that changes to getTimeHistLine_full(...) should be matched with
// getTimeHistHeader_full(...) to ensure that the labels of the data columns
// are consistent.
//
// TODO: Currently, there is no check of the array size so it is possible to
// overwrite the data, which needs to be fixed.
void getTimeHistLine_full(const double currTime,
                          const double *sysState,
                          const idt_sweep_params *idt_ctrl,
                          const cv_param *cvp,
                          const double wallTime,
                          const int ip,
                          char *thistLine)
{
  int j;
  int nSpc   = cvp->mech->getNumSpecies();
  int nState = cvp->nState;
  int nTrack = idt_ctrl->getNumTrackSpecies();
  int strPos,strLen;
  double currPres,currTemp, initTemp;
  double *currMoleFrac;
  long int nsteps,netfails,nniters,nncfails;

  double molWt, cpMixPerMass, intEnergyMixPerMass, intEnergyMixPerMassInit;

  nsteps=netfails=nniters=nncfails=0;

  CVodeGetNumSteps(cvp->cvodeMemPtr, &nsteps);
  CVodeGetNumErrTestFails(cvp->cvodeMemPtr, &netfails);
  CVodeGetNumNonlinSolvIters(cvp->cvodeMemPtr, &nniters);
  CVodeGetNumNonlinSolvConvFails(cvp->cvodeMemPtr, &nncfails);

  currMoleFrac = new double[nSpc];

  initTemp = idt_ctrl->getInitTemp();
  currTemp = sysState[ip*nState+nSpc]*idt_ctrl->getRefTemp();
  currPres = idt_ctrl->getInitPres();

  double dens = idt_ctrl->getDensity(currTemp, currPres, &sysState[ip*nState]);


  molWt=cvp->mech->getMolWtMixFromY(&sysState[ip*nState]);
  // currMoleFrac[] used as temporary storage for species specific heat
  cpMixPerMass=cvp->mech->getMassCpFromTY(currTemp,
                                          &sysState[ip*nState],
                                          &currMoleFrac[0]);
  // currMoleFrac[] used as temporary storage for species internal energy
  intEnergyMixPerMass=cvp->mech->getMassIntEnergyFromTY(currTemp,
                                                        &sysState[ip*nState],
                                                        &currMoleFrac[0]);
  // currMoleFrac[] used as temporary storage for species internal energy
  intEnergyMixPerMassInit=cvp->mech->getMassIntEnergyFromTY(initTemp,
                                                            &sysState[ip*nState],
                                                            &currMoleFrac[0]);
  // currMoleFrac[] is recalculated with mole fraction
  cvp->mech->getXfromY(&sysState[ip*nState],currMoleFrac);

  // sprintf return the number of characters written excluding the '\0'
  // string termination character
  strLen=sprintf(thistLine,
                 "%8d  %8d  %14.7e  %14.7e  %14.7e  %14.7e  %14.7e  %14.7e  %14.7e",
                 idt_ctrl->getRunId(),
                 ip,
                 currTime,
                 currTemp,
                 currPres,
                 dens,
                 molWt,
                 cpMixPerMass,
                 intEnergyMixPerMass-intEnergyMixPerMassInit);
  strPos=strLen;
  for(j=0; j<nTrack; j++)
  {
    strLen=sprintf(&thistLine[strPos],
                   "  %14.7e",
                   currMoleFrac[idt_ctrl->getTrackSpeciesId(j)]);
    strPos+=strLen;
  }

  strLen=sprintf(&thistLine[strPos],
                 "  %6ld  %6ld  %6ld  %6ld  %6d  %6d  %6d  %6d  %6d  %6d  %6d  %6d",
                 nsteps,
                 netfails,
                 nniters,
                 nncfails,
                 cvp->nJacSetup,
                 cvp->nColPerm,
                 cvp->nJacFactor,
                 cvp->nBackSolve,
                 cvp->nFunc,
                 //cvp->sparseMtx->nNonZero,
                 cvp->sparseMtx->nNonZeroMix,
                 //cvp->sparseMtx->reduceNNZ,
                 cvp->sparseMtx->reduceNNZMix,
                 cvp->sparseMtx->LUnnz);
  strPos+=strLen;
  strLen=sprintf(&thistLine[strPos],
                 "  %14.7e  %14.7e  %14.7e  %14.7e  %14.7e  %14.7e\n",
                 cvp->jacSetupTime,
                 cvp->colPermTime,
                 cvp->jacFactorTime,
                 cvp->backsolveTime,
                 cvp->funcTime,wallTime);

  delete [] currMoleFrac;
}

// getTimeHistHeader_full(...)
//
// The purpose of this function is to construct a descriptive header stored in
// a character pointer *thistHead.  The header provides labels of each column
// written by getTimeHistLine_full(...).  The header and data line are
// considered 'full' because they contain all the ODE solver statistics and
// complete timing information. The data recorded per line is as follows:
// run id, ode time, temperature, pressure, density,  mole fraction of tracked
// species (i.e. those specified in the input file), ODE solver procedure
// statistics, SuperLU statistics, solver timings.
//
// Note that changes to getTimeHistLine_full(...) should be matched with
// getTimeHistHeader_full(...) to ensure that the labels of the data columns
// are consistent.
//
// TODO: Currently, there is no check of the array size so it is possible to
// overwrite the data, which needs to be fixed.
void getTimeHistHeader_full(const idt_sweep_params *idt_ctrl,
                            const cv_param *cvp,
                            char *thistHead)
{
  int j;
  int nSpc   = cvp->mech->getNumSpecies();
  int nTrack = idt_ctrl->getNumTrackSpecies();
  int strPos,strLen, colNum;

  strPos=0;
  strLen=sprintf(&thistHead[strPos],
                 "%s%s%s%s%s%s%s%s%s",
                 "# Column  1: [#]       run id\n",
                 "# Column  2: [#]       particle id\n",
                 "# Column  3: [s]       simulated ODE time\n",
                 "# Column  4: [K]       temperature\n",
                 "# Column  5: [Pa]      pressure\n",
                 "# Column  6: [kg/m^3]  density\n",
                 "# Column  7: [kg/kmol] mixture molecular weight\n",
                 "# Column  8: [J/kg/K]  mixture specific heat (cosnt vol)\n",
                 "# Column  9: [J/kg]    energy released\n");
  strPos+=strLen;
  colNum=10;
  for(j=0; j<nTrack; j++) {
    strLen=sprintf(&thistHead[strPos],
                   "# Column %2d: [-]      mole fraction of %s\n",
                   colNum++, // increment after printf
                   cvp->mech->getSpeciesName(idt_ctrl->getTrackSpeciesId(j)));
    strPos+=strLen;
  }
  strLen=sprintf(&thistHead[strPos],
                 "# Column %2d: [#]      CVodeGetNumSteps(...)\n",
                 colNum++);
  strPos+=strLen;
  strLen=sprintf(&thistHead[strPos],
                 "# Column %2d: [#]      CVodeGetNumErrTestFails(...)\n",
                 colNum++);
  strPos+=strLen;
  strLen=sprintf(&thistHead[strPos],
                 "# Column %2d: [#]      CVodeGetNumNonlinSolvIters(...)\n",
                 colNum++);
  strPos+=strLen;
  strLen=sprintf(&thistHead[strPos],
                 "# Column %2d: [#]      CVodeGetNumNonlinSolvConvFails(...)\n",
                 colNum++);
  strPos+=strLen;
  strLen=sprintf(&thistHead[strPos],
                 "# Column %2d: [#]      calls to the precond. setup\n",
                 colNum++);
  strPos+=strLen;
  strLen=sprintf(&thistHead[strPos],
                 "# Column %2d: [#]      calls to the column permutation\n",
                 colNum++);
  strPos+=strLen;
  strLen=sprintf(&thistHead[strPos],
                 "# Column %2d: [#]      calls to the precond. factorization\n",
                 colNum++);
  strPos+=strLen;
  strLen=sprintf(&thistHead[strPos],
                 "# Column %2d: [#]      calls to the precond. back solve\n",
                 colNum++);
  strPos+=strLen;
  strLen=sprintf(&thistHead[strPos],
                 "# Column %2d: [#]      calls to the RHS function\n",
                 colNum++);
  strPos+=strLen;
  strLen=sprintf(&thistHead[strPos],
                 "# Column %2d: [#]      max non-zero terms in precond.\n",
                 colNum++);
  strPos+=strLen;
  strLen=sprintf(&thistHead[strPos],
                 "# Column %2d: [#]      current non-zero terms in precond.\n",
                 colNum++);
  strPos+=strLen;
  strLen=sprintf(&thistHead[strPos],
                 "# Column %2d: [#]      current non-zero terms in LU factors\n",
                 colNum++);
  strPos+=strLen;
  strLen=sprintf(&thistHead[strPos],
                 "# Column %2d: [s]      wall clock time - precond. setup\n",
                 colNum++);
  strPos+=strLen;
  strLen=sprintf(&thistHead[strPos],
                 "# Column %2d: [s]      wall clock time - column permut.\n",
                 colNum++);
  strPos+=strLen;
  strLen=sprintf(&thistHead[strPos],
                 "# Column %2d: [s]      wall clock time - precond. factor.\n",
                 colNum++);
  strPos+=strLen;
  strLen=sprintf(&thistHead[strPos],
                 "# Column %2d: [s]      wall clock time - precond. back solve\n",
                 colNum++);
  strPos+=strLen;
  strLen=sprintf(&thistHead[strPos],
                 "# Column %2d: [s]      wall clock time - RHS evaluation\n",
                 colNum++);
  strPos+=strLen;
  strLen=sprintf(&thistHead[strPos],
                 "# Column %2d: [s]      wall clock time - total simulation\n",
                 colNum++);
  strPos+=strLen;
  strLen=sprintf(&thistHead[strPos],
                 "# -----------------------------------------------------------------------------\n");
  strPos+=strLen;
  /*
  if(idt_ctrl->longOutput()) {
    strLen=sprintf(&thistHead[strPos], "# run id part id  %24s  %24s  %24s  %24s  %24s  %24s  %24s",
                            "sim time","temp", "press", "dens","mol wt", "cp", "E release");
    strPos+=strLen;
    for(j=0; j<nTrack; j++) {
      strLen=sprintf(&thistHead[strPos],
                     " mlfrc%20s",
                     cvp->mech->getSpeciesName(idt_ctrl->getTrackSpeciesId(j)));
      strPos+=strLen;
    }
  } else {
  */
    strLen=sprintf(&thistHead[strPos],
                   "# run id        part id        sim time            temp           press            dens          mol wt              cp       E release");
    strPos+=strLen;
    for(j=0; j<nTrack; j++) {
      strLen=sprintf(&thistHead[strPos],
                     " mlfrc%10s",
                     cvp->mech->getSpeciesName(idt_ctrl->getTrackSpeciesId(j)));
      strPos+=strLen;
    }
    //}
  strLen=sprintf(&thistHead[strPos],
                 "   nstep   nertf  nnlsit  nnlcvf prsetup ncolprm nfactor backslv  intfcl     nnz    rnnz   lunnz    jac setup tm     col perm tm      jac fac tm    backsolve tm     rhs func tm   simulation tm\n");
  strPos+=strLen;
}

// void getSimHeader_full(...)
//
//
int getSimHeader_full(const int inp_argc,
                      char **inp_argv,
                      const idt_sweep_params *idt_ctrl,
                      const cv_param *cvp,
                      const int maxLen,
                      char *simHead)
{
  int j,spcId;
  int nSpc   = cvp->mech->getNumSpecies();
  int nSpcTermsJac = cvp->sparseMtx->nNonZero-2*nSpc+1;
  int nTrack = idt_ctrl->getNumTrackSpecies();
  int strPos,strLen;

  // 1. construct the command line
  strPos=0;
  strLen=sprintf(&simHead[strPos],
                 "# Command line: %s",inp_argv[0]);
  strPos+=strLen;

  for(j=1; j<inp_argc; j++) {
    if(j < inp_argc-1) {
      strLen=sprintf(&simHead[strPos],
                     " %s",
                     inp_argv[j]);
    }
    else {
      strLen=sprintf(&simHead[strPos],
                     " %s\n%s\n",
                     inp_argv[j],
                     "# ---------------------------------------------------------------------------");
    }
    strPos+=strLen;
  }

  {
    zerork::ZeroRKConstants zrk_constants;
    // 1.5 construct git commit information
    strLen=sprintf(&simHead[strPos],
                   "%s%s%s\n%s%s\n%s%s\n%s\n",
                   "# Zero-RK git commit information:\n",
                   "#   commit id                 [hash] : ",
                   zrk_constants.GetCommitId(),
                   "#   commit time-stamp         [date] : ",
                   zrk_constants.GetCommitTimestamp(),
                   "#   branch name               [word] : ",
                   zrk_constants.GetBranchName(),
                   "# ---------------------------------------------------------------------------");
    strPos+=strLen;
  }


  // 2. construct the mechanism information
  strLen=sprintf(&simHead[strPos],
                 "%s%s%s\n%s%s\n%s%s\n%s%d\n%s%d\n%s%d\n%s%d\n%s%d\n%s\n",
                 "# Mechanism constants:\n",
                 "#   mechanism definition      [path] : ",
                 idt_ctrl->getMechFileName(),
                 "#   thermodynamics definition [path] : ",
                 idt_ctrl->getThermFileName(),
                 "#   parser log definition     [path] : ",
                 idt_ctrl->getLogFileName(),
                 "#   number of species            [#] : ",
                 nSpc,
                 "#   number of reactions          [#] : ",
                 cvp->mech->getNumReactions(),
                 "#   number of 1-way steps        [#] : ",
                 cvp->mech->getNumSteps(),
                 "#   max non-zeros in species J   [#] : ",
                 nSpcTermsJac,
                 "#   max non-zeros in const vol J [#] : ",
                 cvp->sparseMtx->nNonZero,
                 "# ---------------------------------------------------------------------------");
  strPos+=strLen;
  // 3. construct the composition information
  strLen=sprintf(&simHead[strPos],
                 "%s\n",
                 "# Fuel composition (moles of species)/(moles of fuel):");
  strPos+=strLen;
  for(j=0; j<idt_ctrl->getNumFuelSpecies(); j++) {
    spcId = idt_ctrl->getFuelSpeciesId(j);
    strLen=sprintf(&simHead[strPos],
                   "#   %16s: %.18g\n",
                   cvp->mech->getSpeciesName(spcId),
                   idt_ctrl->getFuelMoleFrac(spcId));
    strPos+=strLen;
  }
  strLen=sprintf(&simHead[strPos],
              "%s\n",
              "# Oxid(izer) composition (moles of species)/(moles of oxid):");
  strPos+=strLen;
  for(j=0; j<idt_ctrl->getNumOxidSpecies(); j++) {
    spcId = idt_ctrl->getOxidSpeciesId(j);
    strLen=sprintf(&simHead[strPos],
                   "#   %16s: %.18g\n",
                   cvp->mech->getSpeciesName(spcId),
                   idt_ctrl->getOxidMoleFrac(spcId));
    strPos+=strLen;
  }

  strLen=sprintf(&simHead[strPos],
                 "%s\n%s%.18g\n%s%.18g\n%s%.18g\n%s\n",
                 "# Global stoichiometry constants:",
                 "#   fuel (excess mols of atomic-O)/(mols of fuel) : ",
                 idt_ctrl->getFuelOxyBalance(),
                 "#   oxid (excess mols of atomic-O)/(mols of oxid) : ",
                 idt_ctrl->getOxidOxyBalance(),
                 "#   (mols of oxid)/(mols of fuel) at phi=1        : ",
                 idt_ctrl->getMoleOxidStoicPerFuel(),
                 "# ---------------------------------------------------------------------------");
  strPos+=strLen;


  // 4. construct the simulation information
  strLen=sprintf(&simHead[strPos],
                 "%s%s%s\n%s%.18g\n%s%.18g\n%s%d\n%s\n",
                 "# Simulation constants:\n",
                 "#   continue after IDT found   [y/n] : ",
                 (idt_ctrl->continueAfterIDT() != 0) ? "yes" : "no",
                 "#   reference temperature        [K] : ",
                 idt_ctrl->getRefTemp(),
                 "#   ignition delay delta Temp    [K] : ",
                 idt_ctrl->getDeltaTign(),
                 "#   total number of runs         [#] : ",
                 idt_ctrl->getRunTotal(),
                 "# ---------------------------------------------------------------------------");
  strPos+=strLen;

  // 5. construct the solver information
  strLen=sprintf(&simHead[strPos],
                 "%s%s%.18g\n%s%.18g\n%s%.18g\n%s%d\n%s%.18g\n%s%.18g\n%s%d\n%s%d\n%s%.18g\n%s%d\n%s%s\n%s%s\n%s%.18g\n%s%s\n%s%s\n%s\n",
                 "# Solver constants:\n",
                 "#   ode relative tolerance            [-] : ",
                 idt_ctrl->getRTol(),
                 "#   ode absolute tolerance            [-] : ",
                 idt_ctrl->getATol(),
                 "#   ode max internal timestep         [s] : ",
                 idt_ctrl->getMaxInternalDt(),
                 "#   ode max internal steps            [#] : ",
                 idt_ctrl->getMaxInternalSteps(),
                 "#   linear solver rel error           [-] : ",
                 idt_ctrl->getEpsLin(),
                 "#   non-linear solver rel error       [-] : ",
                 idt_ctrl->getNlConvCoeff(),
                 "#   cvode primary failures allowed    [#] : ",
                 idt_ctrl->getMaxPrimaryCvodeFails(),
                 "#   cvode secondary failures allowed  [#] : ",
                 idt_ctrl->getMaxSecondaryCvodeFails(),
                 "#   secondary failure P threshold     [-] : ",
                 idt_ctrl->getSafetyThreshold(),
                 "#   max Krylov dimension              [#] : ",
                 idt_ctrl->getKrylovDim(),
                 "#   use fake P update               [y/n] : ",
                 (idt_ctrl->getUpdate() ? "yes" : "no"),
                 "#   use incomplete LU factorization [y/n] : ",
                 (idt_ctrl->getILU() ? "yes" : "no"),
                 "#   partial pivot threshold           [-] : ",
                 idt_ctrl->getPartialPivotThresh(),
                 "#   P threshold type code             [#] : ",
                 precondThresholdName[idt_ctrl->getThreshType()-1],
                 "#   permutation type code             [#] : ",
                 permutationTypeName[idt_ctrl->getPermutationType()-1],
                 "# ---------------------------------------------------------------------------");
  if(strPos >= maxLen) {
    printf("ERROR: the simulation header info (%d bytes)\n",strPos);
    printf("       is larger than the buffer size (%d bytes).\n",maxLen);
    printf("       Exiting Now!\n");
    exit(-1);
  }

  return strPos;
}

int getIdtLine(const double initMoleFrac[],
               const idt_sweep_params *idt_ctrl,
               const double idt,
               const double runTime,
               const int nCvodeFails,
               const int maxBuffer,
               char *idtLine)
{
  int j;
  int nTrack = idt_ctrl->getNumTrackSpecies();
  int strPos,strLen;

  strPos=0;
  strLen=sprintf(&idtLine[strPos],
                 "%8d  %23.16e  %23.16e  %23.16e  %23.16e  %23.16e  %23.16e  %23.16e  %23.16e  %14d",
                 idt_ctrl->getRunId(),
                 idt_ctrl->getInitTemp(),
                 idt_ctrl->getInitPres(),
                 idt_ctrl->getPhi(),
                 idt_ctrl->getEgr(),
                 idt_ctrl->getThresh(),
                 idt_ctrl->getInitDensity(),
                 idt,
                 runTime,
                 nCvodeFails);
  strPos+=strLen;

  for(j=0; j<nTrack; j++) {
    strLen=sprintf(&idtLine[strPos],
                   "   %23.16e",
                   initMoleFrac[idt_ctrl->getTrackSpeciesId(j)]);
    strPos+=strLen;
  }
  strLen=sprintf(&idtLine[strPos],"\n");
  strPos+=strLen;

  if(strPos >= maxBuffer) {
    printf("ERROR: the IDT file data line (%d bytes)\n",strPos);
    printf("       is larger than the buffer size (%d bytes).\n",maxBuffer);
    printf("       Exiting Now!\n");
    exit(-1);
  }

  return strPos;
}
int getIdtHeader(const idt_sweep_params *idt_ctrl,
                 const cv_param *cvp,
                 const int maxBuffer,
                 char *idtHeader)
{
  int j;
  int nTrack = idt_ctrl->getNumTrackSpecies();
  int strPos,strLen;

  strPos=0;
  strLen=sprintf(&idtHeader[strPos],
                 "%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n",
                 "# Column  1: [#]      run id",
                 "# Column  2: [K]      initial temperature",
                 "# Column  3: [Pa]     initial pressure",
                 "# Column  4: [-]      equivalence ratio",
                 "# Column  5: [-]      EGR fraction",
                 "# Column  6: [-]      preconditioner threshold",
                 "# Column  7: [kg/m^3] initial density",
                 "# Column  8: [s]      ignition delay time (defined delta T)",
                 "# Column  9: [s]      execution wall clock time",
                 "# Column 10: [#]      number of cvode errors");
  strPos+=strLen;

  for(j=0; j<nTrack; j++) {
    strLen=sprintf(&idtHeader[strPos],
                   "# Column %2d: [-]      initial mole frac of %s\n",
                   j+11,
                   cvp->mech->getSpeciesName(idt_ctrl->getTrackSpeciesId(j)));
    strPos+=strLen;
  }
  strLen=sprintf(&idtHeader[strPos],"# ----------------------------------------------------------------------------\n");
  strPos+=strLen;

  strLen=sprintf(&idtHeader[strPos],
                 "%7s |%23s |%23s |%23s |%23s |%23s |%23s |%23s |%23s |%14s |",
                 "# run id",
                 "Temp [K]",
                 "Pres [Pa]",
                 "phi [-]",
                 "EGR frac [-]",
                 "P thresh [-]",
                 "dens [kg/m3]",
                 "IDT [s]",
                 "sim time [s]",
                 "cvode err [#]");
  strPos+=strLen;

  for(j=0; j<nTrack; j++) {
    strLen=sprintf(&idtHeader[strPos],
                   " mlfrc %17s |",
                   cvp->mech->getSpeciesName(idt_ctrl->getTrackSpeciesId(j)));
    strPos+=strLen;
  }
  strLen=sprintf(&idtHeader[strPos],"\n");
  strPos+=strLen;

  if(strPos >= maxBuffer) {
    printf("ERROR: the IDT file header info (%d bytes)\n",strPos);
    printf("       is larger than the buffer size (%d bytes).\n",maxBuffer);
    printf("       Exiting Now!\n");
    exit(-1);
  }

  return strPos;
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


int inflow(const int num_streams, const double flow_rates[], double running_flow_rates[])
{
  double sum_flows, fl_max;
  int i, i_inflow;

  i_inflow = 0;
  sum_flows = 0.0;
  fl_max = 0.0;

  for(i=0; i<num_streams; i++) {
    running_flow_rates[i] += flow_rates[i];
    sum_flows += flow_rates[i];

    if(running_flow_rates[i] > fl_max) {
      fl_max = running_flow_rates[i];
      i_inflow = i;
    }
  }

  running_flow_rates[i_inflow] -= sum_flows;
  return i_inflow;
}

void mix(const int num_particles,
        const int nstate,
        const double dt,
        const double tau_mix,
        double *state)
{
  double decay, delt;
  int j,k;

  decay = 0.5*(1.0-exp(-2.0*dt/tau_mix));

  for(k=0; k<nstate; k++) {
    for(j=0; j<num_particles-1; j +=2) {
      delt = decay*(state[j*nstate+k] - state[(j+1)*nstate+k]);
      state[j*nstate+k] -= delt;
      state[(j+1)*nstate+k] += delt;
    }
  }

}


void select_pairs(const int num_particles,
                  const int nstate,
                  const int num_pairs,
                  const int num_skip,
                  double *state)
{
  int num_half, ipair, iplast, ip, i, j, k, jl, il, jt;
  double temp;

  std::vector<double> random_vec(num_particles);
  // set a random seed?
  for(i=0; i<num_particles; i++) {
    random_vec[i] = rand() / (float)RAND_MAX;
  }

  num_half = num_particles/2;

  for(ipair=0; ipair<num_pairs; ipair++) {

    // select random pair (i,j=i+1)
    iplast = num_half - ipair - 1;
    ip = random_vec[ipair]*iplast;
    i = 2*ip;
    j = i+1;

    il = 2*iplast;
    jl = il+1;

    // commute pair at random
    if(random_vec[ipair+num_pairs] > 0.5) {
      jt=jl;
      jl=il;
      il=jt;
    }

    // commute (i,j) and (il,jl)
    for (k=0; k<nstate; k++) {
      temp = state[i*nstate+k];
      state[i*nstate+k] = state[il*nstate+k];
      state[il*nstate+k] = temp;

      temp = state[j*nstate+k];
      state[j*nstate+k] = state[jl*nstate+k];
      state[jl*nstate+k] = temp;
    }

  }

}
