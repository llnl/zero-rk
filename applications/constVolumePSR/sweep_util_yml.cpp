#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sweep_util_yml.h"

#define UNMARKED 0x0000DEADBEEF0000

idt_sweep_params::idt_sweep_params(char *inputFileName)
  :
      idt_sweep_IFP(inputFileName)
{
  int j,k;

  // read in the filenames
//  mechFileName  = this->mechFile().c_str();
//  thermFileName = this->thermFile().c_str();
//  logFileName   = this->logFile().c_str();
//  idtFileName   = this->idtFile().c_str();
//  thistFileName = this->thistFile().c_str();

  gasMech = new zerork::mechanism(this->mechFile().c_str(),this->thermFile().c_str(),this->logFile().c_str());

  // ----- parse the fuel and oxygen mixtures -----
  nSpc=gasMech->getNumSpecies();
  fuelMoleFrac = std::vector<double>(nSpc);
  oxidMoleFrac = std::vector<double>(nSpc);
  traceMoleFrac = std::vector<double>(nSpc);
  fullMoleFrac = std::vector<double>(nSpc);

  // set the initial mole fractions to a large negative number to identify
  // species declared with zero mole fraction for tracking purposes
  for(j=0; j<nSpc; j++)
  {fuelMoleFrac[j]=oxidMoleFrac[j]=-1.0e300;fullMoleFrac[j]=-1.0e300;} // use this to identify

  fuelSpcIdx = std::vector<int>();
  oxidSpcIdx = std::vector<int>();
  traceSpcIdx = std::vector<int>();
  fullSpcIdx = std::vector<int>();
  getFracsFromCompMap(this->fuel_mole_fracs(),fuelMoleFrac,&nFuelSpc,fuelSpcIdx);
  getFracsFromCompMap(this->oxidizer_mole_fracs(),oxidMoleFrac,&nOxidSpc,oxidSpcIdx);
  getFracsFromCompMap(this->trace_mole_fracs(),traceMoleFrac,&nTraceSpc,traceSpcIdx);
  getFracsFromCompMap(this->full_mole_fracs(),fullMoleFrac,&nFullSpc,fullSpcIdx);

  trackSpcIdx = std::vector<int>();
  if(this->print_all_fractions()) {
    for(j=0; j<nSpc; j++) {
       trackSpcIdx.push_back(j);
    }
  } else {
    if(this->tracked_species_names().size() > 0) {
        for(j = 0; j < this->tracked_species_names().size(); ++j) {
            std::string spcName = this->tracked_species_names()[j];
            int spcIdx = gasMech->getIdxFromName(spcName.c_str());
            if(spcIdx == -1) {
              printf("WARNING: tracked_species %s not found in mechanism. Ignoring\n",spcName.c_str());
            } else {
              trackSpcIdx.push_back(spcIdx);
            }
        }
    }
  }
  for(j=0; j<nSpc; j++)
  {
      if(this->tracked_species_names().size() == 0) {
          if(fuelMoleFrac[j] != UNMARKED || oxidMoleFrac[j] != UNMARKED || traceMoleFrac[j] != UNMARKED || fullMoleFrac[j] != UNMARKED)
          {
               trackSpcIdx.push_back(j);
          }
      }
      if(fuelMoleFrac[j] == UNMARKED) fuelMoleFrac[j] = 0.0;
      if(oxidMoleFrac[j] == UNMARKED) oxidMoleFrac[j] = 0.0;
      if(traceMoleFrac[j] == UNMARKED) traceMoleFrac[j] = 0.0;
      if(fullMoleFrac[j] == UNMARKED) fullMoleFrac[j] = 0.0;
  }
  nTrackSpc=trackSpcIdx.size();

  normalize(fuelMoleFrac);
  normalize(oxidMoleFrac);
  // N.B. trace species are NOT normalized
  totalTraceMoleFraction = 0.0;
  for(j=0; j<nSpc; j++) {
    totalTraceMoleFraction += traceMoleFrac[j];
  }

  // at this point the fuel and oxidizer mole fraction compositions should
  // be properly normalized
  fuelOxygenBal=gasMech->getMolarAtomicOxygenRemainder(&fuelMoleFrac[0]);
  oxidOxygenBal=gasMech->getMolarAtomicOxygenRemainder(&oxidMoleFrac[0]);
  moleOxidStoicPerFuel=fabs(fuelOxygenBal)/oxidOxygenBal;

  // declare the remaining mole and mass fraction arrays for setting the
  // initial composition
  freshMoleFrac   = std::vector<double>(nSpc);
  freshMassFrac   = std::vector<double>(nSpc);
  exhaustMoleFrac = std::vector<double>(nSpc);
  exhaustMassFrac = std::vector<double>(nSpc);
  initMoleFrac    = std::vector<double>(nSpc);
  initMassFrac    = std::vector<double>(nSpc);
  // ----- end of parsing the fuel and oxygen mixtures -----

  // read simulation constants (non-sweep variables)
  //isConstPres=readFileLine_first_int(inputFptr);
  //if(isConstPres != 0)
  //  {
  //    printf("WARNING: at this time constant pressure IDT is not set up\n");
  //    printf("         calculation will be performed as constant volume\n");
  //    isConstPres=0;
  //  }
  stopTime           = this->stop_time();
  printTime          = this->print_time();
  maxInternalDt      = this->max_internal_dt();
  maxInternalSteps   = this->max_internal_steps();
  rtol               = this->relative_tolerance();
  atol               = this->absolute_tolerance();
  maxCvodeFails1     = this->max_cvode_fails1();
  maxCvodeFails2     = this->max_cvode_fails2();
  safetyThresh       = this->safety_threshold();
  Tref               = this->reference_temperature();
  isEnergyEnabled    = this->energy_enabled();
  Kpressure          = this->pressure_controller_coefficient();

  // read sweep parameters
  initTemp = this->initial_temperatures();
  nTempRuns = initTemp.size();

  initPres = this->initial_pressures();
  nPresRuns = initPres.size();

  phi = this->initial_phis();
  nPhiRuns = phi.size();

  egr = this->initial_egrs();
  nEgrRuns = egr.size();

  residenceTimes = this->residence_times();
  nResidRuns = residenceTimes.size();

  thresh = this->preconditioner_thresholds();
  nThreshRuns = thresh.size();

  //readFileLine_first_int(inputFptr);
  krylovDim = this->max_krylov_dimensions();
  nKrylovRuns = krylovDim.size();

  //RAW: New options
  doILU = this->incomplete_LU();
  fakeUpdate = this->fake_update();
  threshType = this->threshold_type();
  partialPivotThresh = this->partial_pivot_threshold();
  epsLin = this->eps_lin();
  nlConvCoeff = this->nonlinear_convergence_coefficient();
  oneStepMode = this->one_step_mode();
  permutationType = this->permutation_type();
  doDumpJacobian = this->dump_jacobian();
  printNetProdRates = this->print_net_production_rates();
  printNetROP = this->print_net_rates_of_progress();

  // set the sweep counters and the initial composition of the first run
  runTotal = (nTempRuns*nPresRuns*nPhiRuns*
             nEgrRuns*nThreshRuns*nKrylovRuns*nResidRuns);
  tempId=presId=phiId=egrId=threshId=krylovId=residId=runId=0;
  if(this->full_mole_fracs().size() != 0) {
    printf("# USING full_mole_fracs\n");
    for(j=0; j<nSpc; ++j) {
      exhaustMoleFrac[j]=0.0;
      exhaustMassFrac[j]=0.0;
      initMoleFrac[j] = fullMoleFrac[j];
    }
    normalize(initMoleFrac);
    gasMech->getYfromX(&initMoleFrac[0],&initMassFrac[0]);
  } else {
    setInitialComp(phi[phiId],egr[egrId]);
  }
}

idt_sweep_params::~idt_sweep_params()
{
  delete gasMech;
}

void idt_sweep_params::setInitialComp(const double phi, const double egr)
{
  int j;

  // compute the fresh mixture mole fraction based on phi and the number
  // of moles of oxidizer needed for stoichiometry per mole of fuel
  for(j=0; j<nSpc; j++)
  {
    freshMoleFrac[j]=phi*fuelMoleFrac[j]
      +moleOxidStoicPerFuel*oxidMoleFrac[j];
  }
  normalize(freshMoleFrac);
  gasMech->getYfromX(&freshMoleFrac[0],&freshMassFrac[0]);

  if(egr > 0.0)
  {
    // get the ideal exhaust molar composition
    gasMech->getMolarIdealExhaust(&freshMoleFrac[0],&exhaustMoleFrac[0]);

    // convert the exhaust mole fractions to mass fractions
    gasMech->getYfromX(&exhaustMoleFrac[0], &exhaustMassFrac[0]);

    // create the initial mass fraction of the blended intake composition
    // here egr represents the fraction by mass the ideal exhaust composition
    // is in the intake composition
    for(j=0; j<nSpc; j++)
    {initMassFrac[j]=(1.0-egr)*freshMassFrac[j]+egr*exhaustMassFrac[j];}
    // convert the initial intake composition to mole fraction
    gasMech->getXfromY(&initMassFrac[0],&initMoleFrac[0]);
  }
  else
  {
    // no egr
    for(j=0; j<nSpc; j++)
    {
      exhaustMoleFrac[j]=0.0;
      exhaustMassFrac[j]=0.0;
      initMoleFrac[j]=freshMoleFrac[j];
      initMassFrac[j]=freshMassFrac[j];
    }
  }

  //Add in trace species if appropriate
  if(totalTraceMoleFraction > 0.0) {
    for(j=0; j<nSpc; ++j) {
      initMoleFrac[j] *= (1.0-totalTraceMoleFraction);
      initMoleFrac[j] += traceMoleFrac[j];
    }
    normalize(initMoleFrac);
    gasMech->getYfromX(&initMoleFrac[0],&initMassFrac[0]);
  }
}

void idt_sweep_params::printInitialMoleComp() const
{
  int j;
  printf("# Fuel:\n");
  for(j=0; j<nTrackSpc; j++)
    {
      printf("#      mole of %16s (id = %5d): %.18g\n",
	     gasMech->getSpeciesName(trackSpcIdx[j]),trackSpcIdx[j],
	     fuelMoleFrac[trackSpcIdx[j]]);
    }
  printf("# Oxidizer:\n");
  for(j=0; j<nTrackSpc; j++)
    {
      printf("#      mole of %16s (id = %5d): %.18g\n",
	     gasMech->getSpeciesName(trackSpcIdx[j]),trackSpcIdx[j],
	     oxidMoleFrac[trackSpcIdx[j]]);
    }
  printf("# Trace:\n");
  for(j=0; j<nTrackSpc; j++)
    {
      printf("#      mole of %16s (id = %5d): %.18g\n",
	     gasMech->getSpeciesName(trackSpcIdx[j]),trackSpcIdx[j],
	     traceMoleFrac[trackSpcIdx[j]]);
    }
  printf("# Fresh:\n");
  for(j=0; j<nTrackSpc; j++)
    {
      printf("#      mole of %16s (id = %5d): %.18g\n",
	     gasMech->getSpeciesName(trackSpcIdx[j]),trackSpcIdx[j],
	     freshMoleFrac[trackSpcIdx[j]]);
    }
  printf("# Ideal exhaust:\n");
  for(j=0; j<nTrackSpc; j++)
    {
      printf("#      mole of %16s (id = %5d): %.18g\n",
	     gasMech->getSpeciesName(trackSpcIdx[j]),trackSpcIdx[j],
	     exhaustMoleFrac[trackSpcIdx[j]]);
    }
  printf("# Initial:\n");
  for(j=0; j<nTrackSpc; j++)
    {
      printf("#      mole of %16s (id = %5d): %.18g\n",
	     gasMech->getSpeciesName(trackSpcIdx[j]),trackSpcIdx[j],
	     initMoleFrac[trackSpcIdx[j]]);
    }
}

void idt_sweep_params::incrementRunId()
{
  tempId++;
  if(tempId==nTempRuns)
    {tempId=0; presId++;}
  if(presId==nPresRuns)
    {presId=0; phiId++;}
  if(phiId==nPhiRuns)
    {phiId=0; egrId++;}
  if(egrId==nEgrRuns)
    {egrId=0; threshId++;}
  if(threshId==nThreshRuns)
    {threshId=0; krylovId++;}
  if(krylovId==nKrylovRuns)
    {krylovId=0; residId++;}
  if(residId==nResidRuns)
    {residId=0;} // flip the odometer

  if(this->full_mole_fracs().size() == 0) {
    setInitialComp(phi[phiId],egr[egrId]); // update initial composition
  }
  runId++;
}

void idt_sweep_params::resetRunId()
{
  tempId=presId=phiId=egrId=threshId=krylovId=residId=runId=0;
  if(this->full_mole_fracs().size() == 0) {
    setInitialComp(phi[phiId],egr[egrId]); // update initial composition
  }
}

double idt_sweep_params::getDensity() const
{
  return gasMech->getDensityFromTPY(getInitTemp(),getInitPres(),&initMassFrac[0]);
}

void idt_sweep_params::getInitMoleFrac(double x[]) const
{
  int j;
  for(j=0; j<nSpc; j++)
    {x[j]=initMoleFrac[j];}
}

void idt_sweep_params::getInitMassFrac(double y[]) const
{
  int j;
  for(j=0; j<nSpc; j++)
    {y[j]=initMassFrac[j];}
}


void idt_sweep_params::getFracsFromCompMap(
    const std::map<std::string, double> comp,
    std::vector<double>& fracArray,
    int* nSpcComp,
    std::vector<int>& idxArray
)
{
  int nSpc = gasMech->getNumSpecies();
  for(int j = 0; j < nSpc; ++j)
  {
    fracArray[j] = UNMARKED;
  }
  std::map<std::string, double>::const_iterator mapit;
  for(mapit = comp.begin();
         mapit != comp.end(); ++mapit)
  {
    std::string spcName = mapit->first;
    int spcIdx = gasMech->getIdxFromName(spcName.c_str());
    if(spcIdx == -1)
    {
      std::cerr << "Species \"" << spcName << "\" in composition array \""
                << "\" not found in mechanism." << std::endl;
      exit(-1);
    }
    fracArray[spcIdx]=mapit->second;
    idxArray.push_back(spcIdx);
  }
  *nSpcComp = idxArray.size();
}



double idt_sweep_params::normalize(std::vector<double> &v)
{
  int j;
  double sum=0.0;
  double invSum=1.0;
  int N = v.size();

  for(j=0; j<N; )
    {sum+=v[j]; ++j;}

  if(sum != 0.0)
    {invSum=1.0/sum;}

  for(j=0; j<N; )
    {v[j]*=invSum; ++j;}
  return sum;
}
