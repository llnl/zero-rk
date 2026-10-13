#ifndef SWEEP_UTIL_SPIFY_H
#define SWEEP_UTIL_SPIFY_H

#include <vector>
#include "idt_sweep_IFP.h"
#include "zerork/mechanism.h"

const int MAX_SPEC_NAME_LEN = 32;
const int MAX_SPEC_MOLE_LEN = 32;
const int MAX_FILE_NAME_LEN = 1024;

const char permutationTypeName[3][27]=
  {"MMD_AT_PLUS_A",
   "Metis_NodeND",
   "External (from perm_c.txt)"};
const char precondThresholdName[5][24]=
  {"absolute",
   "column-norm",
   "row-norm",
   "min(row,col) norm",
   "fixed fraction of terms"};


class idt_sweep_params : public idt_sweep_IFP
{
 public:
  idt_sweep_params(char *inputFileName);
  void setInitialComp(const double phi, const double egr);
  void printInitialMoleComp() const;
  void incrementRunId();

  double getInitTemp()  const {return initTemp[tempId];}
  double getInitPres()  const {return initPres[presId];}
  double getPhi()       const {return phi[phiId];}
  double getEgr()       const {return egr[egrId];}
  double getThresh()    const {return thresh[threshId];}
  int    getKrylovDim() const {return krylovDim[krylovId];}
  int    getThreshId()  const {return threshId;}
  double getThreshFromId(const int id)  const {return thresh[id];}
  int    getNumThreshRuns() const {return nThreshRuns;}
  int    getRunId()     const {return runId;}
  int    getRunTotal()  const {return runTotal;}
  double getInitDensity()   const;
  double getDensity(const double T, const double p, const double y[])   const;
  void   getInitMoleFrac(double x[]) const;
  void   getInitMassFrac(double y[]) const;
  void   getExhaustMassFrac(double y[]) const;
  void   getFuelMassFrac(double y[]) const;
  void   getOxidMassFrac(double y[]) const;
  double getExhaustTemp()  const {return exhaustTemp;}
  int    getNumTrackSpecies() const {return nTrackSpc;}
  int    getTrackSpeciesId(const int id) const {return trackSpcIdx[id];}

  int    getNumFuelSpecies() const {return nFuelSpc;}
  int    getNumOxidSpecies() const {return nOxidSpc;}
  int    getNumTraceSpecies() const {return nTraceSpc;}
  int    getFuelSpeciesId(const int id) const {return fuelSpcIdx[id];}
  int    getOxidSpeciesId(const int id) const {return oxidSpcIdx[id];}
  int    getTraceSpeciesId(const int id) const {return traceSpcIdx[id];}
  double getFuelMoleFrac(const int id) const {return fuelMoleFrac[id];}
  double getOxidMoleFrac(const int id) const {return oxidMoleFrac[id];}
  double getTraceMoleFrac(const int id) const {return traceMoleFrac[id];}

  int    getNumSpecies() const {return nSpc;}
  int    getNumSteps()   const {return gasMech->getNumSteps();}
  double getRefTemp()    const {return Tref;}
  double getStepLimiter()    const {return steplim;}
  double getDeltaTign()  const {return deltaTign;}
  double getRTol()       const {return rtol;}
  double getATol()       const {return atol;}
  int    getMaxInternalSteps() const {return maxInternalSteps;}
  double getMaxInternalDt()    const {return maxInternalDt;}
  const char * getIdtFileName()      const {return this->idtFile().c_str();}
  const char * getTHistFileName()    const {return this->thistFile().c_str();}
  const char * getMechFileName()     const {return this->mechFile().c_str();}
  const char * getThermFileName()    const {return this->thermFile().c_str();}
  const char * getLogFileName()      const {return this->logFile().c_str();}

  // PaSR
  int getNumParticles() const {return numParticles;}
  int getNumResidenceTime() const {return numResidenceTime;}
  double getResidenceTime() const {return residenceTime;}
  double getMixingTime() const {return mixingTime;}
  double getPairingTime() const {return pairingTime;}
  int getCombustionRegime() const {return this->combustionRegime();}
  //

  bool getILU()         const {return doILU;}
  int getPermutationType()         const {return permutationType;}
  bool getUpdate()        const {return fakeUpdate;}
  int getThreshType()        const {return threshType;}
  double getPartialPivotThresh()        const {return partialPivotThresh;}
  double getEpsLin()        const {return epsLin;}
  double getNlConvCoeff()        const {return nlConvCoeff;}
  int oneStep()         const {return oneStepMode;}
  int longOutput()         const {return this->long_output();}
  int continueAfterIDT()  const {return isContinueAfterIDT;}

  int writeState()         const {return this->writeState();}

  zerork::mechanism * getMechPtr() {return gasMech;}

  int getMaxPrimaryCvodeFails() const {return maxCvodeFails1;}
  int getMaxSecondaryCvodeFails() const {return maxCvodeFails2;}
  double getSafetyThreshold() const {return safetyThresh;}

  double getFuelOxyBalance() const {return fuelOxygenBal;}
  double getOxidOxyBalance() const {return oxidOxygenBal;}
  double getMoleOxidStoicPerFuel() const {return moleOxidStoicPerFuel;}

  std::vector<double> getInletRatios() const {return this->inlet_ratios();}
  double getPilotDuration() const {return this->pilot_duration();}

  ~idt_sweep_params();

 private:
  // data directly read from input file
  int nFuelSpc,nOxidSpc,nTraceSpc;
  int isConstPres;
  int isContinueAfterIDT;
  double stopTime,printTime,maxInternalDt;
  int maxInternalSteps;
  double rtol,atol,deltaTign,Tref,steplim;
  bool fakeUpdate,doILU,oneStepMode;
  //bool printAllSteps;
  int permutationType;
  int threshType;
  double partialPivotThresh,epsLin,nlConvCoeff;

  // PaSR
  int numParticles, numResidenceTime;
  double residenceTime, mixingTime, pairingTime;
  double exhaustTemp;

  int nTempRuns,nPresRuns,nPhiRuns,nEgrRuns,nThreshRuns,nKrylovRuns;
  // ----------------------------------
  int nSpc, nTrackSpc;
  double fuelOxygenBal,oxidOxygenBal,moleOxidStoicPerFuel, totalTraceMoleFraction;

  zerork::mechanism *gasMech;

  std::vector<double> fuelMoleFrac;
  std::vector<double> fuelMassFrac;
  std::vector<double> oxidMoleFrac;
  std::vector<double> oxidMassFrac;
  std::vector<double> traceMoleFrac;
  std::vector<double> freshMoleFrac;
  std::vector<double> freshMassFrac;
  std::vector<double> exhaustMoleFrac;
  std::vector<double> exhaustMassFrac;
  std::vector<double> initMoleFrac;
  std::vector<double> initMassFrac;

  std::vector<int> trackSpcIdx;
  std::vector<int> fuelSpcIdx;
  std::vector<int> oxidSpcIdx;
  std::vector<int> traceSpcIdx;

  int tempId,presId,phiId,egrId,threshId,krylovId,runId;
  int runTotal;
  std::vector<double> initTemp;
  std::vector<double> initPres;
  std::vector<double> phi;
  std::vector<double> egr;
  std::vector<double> thresh;
  std::vector<int> krylovDim;

  int maxCvodeFails1,maxCvodeFails2;
  double safetyThresh;

  double normalize(std::vector<double> &v);

  void getFracsFromCompMap(const std::map<std::string, double> comp,
    std::vector<double>& fracArray,
    int* nSpcComp,
    std::vector<int>& idxArray);

};

#endif
