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
  void resetRunId();

  double getInitTemp()  const {return initTemp[tempId];}
  double getInitPres()  const {return initPres[presId];}
  double getPhi()       const {return phi[phiId];}
  double getEgr()       const {return egr[egrId];}
  double getThresh()    const {return thresh[threshId];}
  int    getKrylovDim() const {return krylovDim[krylovId];}
  int    getThreshId()  const {return threshId;}
  double getThreshFromId(const int id)  const {return thresh[id];}
  int    getNumThreshRuns() const {return nThreshRuns;}
  double getResidenceTime()  const {return residenceTimes[residId];}
  int    getRunId()     const {return runId;}
  int    getRunTotal()  const {return runTotal;}
  double getDensity()   const;
  void   getInitMoleFrac(double x[]) const;
  void   getInitMassFrac(double y[]) const;
  int    getNumTrackSpecies() const {return nTrackSpc;}
  int    getTrackSpeciesId(const int id) const {return trackSpcIdx[id];}

  int    getNumFuelSpecies() const {return nFuelSpc;}
  int    getNumOxidSpecies() const {return nOxidSpc;}
  int    getNumTraceSpecies() const {return nTraceSpc;}
  int    getNumFullSpecies() const {return nFullSpc;}
  int    getFuelSpeciesId(const int id) const {return fuelSpcIdx[id];}
  int    getOxidSpeciesId(const int id) const {return oxidSpcIdx[id];}
  int    getTraceSpeciesId(const int id) const {return traceSpcIdx[id];}
  int    getFullSpeciesId(const int id) const {return fullSpcIdx[id];}
  double getFuelMoleFrac(const int id) const {return fuelMoleFrac[id];}
  double getOxidMoleFrac(const int id) const {return oxidMoleFrac[id];}
  double getTraceMoleFrac(const int id) const {return traceMoleFrac[id];}
  double getFullMoleFrac(const int id) const {return fullMoleFrac[id];}

  int    getNumSpecies() const {return nSpc;}
  int    getNumSteps()   const {return gasMech->getNumSteps();}
  double getRefTemp()    const {return Tref;}
  double getRTol()       const {return rtol;}
  double getATol()       const {return atol;}
  int    getMaxInternalSteps() const {return maxInternalSteps;}
  double getMaxInternalDt()    const {return maxInternalDt;}
  double getStopTime()         const {return stopTime;}
  double getPrintTime()        const {return printTime;}
  const char * getIdtFileName()      const {return this->idtFile().c_str();}
  const char * getTHistFileName()    const {return this->thistFile().c_str();}
  const char * getMechFileName()     const {return this->mechFile().c_str();}
  const char * getThermFileName()    const {return this->thermFile().c_str();}
  const char * getLogFileName()      const {return this->logFile().c_str();}
  double getPressureCoefficient() const {return Kpressure;}

  bool getILU()         const {return doILU;}
  int getPermutationType()         const {return permutationType;}
  bool getUpdate()        const {return fakeUpdate;}
  int getThreshType()        const {return threshType;}
  double getPartialPivotThresh()        const {return partialPivotThresh;}
  double getEpsLin()        const {return epsLin;}
  double getNlConvCoeff()        const {return nlConvCoeff;}
  int oneStep()         const {return oneStepMode;}
  int longOutput()         const {return this->long_output();}
  int energyEnabled()  const {return isEnergyEnabled;}
  int dumpJacobian() const {return doDumpJacobian;}
  bool extinctionCurveSweep() const { return runTotal == residenceTimes.size(); }

  int printNetProductionRates() const {return printNetProdRates;}
  int printNetRatesOfProgress() const {return printNetROP;}

  zerork::mechanism * getMechPtr() {return gasMech;}

  int getMaxPrimaryCvodeFails() const {return maxCvodeFails1;}
  int getMaxSecondaryCvodeFails() const {return maxCvodeFails2;}
  double getSafetyThreshold() const {return safetyThresh;}

  double getFuelOxyBalance() const {return fuelOxygenBal;}
  double getOxidOxyBalance() const {return oxidOxygenBal;}
  double getMoleOxidStoicPerFuel() const {return moleOxidStoicPerFuel;}

  ~idt_sweep_params();

 private:
  // data directly read from input file
  int nFuelSpc,nOxidSpc,nTraceSpc,nFullSpc;
  int isConstPres;
  int isEnergyEnabled;
  double stopTime,printTime,maxInternalDt;
  int maxInternalSteps;
  double rtol,atol,Tref,volume, Kpressure;
  std::vector<double> residenceTimes;
  bool fakeUpdate,doILU,oneStepMode,printNetProdRates,printNetROP;
  //bool printAllSteps;
  int permutationType;
  int threshType;
  int doDumpJacobian;
  double partialPivotThresh,epsLin,nlConvCoeff;

  int nTempRuns,nPresRuns,nPhiRuns,nEgrRuns,nThreshRuns,nKrylovRuns,nResidRuns;
  // ----------------------------------
  int nSpc, nTrackSpc;
  double fuelOxygenBal,oxidOxygenBal,moleOxidStoicPerFuel, totalTraceMoleFraction;

  //char *mechFileName,*thermFileName,*logFileName,*idtFileName,*thistFileName;

  zerork::mechanism *gasMech;

  std::vector<double> fuelMoleFrac;
  std::vector<double> oxidMoleFrac;
  std::vector<double> traceMoleFrac;
  std::vector<double> freshMoleFrac;
  std::vector<double> freshMassFrac;
  std::vector<double> exhaustMoleFrac;
  std::vector<double> exhaustMassFrac;
  std::vector<double> initMoleFrac;
  std::vector<double> initMassFrac;
  std::vector<double> fullMoleFrac;

  std::vector<int> trackSpcIdx;
  std::vector<int> fuelSpcIdx;
  std::vector<int> oxidSpcIdx;
  std::vector<int> traceSpcIdx;
  std::vector<int> fullSpcIdx;

  int tempId,presId,phiId,egrId,threshId,krylovId,residId,runId;
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
