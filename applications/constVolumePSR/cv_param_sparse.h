#ifndef CV_PARAM_SPARSE_H
#define CV_PARAM_SPARSE_H

#include "zerork/mechanism.h"
#include <slu_ddefs.h>
#ifdef CVIDT_USE_METIS
#include "metis.h"
#endif

#include "heat_transfer_model.h"

typedef struct
{
  int concIdx;   // or denominator or column index
  int rxnIdx;    // index in the cantera reaction list
  int sparseIdx; // index location in the sparse matrix list
} JsparseIdx;

typedef struct
{
  int nFwdDestroy;  // FD suffix
  int nFwdCreate;   // FC suffix

  JsparseIdx *fwdDestroy;
  JsparseIdx *fwdCreate;

} JsparseTermList;

typedef struct
{
  int nSize;
  int nNonZero;
  // compressed column storage
  double *mtxData;  // sparse jacobian data [length nNonZero]
  int *mtxRowIdx;   // row address [length nNonZero]
  int *mtxColSum;   // running tally of elements in each column [length nSize+1]

  int num_noninteger_jacobian_nonzeros;
  int *noninteger_sparse_id;
  double *noninteger_jacobian;

  // special data access
  int *diagIdx;     // diagonal elements [length nsize]
  int *lastRowIdx;  // last row [length nsize]

  JsparseTermList *termList;

  // sparse solver specific data
  int permutationType;  // TODO: Use enums instead of int's for different modes
  int isFirstFactor;
  char equed[1];
  SuperMatrix Mslu;
  SuperMatrix Lslu;
  SuperMatrix Uslu;
  SuperMatrix Bslu;
  SuperMatrix Xslu;
  int *rowPermutation;
  int *colPermutation;
  int *colElimTree; // etree

  int reduceNNZ;
  double *reduceData;
  int *reduceRowIdx;
  int *reduceColSum;
  int LUnnz;
  double fillFactor;

  double *vecRHS;
  double *vecSoln;
  double *Rvec;
  double *Cvec;
  superlu_options_t optionSLU;
  SuperLUStat_t statSLU;
  mem_usage_t  mem_usage;
#if SUPERLU_MAJOR_VERSION > 4
  GlobalLU_t Glu;
#endif

  double offDiagThreshold;

  bool ILU;
  bool fakeUpdate;
  int threshType;

#ifdef CVIDT_USE_METIS
  //Metis column permutation
  int metis_options[METIS_NOPTIONS];
#endif
} Jsparse;

JsparseTermList * alloc_JsparseTermList(const int inpFD, const int inpFC);
void free_JsparseTermList(JsparseTermList *w);

//Jsparse * alloc_Jsparse(const int inpSize, const int inpNNZ, const int inpFD,
//			const int inpFC, const int inpRD, const int inpRC);
Jsparse * alloc_Jsparse(zerork::mechanism &mechInp, double tol, bool doILU, bool fakeUpdate, int threshType,
                        double DiagPivotThresh, int permutationType);

void free_Jsparse(Jsparse *w);
void print_Jsparse(Jsparse *w);
double getElement_Jsparse(Jsparse *w,const int rowIdx, const int colIdx);
void calcReaction_Jsparse(Jsparse *w, const double invPosConc[],
			  const double fwdROP[]);
void change_JsparseThresh(Jsparse *w, double newThresh);



void countJacobianTerms(zerork::mechanism &mechInp, int *nFwdDestroy, int *nFwdCreate);

typedef struct
{
  int nFwdDestroy;  // FD suffix
  int nFwdCreate;   // FC suffix

  int *rxnFD, *rxnFC;
  int *specNumFD, *specNumFC;
  int *specDenFD, *specDenFC;

} Jterm_param;

Jterm_param * alloc_Jterm_param(zerork::mechanism &mechInp);
void free_Jterm_param(Jterm_param *w);
void print_Jterm_param(Jterm_param *w);

typedef double jacreal; // variable type for jacobian processing

typedef struct
{
  // df/dt constants
  //IdealGasMix *mix;  // pointer to the cantera mixture class
  zerork::mechanism *mech;
  int nSpc;
  double Tref;       // [K] reference temperature
  double Dens;       // [kg/m^3] constant volume reactor density
  double invDens;    // [m^3/kg] inverse of the const volume reactor density
  double *molWt;     // [kg/kmol] array of the molecular weights
  double residenceTime;
  double pressure;
  double volume;
  double mass;
  double enthalpyInlet;
  int energyEnabled;
  double Kpressure;

  // df/dt storage
  double meanCvMass; // [J/(kg-K)] mixture specific heat at constant volume
  double dTemp_dt;   // [K/s] temperature time derivative
  double *Energy;    // [J/kg] dimensional internal energy of each species
  double *CvMass;
  double *netProd;

  double *createRate;
  double *destroyRate;
  double *conc;

  double *yInlet;

  // Jacobian constants
  double minMassFrac;
  double sqrtUnitRnd;
  double *invMolWt;

  // Jacobian storage
  double *fwdROP;
  //jacreal *Mmtx,*Bvec,*Xvec; // ,*Jsave;
  //int *ipvt;

  // sparse Jacobian storage
  Jsparse *sparseMtx;

  // temeprature root
  double tempRoot;

  // use for recreating the divided difference scaling
  void *cvodeMemPtr;
  int nFunc,nJacSetup,nJacRescale,nJacFactor,nBackSolve,nColPerm;
  double colPermTime,jacSetupTime,jacFactorTime,backsolveTime,funcTime;

  long int prevNumErrTestFails;

  std::unique_ptr<zerork::HeatTransferModel> htxModel;
  std::vector<double> multipliers;
} cv_param;


#endif
