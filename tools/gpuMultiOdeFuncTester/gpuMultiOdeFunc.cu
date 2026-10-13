#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <time.h>
#include <sys/time.h>

#include "CKconverter/CKReader.h"
#include "zerork/mechanism_cuda.h"
#include "zerork/zerork_cuda_defs.h"
#include <nvector/nvector_serial.h> // serial N_Vector types, fcts., and macros

#include "cuda_profiler_api.h"

typedef struct
{
  // df/dt constants
  //IdealGasMix *mix;  // pointer to the cantera mixture class
  zerork::mechanism_cuda *mech;
  int nSpc;
  double Tref;       // [K] reference temperature
  double *Dens;       // [kg/m^3] constant volume reactor density
  double *invDens;    // [m^3/kg] inverse of the const volume reactor density
  double *molWt;     // [kg/kmol] array of the molecular weights

  // df/dt storage
  double *meanCvMass; // [J/(kg-K)] mixture specific heat at constant volume
  double *dTemp_dt;   // [K/s] temperature time derivative
  double *Energy;    // [J/kg] dimensional internal energy of each species
  double *CvMass;
  double *netProd;

  double *createRate;
  double *destroyRate;
  double *conc;

  // Jacobian constants
  double minMassFrac;
  double sqrtUnitRnd;
  double *invMolWt;

  // Jacobian storage
  bool doingJacSetup;
  double *fwdROP;


  //CUDA device memory
  double *y_dev;
  double *ydot_dev;
  double *Dens_dev;       // [kg/m^3] constant volume reactor density
  double *invDens_dev;    // [m^3/kg] inverse of the const volume reactor density
  double *molWt_dev;     // [kg/kmol] array of the molecular weights
  double *invMolWt_dev;
  double *reactorTemps_dev;

  // df/dt storage
  double *meanCvMass_dev; // [J/(kg-K)] mixture specific heat at constant volume
  double *Energy_dev;    // [J/kg] dimensional internal energy of each species
  double *CvMass_dev;
  double *netProd_dev;

  double *conc_dev;
  double *createRate_dev;
  double *destroyRate_dev;
  double *fwdROP_dev;


  // use for recreating the divided difference scaling
  void *cvodeMemPtr;
  int nFunc,nJacSetup,nJacRescale,nJacFactor,nBackSolve,nColPerm;
  double colPermTime,jacSetupTime,precSetupTime,jacFactorTime,backsolveTime,funcTime;

  long int prevNumErrTestFails;

  int currReactor; //used for single reactor code

  cudaStream_t concDerivStream,TDerivStream;
} ode_cv_param;


//void checkCudaError(cudaError_t err, const char * msg);




double getHighResolutionTime(void)
{
    struct timeval tod;

    gettimeofday(&tod, NULL);
    double time_seconds = (double) tod.tv_sec + ((double) tod.tv_usec / 1000000.0);
    return time_seconds;
}

using namespace std;
using namespace ckr;

#define MAX_LINE_LEN 1024
int GetLine(FILE *InFile, char *ReadLine, char UntilChar, int MaxChar);

int const_vol_wsr(realtype t, N_Vector y, N_Vector ydot,
		  void *user_data);

int const_vol_wsr_mr_dev(realtype t, N_Vector y, N_Vector ydot,
		  void *user_data);

int main(int argc, char *argv[])
{
  FILE *stateFptr,*outputFptr;
  zerork::mechanism_cuda *mech;

  int j,k;
  int nEval,nReactors;
  int nSpc, nmechSpc,nStep;
  char readLine[MAX_LINE_LEN];
  double *moleFrac;
  double *massFrac;

  double pres,Temp,rvol,dens,molwtMix,presConvert;
  double gasConstant;
  double startTime,stopTime;

  ode_cv_param systemParam;

  if(argc != 8)
    {
      printf("ERROR: incorrect command line usage.\n");
      printf("       use instead %s <ck2 mech file>  <ck2 thermo file> <ck2 converter output file> <state file> <output file> <# reactors> <# func evals>\n",argv[0]);
      exit(-1);
    }


  stateFptr=fopen(argv[4],"r");
  if(stateFptr==NULL)
    {
      printf("ERROR: could not open state vector file %s for read\n",
	     argv[4]);
      exit(-1);
    }
  outputFptr=fopen(argv[5],"w");
  if(outputFptr==NULL)
    {
      printf("ERROR: could not open output file %s for write\n",
	     argv[5]);
      exit(-1);
    }
  nReactors=atoi(argv[6]);
  nEval=atoi(argv[7]);

  mech=new zerork::mechanism_cuda(argv[1],argv[2],argv[3],1,nReactors); //4th param is verbosity

  nmechSpc=mech->getNumSpecies();
  nStep=mech->getNumSteps();
  // parse the input state vector
  GetLine(stateFptr,readLine,'\n',MAX_LINE_LEN);
  sscanf(readLine,"%d",&nSpc);
  if(nSpc != nmechSpc)
    {
      printf("WARNING: number of species in mechanism file %d\n",nmechSpc);
      printf("       differs from state file %d\n",nSpc);
    }
  if(nSpc < nmechSpc)
    {
      printf("ERROR: number of species in mechanism file %d\n",nmechSpc);
      printf("       is more than from state file %d\n",nSpc);
      exit(-1);
    }
  nSpc = nmechSpc;
  
  GetLine(stateFptr,readLine,'\n',MAX_LINE_LEN);
  sscanf(readLine,"%lf",&pres);
  GetLine(stateFptr,readLine,'\n',MAX_LINE_LEN);
  sscanf(readLine,"%lf",&Temp);

  moleFrac   = new double[nSpc*nReactors];
  massFrac   = new double[nSpc*nReactors];

  // set up the system parameters
  systemParam.doingJacSetup = false;
  systemParam.nSpc=nSpc;
  systemParam.minMassFrac=1.0e-30;
  systemParam.sqrtUnitRnd=sqrt(UNIT_ROUNDOFF);
  systemParam.Tref=1000.0;
  systemParam.mech=mech;
  // don't forget to reset densities for each calculations
  //systemParam.Dens=gasMech->getDensityFromTPY(tempSweep[0],pres0,massFracPtr);
  //systemParam.invDens=1.0/systemParam.Dens;
  systemParam.invDens = (double *)malloc(sizeof(double)*nReactors);
  systemParam.Dens    = (double *)malloc(sizeof(double)*nReactors);
  systemParam.meanCvMass = (double *)malloc(sizeof(double)*nReactors);
  systemParam.dTemp_dt = (double *)malloc(sizeof(double)*nReactors);

  systemParam.netProd    =(double *)malloc(sizeof(double)*nSpc*nReactors);
  systemParam.Energy     =(double *)malloc(sizeof(double)*nSpc*nReactors);
  systemParam.CvMass     =(double *)malloc(sizeof(double)*nSpc*nReactors);
  systemParam.molWt      =(double *)malloc(sizeof(double)*nSpc*nReactors);
  systemParam.invMolWt   =(double *)malloc(sizeof(double)*nSpc*nReactors);
  systemParam.fwdROP     =(double *)malloc(sizeof(double)*nStep*nReactors);
  systemParam.createRate =(double *)malloc(sizeof(double)*nSpc*nReactors);
  systemParam.destroyRate=(double *)malloc(sizeof(double)*nSpc*nReactors);
  systemParam.conc       =(double *)malloc(sizeof(double)*nSpc*nReactors);

  int nState = nSpc + 1;
  zerork::checkCudaError
  (
      cudaMalloc((void**)&systemParam.y_dev,sizeof(double)*nState*nReactors),
      "cudaMalloc(... y_dev ...)"
  );
  zerork::checkCudaError
  (
      cudaMalloc((void**)&systemParam.ydot_dev,sizeof(double)*nState*nReactors),
      "cudaMalloc(... ydot_dev ...)"
  );
  zerork::checkCudaError
  (
      cudaMalloc((void**)&systemParam.Dens_dev,sizeof(double)*nSpc*nReactors),
      "cudaMalloc(... Dens_dev ...)"
  );
  zerork::checkCudaError
  (
      cudaMalloc((void**)&systemParam.invDens_dev,sizeof(double)*nSpc*nReactors),
      "cudaMalloc(... invDens_dev ...)"
  );
  zerork::checkCudaError
  (
      cudaMalloc((void**)&systemParam.molWt_dev,sizeof(double)*nSpc),
      "cudaMalloc(... molWt_dev ...)"
  );
  zerork::checkCudaError
  (
      cudaMalloc((void**)&systemParam.invMolWt_dev,sizeof(double)*nSpc),
      "cudaMalloc(... invMolWt_dev ...)"
  );
  zerork::checkCudaError
  (
      cudaMalloc((void**)&systemParam.meanCvMass_dev,sizeof(double)*nReactors),
      "cudaMalloc(... meanCvMass_dev ...)"
  );
  zerork::checkCudaError
  (
      cudaMalloc((void**)&systemParam.Energy_dev,sizeof(double)*nSpc*nReactors),
      "cudaMalloc(... Energy_dev ...)"
  );
  zerork::checkCudaError
  (
      cudaMalloc((void**)&systemParam.CvMass_dev,sizeof(double)*nSpc*nReactors),
      "cudaMalloc(... CvMass_dev ...)"
  );
  zerork::checkCudaError
  (
      cudaMalloc((void**)&systemParam.netProd_dev,sizeof(double)*nSpc*nReactors),
      "cudaMalloc(... netProd_dev ...)"
  );
  zerork::checkCudaError
  (
      cudaMalloc((void**)&systemParam.conc_dev,sizeof(double)*nSpc*nReactors),
      "cudaMalloc(... conc_dev ...)"
  );
  zerork::checkCudaError
  (
      cudaMalloc((void**)&systemParam.createRate_dev,sizeof(double)*nSpc*nReactors),
      "cudaMalloc(... createRate_dev ...)"
  );
  zerork::checkCudaError
  (
      cudaMalloc((void**)&systemParam.destroyRate_dev,sizeof(double)*nSpc*nReactors),
      "cudaMalloc(... destroyRate_dev ...)"
  );
  zerork::checkCudaError
  (
      cudaMalloc((void**)&systemParam.fwdROP_dev,sizeof(double)*nStep*nReactors),
      "cudaMalloc(... fwdROP_dev ...)"
  );
  zerork::checkCudaError
  (
      cudaMalloc((void**)&systemParam.reactorTemps_dev,sizeof(double)*nReactors),
      "cudaMalloc(... reactorTemps_dev ...)"
  );

  // set constant parameters //TODO: Possibly calculate for each reactor
  systemParam.mech->getMolWtSpc(systemParam.molWt);
  for(j=0; j<nSpc; j++)
    {systemParam.invMolWt[j]=1.0/systemParam.molWt[j];}

  cudaMemcpy(systemParam.molWt_dev,systemParam.molWt,sizeof(double)*nSpc,cudaMemcpyHostToDevice);
  cudaMemcpy(systemParam.invMolWt_dev,systemParam.invMolWt,sizeof(double)*nSpc,cudaMemcpyHostToDevice);

  cudaStreamCreate(&(systemParam.concDerivStream));
  cudaStreamCreate(&(systemParam.TDerivStream));

  for(j=0; j<nSpc-1; j++)
    {
      GetLine(stateFptr,readLine,'\n',MAX_LINE_LEN);
      sscanf(readLine,"%lf",&moleFrac[j]);
    }
  // readin the last mole frac ignoring the carriage return
  fscanf(stateFptr,"%lf",&moleFrac[nSpc-1]);
 
  // first echo the input
  fprintf(outputFptr,"%24d ! [#]             number of species\n",nSpc);
  fprintf(outputFptr,"%24.16e ! [Pa]            input pressure\n",pres);
  fprintf(outputFptr,"%24.16e ! [K]             input temperature\n",Temp);
  for(j=0; j<nSpc; j++)
    {
      fprintf(outputFptr,"%24.16e ! [-]             input mole frac - %s\n",
	      moleFrac[j],mech->getSpeciesName(j));
    }
  // compute the mass fractions
  mech->getYfromX(moleFrac,massFrac);


  // compute the mixture properties
  dens=mech->getDensityFromTPY(Temp,pres,massFrac);
  rvol=1.0/dens;
  molwtMix=mech->getMolWtMixFromY(massFrac);
  presConvert=mech->getPressureFromTVY(Temp,rvol,massFrac);
  gasConstant=mech->getGasConstant();

  fprintf(outputFptr,"%24.16e ! [kg/m^3]        density\n",dens);
  fprintf(outputFptr,"%24.16e ! [m^3/kg]        relative volume\n",rvol);
  fprintf(outputFptr,"%24.16e ! [kg/kmol]       molecular weight\n",molwtMix);
  fprintf(outputFptr,"%24.16e ! [Pa]            pressure (from rel volume)\n",
	  presConvert);
  fprintf(outputFptr,"%24.16e ! [J/kmol/K]      univerisal gas const\n",
	  gasConstant);

 double *Temp_array = new double[nReactors];

 int prevConcIdx = nSpc-1; //set to 0 for identical concentrations across reactors
 double Temp_mod = 0.1; //set to 0.0 for identical temps across reactors
 Temp_array[0] = Temp;
 for(j=1; j<nReactors; j+=1)
    {
      Temp_array[j] = Temp+j*Temp_mod;
      for( k = 0; k < nSpc; ++k )
        {
          massFrac[j*nSpc+k] = massFrac[(j-1)*nSpc + prevConcIdx];
          ++prevConcIdx;
          if(prevConcIdx >= nSpc) prevConcIdx=0;
        }
    }
  cudaMemcpy(systemParam.reactorTemps_dev,Temp_array,sizeof(double)*nReactors,cudaMemcpyHostToDevice);

  for(k=0;k<nReactors;++k)
  {
      systemParam.Dens[k]=mech->getDensityFromTPY(Temp_array[k],pres,massFrac);
      systemParam.invDens[k]=1.0/systemParam.Dens[k];
  }

  cudaMemcpy(systemParam.Dens_dev,systemParam.Dens,sizeof(double)*nReactors,cudaMemcpyHostToDevice);
  cudaMemcpy(systemParam.invDens_dev,systemParam.invDens,sizeof(double)*nReactors,cudaMemcpyHostToDevice);

  N_Vector systemState;
  N_Vector systemDeriv;

  N_Vector systemStateIns[nReactors];
  N_Vector systemDerivIns[nReactors];
  
  systemState=N_VNew_Serial((nSpc+1)*nReactors);
  systemDeriv=N_VNew_Serial((nSpc+1)*nReactors);

  for(k=0; k<nReactors; k++)
  {
      systemStateIns[k]=N_VNew_Serial((nSpc+1));
      systemDerivIns[k]=N_VNew_Serial((nSpc+1));
  }

  //systemState in data order
  for(j=0; j<nSpc; j++)
  {
      for(k=0; k<nReactors; k++)
      {
          NV_Ith_S(systemState,j*nReactors+k)=massFrac[k*nSpc+j];
          NV_Ith_S((systemStateIns[k]),j)=massFrac[k*nSpc+j];
      }
  }
  for(k=0; k<nReactors; k++)
  {
      NV_Ith_S(systemState,nSpc*nReactors+k)=Temp_array[k]/systemParam.Tref;
      NV_Ith_S((systemStateIns[k]),nSpc)=Temp_array[k]/systemParam.Tref;
  }


  startTime=getHighResolutionTime();
  for( k = 0; k < nEval; ++k)
  {
     for(j=0; j<nReactors; ++j)
     {
          // calculate the ODE RHS
          systemParam.currReactor = j;
          const_vol_wsr(0.0, systemStateIns[j], systemDerivIns[j], (void *)&systemParam);
     }
  }
  stopTime=getHighResolutionTime();
  double cpuTime = stopTime-startTime;
  printf("# Elapsed time for %d*%d dy/dt calls on CPU [s]: %16.8e\n",nReactors,nEval, cpuTime);


  startTime=getHighResolutionTime();
  for(j=0; j<nEval; ++j)
     {
           // calculate the ODE RHS
           const_vol_wsr_mr_dev(0.0, systemState, systemDeriv, (void *)&systemParam);
     }

  stopTime=getHighResolutionTime();
  double gpuTime = stopTime-startTime;
  printf("# Elapsed time for %d*%d dy/dt calls on GPU [s]: %16.8e\n",nReactors,nEval, gpuTime);


  //Test GPU_dev results
  const int max_diff_print = 50;
  const double thresh = 1.0e-10;
  const double cutoff = 1.0e-30;
  int diffcount = 0;
  for( k=0; k<nReactors; ++k)
  {  
      for( j=0; j<nState; ++j)
      {  
          double cpu_val,gpu_val,diff;
          cpu_val = NV_Ith_S(systemDerivIns[k],j);
          gpu_val = NV_Ith_S(systemDeriv,j*nReactors+k);
          
          diff = fabs((cpu_val-gpu_val)/(cpu_val+gpu_val+cutoff));
          if( diff > thresh )
          {
               printf("Diff in system derivative [%d,%d]: %g, %g, %g\n",k,j,cpu_val,gpu_val,diff);
               ++diffcount;
          }
          if(diffcount > max_diff_print) break;      
      }  
      if(diffcount > max_diff_print) break;      
  }  

  printf(" +++ %d  %d  %g \n",nSpc,nReactors,cpuTime/gpuTime);

  //Clean-up
  delete mech;
  delete [] moleFrac;
  delete [] massFrac;
  delete [] Temp_array;

  free(systemParam.invDens);
  free(systemParam.Dens);
  free(systemParam.meanCvMass);
  free(systemParam.dTemp_dt);

  free(systemParam.netProd);
  free(systemParam.Energy);
  free(systemParam.CvMass);
  free(systemParam.molWt);
  free(systemParam.invMolWt);
  free(systemParam.fwdROP);
  free(systemParam.createRate);
  free(systemParam.destroyRate);
  free(systemParam.conc);

  cudaFree(systemParam.y_dev);
  cudaFree(systemParam.ydot_dev);
  cudaFree(systemParam.Dens_dev);
  cudaFree(systemParam.invDens_dev);
  cudaFree(systemParam.molWt_dev);
  cudaFree(systemParam.invMolWt_dev);
  cudaFree(systemParam.meanCvMass_dev);
  cudaFree(systemParam.Energy_dev);
  cudaFree(systemParam.CvMass_dev);
  cudaFree(systemParam.netProd_dev);
  cudaFree(systemParam.conc_dev);
  cudaFree(systemParam.createRate_dev);
  cudaFree(systemParam.destroyRate_dev);
  cudaFree(systemParam.fwdROP_dev);
  cudaFree(systemParam.reactorTemps_dev);

  N_VDestroy_Serial(systemState);
  N_VDestroy_Serial(systemDeriv);

  for(k=0; k<nReactors; k++)
  {
      N_VDestroy_Serial(systemStateIns[k]);
      N_VDestroy_Serial(systemDerivIns[k]);
  }

  cudaStreamDestroy(systemParam.concDerivStream);
  cudaStreamDestroy(systemParam.TDerivStream);

  fclose(outputFptr);
  fclose(stateFptr);

  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}
  

int GetLine(FILE *InFile,char *ReadLine, char UntilChar, int MaxChar)
{
  int CharCtr=0;
  char LocalReadChar='\0';


  while((LocalReadChar != UntilChar) && CharCtr < (MaxChar-1))
    {
      fscanf(InFile,"%c",&LocalReadChar);
      //printf("LocalReadChar[%d]: %c    UntilChar: %c\n",CharCtr,
      //             LocalReadChar,UntilChar);
      ReadLine[CharCtr]=LocalReadChar;
      CharCtr++;
    }
  if(CharCtr == (MaxChar-1) && LocalReadChar != UntilChar) // ran out of space
    {ReadLine[0]='\0'; return -1;}

  ReadLine[CharCtr]='\0';
  return CharCtr; // exit normally
}


int const_vol_wsr(realtype t, N_Vector y, N_Vector ydot,
                         void *user_data)
{
  ode_cv_param *cvp=(ode_cv_param *)user_data;
  double *mfp=NV_DATA_S(y); // caution: assumes realtype == double
  double Esum,Temp;
  int j;
  double startTime=getHighResolutionTime();

  // set concentration via density and mass fraction
  cvp->mech->getCfromVY(cvp->invDens[cvp->currReactor],mfp,cvp->conc);
  // set temperature
  Temp = NV_Ith_S(y,cvp->nSpc)*cvp->Tref;

  // compute the molar production rates at the current state (aka wdot)
  cvp->mech->getReactionRates(Temp,cvp->conc,cvp->netProd,cvp->createRate,
                              cvp->destroyRate,cvp->fwdROP);

  cvp->mech->getIntEnergy_RT(Temp,cvp->Energy);
  cvp->meanCvMass[cvp->currReactor]=cvp->mech->getMassCvFromTY(Temp,mfp,cvp->CvMass);

  // ydot = [kmol/m^3/s] * [kg/kmol] * [m^3/kg] = [(kg spec j)/(kg mix)/s]
  for(j=0; j<cvp->nSpc; j++)
    {NV_Ith_S(ydot,j)=(cvp->netProd[j])*(cvp->molWt[j])*(cvp->invDens[cvp->currReactor]);}

  Esum=0.0;
  for(j=0; j<cvp->nSpc; j++)
    {Esum+=(cvp->Energy[j])*(cvp->netProd[j]);}

  Esum*=cvp->mech->getGasConstant()*NV_Ith_S(y,cvp->nSpc)*(cvp->invDens[cvp->currReactor])
    /(cvp->meanCvMass[cvp->currReactor]);
  cvp->dTemp_dt[cvp->currReactor]=-Esum;

  NV_Ith_S(ydot,cvp->nSpc)=cvp->dTemp_dt[cvp->currReactor];

  (cvp->nFunc)++;
  cvp->funcTime += getHighResolutionTime() - startTime;
  return 0;
}


void __global__ multiplyElements
(
    const int nElems,
    const double multFact,
    double *A_dev
)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < nElems)
    {
        A_dev[tid] *= multFact;
    }
}


void __global__ cv_wsr_conc_deriv
(
    const int nReactors,
    const int nSpc,
    const double *netProd_dev,
    const double *molWt_dev,
    const double *invDens_dev,
    double *ydot_dev
)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int reactorid = blockIdx.y*blockDim.y + threadIdx.y;
    if(reactorid < nReactors)
    {
        if(tid < nSpc)
        {
            ydot_dev[nReactors*tid+reactorid] = netProd_dev[nReactors*tid+reactorid]*molWt_dev[tid]*invDens_dev[reactorid];
        }
    }
}


void __global__ cv_wsr_T_deriv
(
    const int nReactors,
    const int nSpc,
    const double gasConstant,
    const double *intEnergy_dev,
    const double *netProd_dev,
    const double *invDens_dev,
    const double *meanCvMass_dev,
    const double *y_dev,
    double *ydot_dev
)   
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < nReactors)
    {
        extern __shared__ double ydot_shr[];
        ydot_shr[threadIdx.x] = 0;
        for(int k=0; k<nSpc; ++k)
        {
            ydot_shr[threadIdx.x] += intEnergy_dev[nReactors*k+tid]*netProd_dev[nReactors*k+tid];
        }
        ydot_shr[threadIdx.x] *= -1*gasConstant*y_dev[nSpc*nReactors+tid]*invDens_dev[tid]/meanCvMass_dev[tid];
        ydot_dev[nSpc*nReactors+tid] = ydot_shr[threadIdx.x];
    }
}

int const_vol_wsr_mr_dev(realtype t, N_Vector y, N_Vector ydot,
			 void *user_data)
{
//  printf("Zero-RK const_vol_wsr_mr_dev\n");
  ode_cv_param *cvp=(ode_cv_param *)user_data;
  int nSize = cvp->nSpc+1; //single reactor system size (nSpc + 1)
  int nSpc = cvp->nSpc;
  int nReactors = NV_LENGTH_S(y)/nSize;

  int nThreads,nBlocks;
  dim3 nThreads2D,nBlocks2D;
  double startTime=getHighResolutionTime();

  cudaMemcpy(cvp->y_dev,NV_DATA_S(y),sizeof(double)*nSize*nReactors,cudaMemcpyHostToDevice);
  cudaMemcpy(cvp->reactorTemps_dev,&(cvp->y_dev[nSpc*nReactors]),sizeof(double)*nReactors,cudaMemcpyDeviceToDevice);

  // set concentration via density and mass fraction
  cvp->mech->getCfromVY_mr_dev(nReactors,cvp->invDens_dev,cvp->y_dev,cvp->conc_dev);

  // set temperature
  nThreads = 1024;
  nBlocks = (nReactors + nThreads -1)/nThreads;
  multiplyElements<<<nBlocks,nThreads>>>(nReactors,cvp->Tref,cvp->reactorTemps_dev);

  // compute the molar production rates at the current state (aka wdot)
  cvp->mech->getReactionRates_CUDA_mr_dev(nReactors,cvp->reactorTemps_dev,
                                                    cvp->conc_dev,cvp->netProd_dev,
                                                    cvp->createRate_dev,cvp->destroyRate_dev,
                                                    cvp->fwdROP_dev);

  cvp->mech->getIntEnergy_RT_mr_dev(nReactors,cvp->reactorTemps_dev,cvp->Energy_dev);
  cvp->mech->getMassCvFromTY_mr_dev(nReactors,cvp->reactorTemps_dev,cvp->y_dev,cvp->CvMass_dev,cvp->meanCvMass_dev);

  // ydot = [kmol/m^3/s] * [kg/kmol] * [m^3/kg] = [(kg spec j)/(kg mix)/s]
  nThreads2D.x = 1;
  nThreads2D.y = min(nReactors,1024/nThreads2D.x);
  nBlocks2D.x = (nSpc+nThreads2D.x-1)/nThreads2D.x;
  nBlocks2D.y = (nReactors+nThreads2D.y-1)/nThreads2D.y;

  cv_wsr_conc_deriv<<<nBlocks2D,nThreads2D,0,cvp->concDerivStream>>>(nReactors,nSpc,cvp->netProd_dev,cvp->molWt_dev,cvp->invDens_dev,cvp->ydot_dev);   

  cudaMemset(&(cvp->ydot_dev[nReactors*nSpc]),0,sizeof(double)*nReactors);
  nThreads = 1024;
  nBlocks = (nReactors+nThreads-1)/nThreads;
  cv_wsr_T_deriv<<<nBlocks,nThreads,sizeof(double)*nThreads,cvp->TDerivStream>>>(nReactors,nSpc,cvp->mech->getGasConstant(),
                                       cvp->Energy_dev,cvp->netProd_dev,cvp->invDens_dev,
                                       cvp->meanCvMass_dev,cvp->y_dev,cvp->ydot_dev);

  cudaMemcpy(NV_DATA_S(ydot),cvp->ydot_dev,sizeof(double)*nSize*nReactors,cudaMemcpyDeviceToHost);

  (cvp->nFunc)++;
  cvp->funcTime += getHighResolutionTime() - startTime;
  return 0;
}



