#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <time.h>
#include <sys/time.h>

#include "CKconverter/CKReader.h"
#include "zerork/mechanism_cuda.h"
#include <nvector/nvector_serial.h> // serial N_Vector types, fcts., and macros

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

typedef struct
{
  int nSpc;
  double dens;
  double invDens;
  double dTemp_dt;
  double meanCvMass;
  double *netProdRate;
  double *stepROP;
  double *createRate;
  double *destroyRate;
  
  double *molWt; 
  double *conc;
  double *cvSpc;
  double *internalEnergy;
  zerork::mechanism_cuda *mech;

} ode_cv_param;

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

int main(int argc, char *argv[])
{
  FILE *stateFptr,*outputFptr;
  zerork::mechanism_cuda *mech;

  int j,k;
  int nEval,nReactors;
  int nSpc, nRxn, nmechSpc,nStep;
  char readLine[MAX_LINE_LEN];
  double *moleFrac;
  double *massFrac,*cpSpc,*hSpc,*gSpc;
  double *Kfwd,*Krev;//,*stepROP;
//  double *netSpc,*createSpc,*destroySpc;
//  double dY;

  double pres,Temp,rvol,dens,molwtMix,presConvert,cpMix,cvMix,uMix,hMix;
  double gasConstant;
  double startTime,stopTime;

  ode_cv_param sysParam;

  if(argc != 8)
    {
      printf("ERROR: incorrect command line usage.\n");
      printf("       use instead %s <ck2 mech file>  <ck2 thermo file> "
             "<ck2 converter output file> <state file> <output file> "
             "<# reactors> <# func evals>\n",argv[0]);
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
  nRxn=mech->getNumReactions();
  nStep=mech->getNumSteps();
  // parse the input state vector
  GetLine(stateFptr,readLine,'\n',MAX_LINE_LEN);
  sscanf(readLine,"%d",&nSpc);
  if(nSpc != nmechSpc)
    {
      printf("WARNING: number of species in mechanism file %d\n",nmechSpc);
      printf("         differs from state file %d\n",nSpc);
    }
  if(nSpc < nmechSpc)
    {
      printf("ERROR: number of species in mechanism file %d\n",nmechSpc);
      printf("         more than from state file %d\n",nSpc);
      exit(-1);
    }
  nSpc = nmechSpc;

  GetLine(stateFptr,readLine,'\n',MAX_LINE_LEN);
  sscanf(readLine,"%lf",&pres);
  GetLine(stateFptr,readLine,'\n',MAX_LINE_LEN);
  sscanf(readLine,"%lf",&Temp);

  moleFrac   = new double[nSpc];
  massFrac   = new double[nSpc];
  cpSpc      = new double[nSpc];
  hSpc       = new double[nSpc];
  gSpc       = new double[nSpc];
  Kfwd       = new double[nRxn];
  Krev       = new double[nRxn];

  sysParam.cvSpc      = new double[nSpc];
  sysParam.conc = new double[nSpc*nReactors];
  sysParam.internalEnergy = new double[nSpc];
  sysParam.molWt          = new double[nSpc];
  sysParam.stepROP        = new double[nStep*nReactors];
  sysParam.netProdRate    = new double[nSpc*nReactors];
  sysParam.createRate     = new double[nSpc*nReactors];
  sysParam.destroyRate    = new double[nSpc*nReactors];

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
  uMix=mech->getMassIntEnergyFromTY(Temp,massFrac,sysParam.internalEnergy);
  hMix=mech->getMassEnthalpyFromTY(Temp,massFrac,hSpc);
  cpMix=mech->getMassCpFromTY(Temp,massFrac,cpSpc);
  cvMix=mech->getMassCvFromTY(Temp,massFrac,sysParam.cvSpc);
  gasConstant=mech->getGasConstant();

  fprintf(outputFptr,"%24.16e ! [kg/m^3]        density\n",dens);
  fprintf(outputFptr,"%24.16e ! [m^3/kg]        relative volume\n",rvol);
  fprintf(outputFptr,"%24.16e ! [kg/kmol]       molecular weight\n",molwtMix);
  fprintf(outputFptr,"%24.16e ! [Pa]            pressure (from rel volume)\n",
	  presConvert);
  fprintf(outputFptr,"%24.16e ! [J/kg]          mixture internal energy\n",uMix);
  fprintf(outputFptr,"%24.16e ! [J/kg]          mixture enthalpy\n",hMix);
  fprintf(outputFptr,"%24.16e ! [J/kg/K]        specific heat (const vol)\n",
	  cvMix);
  fprintf(outputFptr,"%24.16e ! [J/kg/K]        specific heat (const pres)\n",
	  cpMix);
  fprintf(outputFptr,"%24.16e ! [J/kmol/K]      univerisal gas const\n",
	  gasConstant);

  // calculate species properties
  mech->getMolWtSpc(sysParam.molWt);
  mech->getCfromVY(rvol,massFrac,sysParam.conc);
  mech->getNonDimGibbsFromT(Temp,gSpc);

 double *Temp_array = new double[nReactors];
 double *conc_ptr, *net_ptr, *cre_ptr, *des_ptr, *rop_ptr;

 int prevConcIdx = nSpc-1; //set to 0 for identical concentrations across reactors
 double Temp_mod = 0.1; //set to 0.0 for identical temps across reactors
 Temp_array[0] = Temp;
 for(j=1; j<nReactors; j+=1)
    {
      Temp_array[j] = Temp+j*Temp_mod;
      for( k = 0; k < nSpc; ++k )
        {
          sysParam.conc[j*nSpc+k] = sysParam.conc[(j-1)*nSpc + prevConcIdx];
          ++prevConcIdx;
          if(prevConcIdx >= nSpc) prevConcIdx=0;
        }
    }

 startTime=getHighResolutionTime();

 for( k = 0; k < nEval; ++k)
 {
     for(j=0; j<nReactors; ++j)
     {
          conc_ptr = sysParam.conc + j * nSpc;
          net_ptr = sysParam.netProdRate + j * nSpc;
          cre_ptr = sysParam.createRate + j * nSpc;
          des_ptr = sysParam.destroyRate + j * nSpc;
          rop_ptr = sysParam.stepROP + j * nStep;
          // calculate the reaction rates
          mech->getReactionRates(Temp_array[j],conc_ptr,net_ptr,
  			     cre_ptr,des_ptr,
			     rop_ptr);
     }
 }
 stopTime=getHighResolutionTime();
 double cpuTime = stopTime - startTime;
 printf("# Elapsed time for %d*%d dy/dt calls on CPU [s]: %16.8e\n",nReactors,nEval,
	stopTime-startTime);

double* Temp_array_dev;
double* conc_dev;
double* netProdRate_dev;
double* createRate_dev;
double* destroyRate_dev;
double* stepROP_dev;
cudaMalloc((void**)&Temp_array_dev,sizeof(double)*nReactors);
cudaMalloc((void**)&conc_dev,sizeof(double)*nReactors*nSpc);
cudaMalloc((void**)&netProdRate_dev,sizeof(double)*nReactors*nSpc);
cudaMalloc((void**)&createRate_dev,sizeof(double)*nReactors*nSpc);
cudaMalloc((void**)&destroyRate_dev,sizeof(double)*nReactors*nSpc);
cudaMalloc((void**)&stepROP_dev,sizeof(double)*nReactors*nStep);


 cudaMemcpy(Temp_array_dev,&Temp_array[0],sizeof(double)*nReactors,cudaMemcpyHostToDevice);

 std::vector<double> transposedC(nSpc*nReactors);
 for( k = 0; k < nReactors; ++k)
 {
     for(j=0;j<nSpc;++j)
     {
         transposedC[j*nReactors+k] = sysParam.conc[k*nSpc+j];
     }
 }
 cudaMemcpy(conc_dev,&transposedC[0],sizeof(double)*nReactors*nSpc,cudaMemcpyHostToDevice);

 startTime=getHighResolutionTime();

 for(j=0; j<nEval; ++j) {
      // calculate the reaction rates
      mech->getReactionRates_CUDA_mr_dev(nReactors,
                         Temp_array_dev,
                         conc_dev,
                         netProdRate_dev,
  			 createRate_dev,
                         destroyRate_dev,
                         stepROP_dev);
  }


 stopTime=getHighResolutionTime();
 double gpu_devTime = stopTime - startTime;
 printf("# Elapsed time for %d*%d dy/dt calls on GPU [s]: %16.8e\n",nReactors,nEval,
	stopTime-startTime);

  //Test results
  double *stepROP_cpu        = new double[nStep*nReactors];
  double *netProdRate_cpu    = new double[nSpc*nReactors];
  double *createRate_cpu     = new double[nSpc*nReactors];
  double *destroyRate_cpu    = new double[nSpc*nReactors];

 for(j=0; j<nReactors; ++j)
    {
      conc_ptr = sysParam.conc + j * nSpc;
      net_ptr = netProdRate_cpu + j * nSpc;
      cre_ptr = createRate_cpu + j * nSpc;
      des_ptr = destroyRate_cpu + j * nSpc;
      rop_ptr = stepROP_cpu + j * nStep;
      // calculate the reaction rates
      mech->getReactionRates(Temp_array[j],conc_ptr,net_ptr,
     			     cre_ptr,des_ptr,
			     rop_ptr);
    }

  //We copy the values from the last call inside the testing loop
  double *stepROP_gpu        = new double[nStep*nReactors];
  double *netProdRate_gpu    = new double[nSpc*nReactors];
  double *createRate_gpu     = new double[nSpc*nReactors];
  double *destroyRate_gpu    = new double[nSpc*nReactors];

  cudaMemcpy(stepROP_gpu, stepROP_dev,sizeof(double)*nReactors*nStep,cudaMemcpyDeviceToHost);
  cudaMemcpy(createRate_gpu, createRate_dev,sizeof(double)*nReactors*nSpc,cudaMemcpyDeviceToHost);
  cudaMemcpy(destroyRate_gpu, destroyRate_dev,sizeof(double)*nReactors*nSpc,cudaMemcpyDeviceToHost);
  cudaMemcpy(netProdRate_gpu, netProdRate_dev,sizeof(double)*nReactors*nSpc,cudaMemcpyDeviceToHost);

  cudaFree(Temp_array_dev);
  cudaFree(conc_dev);
  cudaFree(netProdRate_dev);
  cudaFree(createRate_dev);
  cudaFree(destroyRate_dev);
  cudaFree(stepROP_dev);

  // Check gpu results
  double cutoff = 1.0e-30;
  double thresh = 1.0e-10;
  const int max_diff_print = 50;
  int diffcount = 0;
  for( j=0; j<nStep*nReactors; ++j)
  {  
      int currReactor = j/nStep;
      int currStep = j%nStep;
      int j_gpu = currStep*nReactors+currReactor;
      double diff = fabs((stepROP_cpu[j]-stepROP_gpu[j_gpu])/(stepROP_cpu[j]+stepROP_gpu[j_gpu]+cutoff));
      if( diff > thresh ) printf("Diff in ROP [%d]: %g, %g, %g\n",j,stepROP_cpu[j],stepROP_gpu[j_gpu],diff), ++diffcount;
      if(diffcount > max_diff_print) break;      
  }

  diffcount = 0;  
  for( j=0; j<nSpc*nReactors; ++j)
  {  
      double cdiff,ddiff,ndiff,maxdiff;
      int currReactor = j/nSpc;
      int currSpc = j%nSpc;
      int j_gpu = currSpc*nReactors+currReactor;
      cdiff = fabs((createRate_cpu[j]-createRate_gpu[j_gpu])/(createRate_cpu[j]+createRate_gpu[j_gpu]+cutoff));
      ddiff = fabs((destroyRate_cpu[j]-destroyRate_gpu[j_gpu])/(destroyRate_cpu[j]+destroyRate_gpu[j_gpu]+cutoff));
      ndiff = fabs((netProdRate_cpu[j]-netProdRate_gpu[j_gpu])/(netProdRate_cpu[j]+netProdRate_gpu[j_gpu]+cutoff));
      maxdiff = max(cdiff,ddiff);
      if( maxdiff > thresh )
      {
           printf("Diff in createRate [%d]: %g, %g, %g\n",j,createRate_cpu[j],createRate_gpu[j_gpu],cdiff);
           printf("Diff in destroyRate [%d]: %g, %g, %g\n",j,destroyRate_cpu[j],destroyRate_gpu[j_gpu],ddiff);
           printf("Diff in netRate [%d]: %g, %g, %g, %d\n",j,netProdRate_cpu[j],netProdRate_gpu[j_gpu],ndiff,currReactor);
           ++diffcount;
      }
      if(diffcount > max_diff_print) break;      
  }  

  printf(" +++ %d  %d  %g \n",nSpc,nReactors,cpuTime/gpu_devTime);

  delete mech;
  delete [] moleFrac;
  delete [] massFrac;
  delete [] sysParam.conc;
  delete [] sysParam.cvSpc;
  delete [] cpSpc;
  delete [] sysParam.internalEnergy;
  delete [] hSpc;
  delete [] gSpc;
  delete [] sysParam.molWt;
  delete [] sysParam.stepROP;
  delete [] sysParam.netProdRate;
  delete [] sysParam.createRate;
  delete [] sysParam.destroyRate;
  delete [] Kfwd;
  delete [] Krev;

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


