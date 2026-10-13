#include <cassert>
#include "perf_net_cuda.h"
#include <algorithm>

#include "zerork_cuda_defs.h"
#include "perf_net_kernels.h"
#include "misc_kernels.h"

#include <sys/time.h>

//#define ZERORK_CUDA_EVENTS
//#define ZERORK_SCATTER_ADD_FUSED

namespace zerork {

static double getHighResolutionTime(void)
{
    struct timeval tod;

    gettimeofday(&tod, NULL);
    double time_seconds = (double) tod.tv_sec + ((double) tod.tv_usec / 1000000.0);
    return time_seconds;
}


perf_net_cuda::perf_net_cuda(info_net &netobj, rate_const &Kobj)
  : perf_net(netobj, Kobj)
{
  int j,k;
  int rCtrU;
  int nReactorsMax = static_cast<rate_const_cuda*>(rateConstPtr)->nReactorsMax();

  reactantSpcIdxListUnwrapped = new int[nStep*maxReactants];

  rCtrU = 0; // reactant species counter (for unwrapped list)
  for(j=0; j<nStep; j++)
  {
    if(non_integer_network_.HasStep(j)) {
      const int num_reac = non_integer_network_.GetNumReactantsOfStep(j);
      for(k = 0; k < num_reac; ++k) {
        reactantSpcIdxListUnwrapped[rCtrU] = non_integer_network_.GetReactantIndexOfStep(j,k);
        ++rCtrU;
      }
      for(k = num_reac; k < maxReactants; ++k) {
        reactantSpcIdxListUnwrapped[rCtrU]  = nSpc;
        ++rCtrU;
      }
    } else {
      for(k=0; k<netobj.getOrderOfStep(j); k++)
      {
        reactantSpcIdxListUnwrapped[rCtrU]  = netobj.getSpecIdxOfStepReactant(j,k);
        ++rCtrU;
      }
      for(k=netobj.getOrderOfStep(j); k < maxReactants; k++)
      {
        reactantSpcIdxListUnwrapped[rCtrU]  = nSpc;
        ++rCtrU;
      }
    }
  }
  assert(rCtrU == nStep*maxReactants);
  //re-order reactantSpcIdxListUnwrapped to optimize rxn_conc_mult kernel

  int* reactantSpcIdxListUnwrapped_tmp = new int[nStep*maxReactants];
  for(j=0; j<nStep; j++)
  {
    for(k=0; k<maxReactants; k++)
    {
      reactantSpcIdxListUnwrapped_tmp[k*nStep+j]= reactantSpcIdxListUnwrapped[j*maxReactants+k];
    }
  }
  memcpy(reactantSpcIdxListUnwrapped, reactantSpcIdxListUnwrapped_tmp, sizeof(int)*maxReactants*nStep);
  delete[] reactantSpcIdxListUnwrapped_tmp;

  

  //Reorder scatter add operations to reduce atomic races in production rate calcs.
  sa64c=reorderScatterAdd_by_ntuple(128, totProd, nStep, nSpc,
                                       productStepIdxList,
                                       productSpcIdxList);
  maxOpsCreate = std::max(sa64c/128,0);

  sa64d=reorderScatterAdd_by_ntuple(128, totReac, nStep, nSpc,
                                       reactantStepIdxList,
                                       reactantSpcIdxList);
  maxOpsDestroy = std::max(sa64d/128,0);

  sa32c=reorderScatterAdd_by_ntuple(64, totProd-sa64c, nStep, nSpc,
                                       &productStepIdxList[sa64c],
                                       &productSpcIdxList[sa64c]);
  maxOpsCreate = std::max(sa32c/64,maxOpsCreate);
  sa32c+=sa64c;

  sa32d=reorderScatterAdd_by_ntuple(64, totReac-sa64d, nStep, nSpc,
                                       &reactantStepIdxList[sa64d],
                                       &reactantSpcIdxList[sa64d]);
  maxOpsDestroy = std::max(sa32d/64,maxOpsDestroy);
  sa32d+=sa64d;

  sa16c=reorderScatterAdd_by_ntuple(32, totProd-sa32c, nStep, nSpc,
                                       &productStepIdxList[sa32c],
                                       &productSpcIdxList[sa32c]);
  maxOpsCreate = std::max(sa16c/32,maxOpsCreate);
  sa16c+=sa32c;

  sa16d=reorderScatterAdd_by_ntuple(32, totReac-sa32d, nStep, nSpc,
                                       &reactantStepIdxList[sa32d],
                                       &reactantSpcIdxList[sa32d]);
  maxOpsDestroy = std::max(sa16d/32,maxOpsDestroy);
  sa16d+=sa32d;

  sa8c=reorderScatterAdd_by_ntuple(16, totProd-sa16c, nStep, nSpc,
                                       &productStepIdxList[sa16c],
                                       &productSpcIdxList[sa16c]);
  maxOpsCreate = std::max(sa8c/16,maxOpsCreate);
  sa8c+=sa16c;

  sa8d=reorderScatterAdd_by_ntuple(16, totReac-sa16d, nStep, nSpc,
                                       &reactantStepIdxList[sa16d],
                                       &reactantSpcIdxList[sa16d]);
  maxOpsDestroy = std::max(sa8d/16,maxOpsDestroy);
  sa8d+=sa16d;

  sa4c=reorderScatterAdd_by_ntuple(8, totProd-sa8c, nStep, nSpc,
                                       &productStepIdxList[sa8c],
                                       &productSpcIdxList[sa8c]);
  maxOpsCreate = std::max(sa4c/8,maxOpsCreate);
  sa4c+=sa8c;

  sa4d=reorderScatterAdd_by_ntuple(8, totReac-sa8d, nStep, nSpc,
                                       &reactantStepIdxList[sa8d],
                                       &reactantSpcIdxList[sa8d]);
  maxOpsDestroy = std::max(sa4d/8,maxOpsDestroy);
  sa4d+=sa8d;

  sa2c=reorderScatterAdd_by_ntuple(4, totProd-sa4c, nStep, nSpc,
                                       &productStepIdxList[sa4c],
                                       &productSpcIdxList[sa4c]);
  maxOpsCreate = std::max(sa2c/4,maxOpsCreate);
  sa2c+=sa4c;

  sa2d=reorderScatterAdd_by_ntuple(4, totReac-sa4d, nStep, nSpc,
                                       &reactantStepIdxList[sa4d],
                                       &reactantSpcIdxList[sa4d]);
  maxOpsDestroy = std::max(sa2d/4,maxOpsDestroy);
  sa2d+=sa4d;

  sa1c=reorderScatterAdd_by_ntuple(2, totProd-sa2c, nStep, nSpc,
                                       &productStepIdxList[sa2c],
                                       &productSpcIdxList[sa2c]);
  maxOpsCreate = std::max(sa1c/2,maxOpsCreate);
  sa1c+=sa2c;

  sa1d=reorderScatterAdd_by_ntuple(2, totReac-sa2d, nStep, nSpc,
                                       &reactantStepIdxList[sa2d],
                                       &reactantSpcIdxList[sa2d]);
  maxOpsDestroy = std::max(sa1d/2,maxOpsDestroy);
  sa1d+=sa2d;
  maxOpsCreate  = std::max(totProd-sa1c,maxOpsCreate);
  maxOpsDestroy = std::max(totReac-sa1d,maxOpsDestroy);

  int* nOpsDestroy = new int[9];
  nOpsDestroy[8] = 0;
  nOpsDestroy[7] = sa64d;
  nOpsDestroy[6] = sa32d;
  nOpsDestroy[5] = sa16d;
  nOpsDestroy[4] = sa8d;
  nOpsDestroy[3] = sa4d;
  nOpsDestroy[2] = sa2d;
  nOpsDestroy[1] = sa1d;
  nOpsDestroy[0] = totReac;

  int* nOpsCreate = new int[9];
  nOpsCreate[8] = 0;
  nOpsCreate[7] = sa64c;
  nOpsCreate[6] = sa32c;
  nOpsCreate[5] = sa16c;
  nOpsCreate[4] = sa8c;
  nOpsCreate[3] = sa4c;
  nOpsCreate[2] = sa2c;
  nOpsCreate[1] = sa1c;
  nOpsCreate[0] = totProd;

  //8 create and 8 destroy kernels  2^[0-7]
  scatterAddCount = 16;
  scatterAddStreams = new cudaStream_t[scatterAddCount];
  for(j = 0; j < scatterAddCount; ++j)
  {
      cudaStreamCreate( &scatterAddStreams[j] );
  }

  checkCudaError
  (
     cudaMalloc((void**)&nOpsDestroy_dev,sizeof(int)*9),
     "cudaMalloc(... nOpsDestroy_dev ...)"
  );
  cudaMemcpy(nOpsDestroy_dev, nOpsDestroy,sizeof(int)*9,cudaMemcpyHostToDevice);
  delete[] nOpsDestroy;

  checkCudaError
  (
     cudaMalloc((void**)&nOpsCreate_dev,sizeof(int)*9),
     "cudaMalloc(... nOpsCreate_dev ...)"
  );
  cudaMemcpy(nOpsCreate_dev, nOpsCreate,sizeof(int)*9,cudaMemcpyHostToDevice);
  delete[] nOpsCreate;

  checkCudaError
  (
     cudaMalloc((void**)&reactantSpcIdxListUnwrapped_dev,sizeof(int)*rCtrU),
     "cudaMalloc(... reactantSpcIdxListUnwrapped_dev ...)"
  );
  cudaMemcpy(reactantSpcIdxListUnwrapped_dev,reactantSpcIdxListUnwrapped,sizeof(int)*rCtrU,cudaMemcpyHostToDevice);

  checkCudaError
  (
     cudaMalloc((void**)&reactantSpcIdxList_dev,sizeof(int)*totReac),
     "cudaMalloc(... reactantSpcIdxList_dev ...)"
  );
  cudaMemcpy(reactantSpcIdxList_dev,reactantSpcIdxList,sizeof(int)*totReac,cudaMemcpyHostToDevice);
  checkCudaError
  (
     cudaMalloc((void**)&reactantStepIdxList_dev,sizeof(int)*totReac),
     "cudaMalloc(... reactantStepIdxList_dev ...)"
  );
  cudaMemcpy(reactantStepIdxList_dev,reactantStepIdxList,sizeof(int)*totReac,cudaMemcpyHostToDevice);


  checkCudaError
  (
     cudaMalloc((void**)&productSpcIdxList_dev,sizeof(int)*totProd),
     "cudaMalloc(... productSpcIdxList_dev ...)"
  );
  cudaMemcpy(productSpcIdxList_dev,productSpcIdxList,sizeof(int)*totProd,cudaMemcpyHostToDevice);

  checkCudaError
  (
     cudaMalloc((void**)&productStepIdxList_dev,sizeof(int)*totProd),
     "cudaMalloc(... productStepIdxList_dev ...)"
  );
  cudaMemcpy(productStepIdxList_dev,productStepIdxList,sizeof(int)*totProd,cudaMemcpyHostToDevice);

  checkCudaError
  (
     cudaMalloc((void**)&C_dev,sizeof(double)*(nSpc+1)*nReactorsMax),
     "cudaMalloc(... C_dev ...)"
  );
  //Set extra concentration to 1.0 to keep scatter mult kernel in sync
  setDoubleArrayVal(C_dev,1.0,nSpc*nReactorsMax,(nSpc+1)*nReactorsMax);

  rop_concentration_powers_dev = nullptr;
  if(use_non_integer_network_) {
    std::vector<double> rop_concentration_powers(nStep*maxReactants, 1.0);
    for(j=0; j<nStep; j++)
    {
      if(non_integer_network_.HasStep(j)) {
        std::vector<double> step_rop_concentration_powers = non_integer_network_.GetRateOfProgressConcentrationPowersOfStep(j);
        const int rop_size = step_rop_concentration_powers.size();
        for(k=0; k<rop_size; k++)
        {
          rop_concentration_powers[k*nStep+j] = step_rop_concentration_powers[k];
        }
        for(k=rop_size; k<maxReactants; k++)
        {
          rop_concentration_powers[k*nStep+j] = 0.0;
        }
      } else {
        for(k=infoPtr->getOrderOfStep(j); k<maxReactants; k++)
        {
          rop_concentration_powers[k*nStep+j] = 0.0;
        }
      }
    }

    checkCudaError
    (
       cudaMalloc((void**)&rop_concentration_powers_dev,sizeof(double)*nStep*maxReactants),
       "cudaMalloc(... rop_concentration_powers_dev ...)"
    );
    cudaMemcpy(rop_concentration_powers_dev,&rop_concentration_powers[0],sizeof(double)*nStep*maxReactants,cudaMemcpyHostToDevice);

    //Reorder non_integer scatter add operations to reduce atomic races in production rate calcs.
    sa64c_ni=reorderScatterAdd_by_ntuple(128, niTotProd, nStep, nSpc,
                                         &niProductStepIdxList[0],
                                         &niProductSpcIdxList[0],
                                         &niProductStoichNumList[0]);
    maxOpsCreate_ni = std::max(sa64c_ni/128,0);

    sa64d_ni=reorderScatterAdd_by_ntuple(128, niTotReac, nStep, nSpc,
                                         &niReactantStepIdxList[0],
                                         &niReactantSpcIdxList[0],
                                         &niReactantStoichNumList[0]);
    maxOpsDestroy_ni = std::max(sa64d_ni/128,0);

    sa32c_ni=reorderScatterAdd_by_ntuple(64, niTotProd-sa64c_ni, nStep, nSpc,
                                         &niProductStepIdxList[sa64c_ni],
                                         &niProductSpcIdxList[sa64c_ni],
                                         &niProductStoichNumList[sa64c_ni]);
    maxOpsCreate_ni = std::max(sa32c_ni/64,maxOpsCreate_ni);
    sa32c_ni+=sa64c_ni;

    sa32d_ni=reorderScatterAdd_by_ntuple(64, niTotReac-sa64d_ni, nStep, nSpc,
                                         &niReactantStepIdxList[sa64d_ni],
                                         &niReactantSpcIdxList[sa64d_ni],
                                         &niReactantStoichNumList[sa64d_ni]);
    maxOpsDestroy_ni = std::max(sa32d_ni/64,maxOpsDestroy_ni);
    sa32d_ni+=sa64d_ni;

    sa16c_ni=reorderScatterAdd_by_ntuple(32, niTotProd-sa32c_ni, nStep, nSpc,
                                         &niProductStepIdxList[sa32c_ni],
                                         &niProductSpcIdxList[sa32c_ni],
                                         &niProductStoichNumList[sa32c_ni]);
    maxOpsCreate_ni = std::max(sa16c_ni/32,maxOpsCreate_ni);
    sa16c_ni+=sa32c_ni;

    sa16d_ni=reorderScatterAdd_by_ntuple(32, niTotReac-sa32d_ni, nStep, nSpc,
                                         &niReactantStepIdxList[sa32d_ni],
                                         &niReactantSpcIdxList[sa32d_ni],
                                         &niReactantStoichNumList[sa32d_ni]);
    maxOpsDestroy_ni = std::max(sa16d_ni/32,maxOpsDestroy_ni);
    sa16d_ni+=sa32d_ni;

    sa8c_ni=reorderScatterAdd_by_ntuple(16, niTotProd-sa16c_ni, nStep, nSpc,
                                         &niProductStepIdxList[sa16c_ni],
                                         &niProductSpcIdxList[sa16c_ni],
                                         &niProductStoichNumList[sa16c_ni]);
    maxOpsCreate_ni = std::max(sa8c_ni/16,maxOpsCreate_ni);
    sa8c_ni+=sa16c_ni;

    sa8d_ni=reorderScatterAdd_by_ntuple(16, niTotReac-sa16d_ni, nStep, nSpc,
                                         &niReactantStepIdxList[sa16d_ni],
                                         &niReactantSpcIdxList[sa16d_ni],
                                         &niReactantStoichNumList[sa16d_ni]);
    maxOpsDestroy_ni = std::max(sa8d_ni/16,maxOpsDestroy_ni);
    sa8d_ni+=sa16d_ni;

    sa4c_ni=reorderScatterAdd_by_ntuple(8, niTotProd-sa8c_ni, nStep, nSpc,
                                         &niProductStepIdxList[sa8c_ni],
                                         &niProductSpcIdxList[sa8c_ni],
                                         &niProductStoichNumList[sa8c_ni]);
    maxOpsCreate_ni = std::max(sa4c_ni/8,maxOpsCreate_ni);
    sa4c_ni+=sa8c_ni;

    sa4d_ni=reorderScatterAdd_by_ntuple(8, niTotReac-sa8d_ni, nStep, nSpc,
                                         &niReactantStepIdxList[sa8d_ni],
                                         &niReactantSpcIdxList[sa8d_ni],
                                         &niReactantStoichNumList[sa8d_ni]);
    maxOpsDestroy_ni = std::max(sa4d_ni/8,maxOpsDestroy_ni);
    sa4d_ni+=sa8d_ni;

    sa2c_ni=reorderScatterAdd_by_ntuple(4, niTotProd-sa4c_ni, nStep, nSpc,
                                         &niProductStepIdxList[sa4c_ni],
                                         &niProductSpcIdxList[sa4c_ni],
                                         &niProductStoichNumList[sa4c_ni]);
    maxOpsCreate_ni = std::max(sa2c_ni/4,maxOpsCreate_ni);
    sa2c_ni+=sa4c_ni;

    sa2d_ni=reorderScatterAdd_by_ntuple(4, niTotReac-sa4d_ni, nStep, nSpc,
                                         &niReactantStepIdxList[sa4d_ni],
                                         &niReactantSpcIdxList[sa4d_ni],
                                         &niReactantStoichNumList[sa4d_ni]);
    maxOpsDestroy_ni = std::max(sa2d_ni/4,maxOpsDestroy_ni);
    sa2d_ni+=sa4d_ni;

    sa1c_ni=reorderScatterAdd_by_ntuple(2, niTotProd-sa2c_ni, nStep, nSpc,
                                         &niProductStepIdxList[sa2c_ni],
                                         &niProductSpcIdxList[sa2c_ni],
                                         &niProductStoichNumList[sa2c_ni]);
    maxOpsCreate_ni = std::max(sa1c_ni/2,maxOpsCreate_ni);
    sa1c_ni+=sa2c_ni;

    sa1d_ni=reorderScatterAdd_by_ntuple(2, niTotReac-sa2d_ni, nStep, nSpc,
                                         &niReactantStepIdxList[sa2d_ni],
                                         &niReactantSpcIdxList[sa2d_ni],
                                         &niReactantStoichNumList[sa2d_ni]);
    maxOpsDestroy_ni = std::max(sa1d_ni/2,maxOpsDestroy_ni);
    sa1d_ni+=sa2d_ni;

    maxOpsCreate_ni  = std::max(niTotProd-sa1c_ni,maxOpsCreate_ni);
    maxOpsDestroy_ni = std::max(niTotReac-sa1d_ni,maxOpsDestroy_ni);

    std::vector<int> nOpsDestroy_ni(9);
    nOpsDestroy_ni[8] = 0;
    nOpsDestroy_ni[7] = sa64d_ni;
    nOpsDestroy_ni[6] = sa32d_ni;
    nOpsDestroy_ni[5] = sa16d_ni;
    nOpsDestroy_ni[4] = sa8d_ni;
    nOpsDestroy_ni[3] = sa4d_ni;
    nOpsDestroy_ni[2] = sa2d_ni;
    nOpsDestroy_ni[1] = sa1d_ni;
    nOpsDestroy_ni[0] = niTotReac;

    std::vector<int> nOpsCreate_ni(9);
    nOpsCreate_ni[8] = 0;
    nOpsCreate_ni[7] = sa64c_ni;
    nOpsCreate_ni[6] = sa32c_ni;
    nOpsCreate_ni[5] = sa16c_ni;
    nOpsCreate_ni[4] = sa8c_ni;
    nOpsCreate_ni[3] = sa4c_ni;
    nOpsCreate_ni[2] = sa2c_ni;
    nOpsCreate_ni[1] = sa1c_ni;
    nOpsCreate_ni[0] = niTotProd;

    checkCudaError
    (
       cudaMalloc((void**)&niReactantSpcIdxList_dev,sizeof(int)*niTotReac),
       "cudaMalloc(... niReactantSpcIdxList_dev ...)"
    );
    cudaMemcpy(niReactantSpcIdxList_dev,&niReactantSpcIdxList[0],sizeof(int)*niTotReac,cudaMemcpyHostToDevice);
    checkCudaError
    (
       cudaMalloc((void**)&niReactantStepIdxList_dev,sizeof(int)*niTotReac),
       "cudaMalloc(... niReactantStepIdxList_dev ...)"
    );
    cudaMemcpy(niReactantStepIdxList_dev,&niReactantStepIdxList[0],sizeof(int)*niTotReac,cudaMemcpyHostToDevice);
    checkCudaError
    (
       cudaMalloc((void**)&niReactantStoichNumList_dev,sizeof(double)*niTotReac),
       "cudaMalloc(... niReactantStoichNumList_dev ...)"
    );
    cudaMemcpy(niReactantStoichNumList_dev,&niReactantStoichNumList[0],sizeof(double)*niTotReac,cudaMemcpyHostToDevice);


    checkCudaError
    (
       cudaMalloc((void**)&niProductSpcIdxList_dev,sizeof(int)*niTotProd),
       "cudaMalloc(... niProductSpcIdxList_dev ...)"
    );
    cudaMemcpy(niProductSpcIdxList_dev,&niProductSpcIdxList[0],sizeof(int)*niTotProd,cudaMemcpyHostToDevice);
    checkCudaError
    (
       cudaMalloc((void**)&niProductStepIdxList_dev,sizeof(int)*niTotProd),
       "cudaMalloc(... niProductStepIdxList_dev ...)"
    );
    cudaMemcpy(niProductStepIdxList_dev,&niProductStepIdxList[0],sizeof(int)*niTotProd,cudaMemcpyHostToDevice);
    checkCudaError
    (
       cudaMalloc((void**)&niProductStoichNumList_dev,sizeof(double)*niTotProd),
       "cudaMalloc(... niProductStoichNumList_dev ...)"
    );
    cudaMemcpy(niProductStoichNumList_dev,&niProductStoichNumList[0],sizeof(double)*niTotProd,cudaMemcpyHostToDevice);

    checkCudaError
    (
       cudaMalloc((void**)&nOpsDestroy_ni_dev,sizeof(int)*9),
       "cudaMalloc(... nOpsDestroy_ni_dev ...)"
    );
    cudaMemcpy(nOpsDestroy_ni_dev, &nOpsDestroy_ni[0],sizeof(int)*9,cudaMemcpyHostToDevice);

    checkCudaError
    (
       cudaMalloc((void**)&nOpsCreate_ni_dev,sizeof(int)*9),
       "cudaMalloc(... nOpsCreate_ni_dev ...)"
    );
    cudaMemcpy(nOpsCreate_ni_dev, &nOpsCreate_ni[0],sizeof(int)*9,cudaMemcpyHostToDevice);
  } //use_non_integer_network_

  //Initialize gpu timing data
  gpuKTime = gpuStepTime = gpuProdTime = gpuNetTime = gpuTxTime = 0.0;
}

perf_net_cuda::~perf_net_cuda()
{
#ifdef ZERORK_CUDA_EVENTS
  double cpuTotTime = cpuKTime + cpuStepTime + cpuProdTime + cpuNetTime;
  double gpuTotTime = gpuKTime + gpuStepTime + gpuProdTime + gpuNetTime + gpuTxTime;
  printf("#Timing data: KTime         StepTime      ProdTime      NetTime       TransferTime  TotalTime    \n");
  printf("#  CPU:       %11.7e %11.7e %11.7e %11.7e %11.7e %11.7e\n",
                   cpuKTime,cpuStepTime,cpuProdTime,cpuNetTime,0.0,cpuTotTime);
  printf("#  GPU:       %11.7e %11.7e %11.7e %11.7e %11.7e %11.7e\n",
                   gpuKTime,gpuStepTime,gpuProdTime,gpuNetTime,gpuTxTime,gpuTotTime);
#endif

  delete [] reactantSpcIdxListUnwrapped;
  cudaFree(reactantSpcIdxListUnwrapped_dev);
  cudaFree(nOpsDestroy_dev);
  cudaFree(nOpsCreate_dev);
  cudaFree(C_dev);
  cudaFree(productSpcIdxList_dev);
  cudaFree(productStepIdxList_dev);
  cudaFree(reactantSpcIdxList_dev);
  cudaFree(reactantStepIdxList_dev);
  if(use_non_integer_network_) {
    cudaFree(rop_concentration_powers_dev);
    cudaFree(niReactantSpcIdxList_dev);
    cudaFree(niReactantStepIdxList_dev);
    cudaFree(niReactantStoichNumList_dev);
    cudaFree(niProductSpcIdxList_dev);
    cudaFree(niProductStepIdxList_dev);
    cudaFree(niProductStoichNumList_dev);
    cudaFree(nOpsDestroy_ni_dev);
    cudaFree(nOpsCreate_ni_dev);
  }
  for(int j = 0; j < scatterAddCount; ++j)
  {
      cudaStreamDestroy( scatterAddStreams[j] );
  }
  delete [] scatterAddStreams;
}


void perf_net_cuda::calcRatesFromTC_CUDA_mr_dev(const int nReactors, const double T_out_dev[], const double C_out_dev[],
	                           const double stepLimiter_dev[],
				   double* netOut_out_dev, double* createOut_out_dev,
				   double* destroyOut_out_dev, double* stepOut_out_dev)
{
    float cudaTime_ms;

    perf_net_cuda_setup_memory(nReactors, nSpc, C_dev, C_out_dev,
                               createOut_out_dev, destroyOut_out_dev,
                               scatterAddStreams[0]);

#ifdef ZERORK_FULL_DEBUG
    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(),"perf_net_cuda::calcRatesFromTC_CUDA_mr_dev( ... ) {cudaMemCpy}");
#endif

#ifdef ZERORK_CUDA_EVENTS
    cudaEvent_t startTxStep,endTxStep,startK,endK,startStep,endStep,
                startTxProd,endTxProd,startProd,endProd,startTxNet,endTxNet,startNet,endNet;
    cudaEventCreate(&startTxStep);
    cudaEventCreate(&endTxStep);
    cudaEventCreate(&startK);
    cudaEventCreate(&endK);
    cudaEventCreate(&startStep);
    cudaEventCreate(&endStep);
    cudaEventCreate(&startTxProd);
    cudaEventCreate(&endTxProd);
    cudaEventCreate(&startProd);
    cudaEventCreate(&endProd);
    cudaEventCreate(&startTxNet);
    cudaEventCreate(&endTxNet);
    cudaEventCreate(&startNet);
    cudaEventCreate(&endNet);

    cudaEventRecord(startK);
#endif
    static_cast<rate_const_cuda*>(rateConstPtr)->updateK_CUDA_mr(nReactors,T_out_dev,C_out_dev,stepOut_out_dev);
#ifdef ZERORK_CUDA_EVENTS
    cudaEventRecord(endK);
#endif

    if(stepLimiter_dev != nullptr) {
      //stepOut[j] *= step_limiter[j]/(step_limiter[j]+stepOut[j]);
      perf_net_cuda_step_limiter_mr(nReactors, nStep, stepLimiter_dev, stepOut_out_dev);
    }

#ifdef ZERORK_CUDA_EVENTS
    cudaEventRecord(startStep);
#endif
    //N.B. this happens in default stream so we don't need to sync the streams
    //  Currently if non-integer reactions exist all reactions are processed as
    //  if they are non-integer.
    bool ford_correction_derivative = non_integer_network_.GetFORDCorrectionDerivative();
    double ford_correction_k = non_integer_network_.GetFORDCorrectionDecay();
    double ford_correction_c0 = non_integer_network_.GetFORDCorrectionOffset();
    perf_net_cuda_rxn_conc_mult_mr(nReactors, nSpc, nStep, maxReactants,
        ford_correction_derivative, ford_correction_k, ford_correction_c0,
        reactantSpcIdxListUnwrapped_dev, rop_concentration_powers_dev, C_dev, stepOut_out_dev);
#ifdef ZERORK_CUDA_EVENTS
    cudaEventRecord(endStep);
#endif

#ifdef ZERORK_CUDA_EVENTS
    cudaEventRecord(startTxStep);
    cudaEventRecord(endTxStep);
    cudaDeviceSynchronize();

    cudaEventSynchronize(endK);
    cudaEventSynchronize(endStep);

    cudaEventElapsedTime(&cudaTime_ms, startK, endK);
    gpuKTime += cudaTime_ms/1000.0;
    cudaEventElapsedTime(&cudaTime_ms, startStep, endStep);
    gpuStepTime += cudaTime_ms/1000.0;

    cudaEventRecord(startProd);
#endif

#ifdef ZERORK_CUDA_EVENTS
    cudaEventRecord(endStep);
#endif

#ifndef ZERORK_SCATTER_ADD_FUSED
        perf_net_scatterAdd_gpu_atomic_global_128op
          (sa64d,
           &reactantStepIdxList_dev[0],
           &reactantSpcIdxList_dev[0],
           nReactors,
           nStep,
           stepOut_out_dev,
           nSpc,
           destroyOut_out_dev, scatterAddStreams[0]);
        perf_net_scatterAdd_gpu_atomic_global_64op
          (sa32d-sa64d,
           &reactantStepIdxList_dev[sa64d],
           &reactantSpcIdxList_dev[sa64d],
           nReactors,
           nStep,
           stepOut_out_dev,
           nSpc,
           destroyOut_out_dev, scatterAddStreams[1]);
        perf_net_scatterAdd_gpu_atomic_global_32op
          (sa16d-sa32d,
           &reactantStepIdxList_dev[sa32d],
           &reactantSpcIdxList_dev[sa32d],
           nReactors,
           nStep,
           stepOut_out_dev,
           nSpc,
           destroyOut_out_dev, scatterAddStreams[2]);
        perf_net_scatterAdd_gpu_atomic_global_16op
          (sa8d-sa16d,
           &reactantStepIdxList_dev[sa16d],
           &reactantSpcIdxList_dev[sa16d],
           nReactors,
           nStep,
           stepOut_out_dev,
           nSpc,
           destroyOut_out_dev, scatterAddStreams[3]);
        perf_net_scatterAdd_gpu_atomic_global_8op
          (sa4d-sa8d,
           &reactantStepIdxList_dev[sa8d],
           &reactantSpcIdxList_dev[sa8d],
           nReactors,
           nStep,
           stepOut_out_dev,
           nSpc,
           destroyOut_out_dev, scatterAddStreams[4]);
         perf_net_scatterAdd_gpu_atomic_global_4op
          (sa2d-sa4d,
           &reactantStepIdxList_dev[sa4d],
           &reactantSpcIdxList_dev[sa4d],
           nReactors,
           nStep,
           stepOut_out_dev,
           nSpc,
           destroyOut_out_dev, scatterAddStreams[5]);
        perf_net_scatterAdd_gpu_atomic_global_2op
          (sa1d-sa2d,
           &reactantStepIdxList_dev[sa2d],
           &reactantSpcIdxList_dev[sa2d],
           nReactors,
           nStep,
           stepOut_out_dev,
           nSpc,
           destroyOut_out_dev, scatterAddStreams[6]);
        perf_net_scatterAdd_gpu_atomic_global
          (totReac-sa1d,
           &reactantStepIdxList_dev[sa1d],
           &reactantSpcIdxList_dev[sa1d],
           nReactors,
           nStep,
           stepOut_out_dev,
           nSpc,
           destroyOut_out_dev, scatterAddStreams[7]);

        perf_net_scatterAdd_gpu_atomic_global_128op
          (sa64c,
           &productStepIdxList_dev[0],
           &productSpcIdxList_dev[0],
           nReactors,
           nStep,
           stepOut_out_dev,
           nSpc,
           createOut_out_dev, scatterAddStreams[8]);
        perf_net_scatterAdd_gpu_atomic_global_64op
          (sa32c-sa64c,
           &productStepIdxList_dev[sa64c],
           &productSpcIdxList_dev[sa64c],
           nReactors,
           nStep,
           stepOut_out_dev,
           nSpc,
           createOut_out_dev, scatterAddStreams[9]);
        perf_net_scatterAdd_gpu_atomic_global_32op
          (sa16c-sa32c,
           &productStepIdxList_dev[sa32c],
           &productSpcIdxList_dev[sa32c],
           nReactors,
           nStep,
           stepOut_out_dev,
           nSpc,
           createOut_out_dev, scatterAddStreams[10]);
        perf_net_scatterAdd_gpu_atomic_global_16op
          (sa8c-sa16c,
           &productStepIdxList_dev[sa16c],
           &productSpcIdxList_dev[sa16c],
           nReactors,
           nStep,
           stepOut_out_dev,
           nSpc,
           createOut_out_dev, scatterAddStreams[11]);
        perf_net_scatterAdd_gpu_atomic_global_8op
          (sa4c-sa8c,
           &productStepIdxList_dev[sa8c],
           &productSpcIdxList_dev[sa8c],
           nReactors,
           nStep,
           stepOut_out_dev,
           nSpc,
           createOut_out_dev, scatterAddStreams[12]);
         perf_net_scatterAdd_gpu_atomic_global_4op
          (sa2c-sa4c,
           &productStepIdxList_dev[sa4c],
           &productSpcIdxList_dev[sa4c],
           nReactors,
           nStep,
           stepOut_out_dev,
           nSpc,
           createOut_out_dev, scatterAddStreams[13]);
        perf_net_scatterAdd_gpu_atomic_global_2op
          (sa1c-sa2c,
           &productStepIdxList_dev[sa2c],
           &productSpcIdxList_dev[sa2c],
           nReactors,
           nStep,
           stepOut_out_dev,
           nSpc,
           createOut_out_dev, scatterAddStreams[14]);
        perf_net_scatterAdd_gpu_atomic_global
          (totProd-sa1c,
           &productStepIdxList_dev[sa1c],
           &productSpcIdxList_dev[sa1c],
           nReactors,
           nStep,
           stepOut_out_dev,
           nSpc,
           createOut_out_dev, scatterAddStreams[15]);
#else
        perf_net_scatterAdd_gpu_atomic_global_fused
          (maxOpsDestroy, nOpsDestroy_dev,
           &reactantStepIdxList_dev[0],
           &reactantSpcIdxList_dev[0],
           nReactors,
           nStep,
           stepOut_out_dev,
           nSpc,
           destroyOut_out_dev, scatterAddStreams[0]);

        perf_net_scatterAdd_gpu_atomic_global_fused
          (maxOpsCreate, nOpsCreate_dev,
           &productStepIdxList_dev[0],
           &productSpcIdxList_dev[0],
           nReactors,
           nStep,
           stepOut_out_dev,
           nSpc,
           createOut_out_dev, scatterAddStreams[8]);
#endif

    if(use_non_integer_network_) {
        perf_net_multScatterAdd_gpu_atomic_global_fused
          (maxOpsDestroy_ni, nOpsDestroy_ni_dev,
           &niReactantStepIdxList_dev[0],
           &niReactantSpcIdxList_dev[0],
           &niReactantStoichNumList_dev[0],
           nReactors,
           nStep,
           stepOut_out_dev,
           nSpc,
           destroyOut_out_dev, scatterAddStreams[0]);

        perf_net_multScatterAdd_gpu_atomic_global_fused
          (maxOpsCreate_ni, nOpsCreate_ni_dev,
           &niProductStepIdxList_dev[0],
           &niProductSpcIdxList_dev[0],
           &niProductStoichNumList_dev[0],
           nReactors,
           nStep,
           stepOut_out_dev,
           nSpc,
           createOut_out_dev, scatterAddStreams[8]);
    }


#ifdef ZERORK_CUDA_EVENTS
    cudaEventRecord(endProd);

    cudaEventRecord(startTxProd);
    cudaEventRecord(endTxProd);

    cudaEventRecord(startNet);
#endif
    //N.B. this happens in default stream so we don't need to sync the streams
    perf_net_cuda_net_rates(nSpc*nReactors, createOut_out_dev,
        destroyOut_out_dev, netOut_out_dev);
#ifdef ZERORK_FULL_DEBUG
    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(),"cuda_net_rates (multi_dev)");
#endif

#ifdef ZERORK_CUDA_EVENTS
    cudaEventRecord(endNet);

    cudaEventRecord(startTxNet);
    cudaEventRecord(endTxNet);

    cudaEventSynchronize(endTxStep);
    cudaEventSynchronize(endTxProd);
    cudaEventSynchronize(endTxNet);

    cudaEventSynchronize(endProd);
    cudaEventSynchronize(endNet);

    cudaEventElapsedTime(&cudaTime_ms, startTxStep, endTxStep);
    gpuTxTime += cudaTime_ms/1000.0;
    cudaEventElapsedTime(&cudaTime_ms, startProd, endProd);
    gpuProdTime += cudaTime_ms/1000.0;
    cudaEventElapsedTime(&cudaTime_ms, startTxProd, endTxProd);
    gpuTxTime += cudaTime_ms/1000.0;
    cudaEventElapsedTime(&cudaTime_ms, startNet, endNet);
    gpuNetTime += cudaTime_ms/1000.0;
    cudaEventElapsedTime(&cudaTime_ms, startTxNet, endTxNet);
    gpuTxTime += cudaTime_ms/1000.0;

    cudaEventDestroy(startTxStep);
    cudaEventDestroy(endTxStep);
    cudaEventDestroy(startK);
    cudaEventDestroy(endK);
    cudaEventDestroy(startStep);
    cudaEventDestroy(endStep);
    cudaEventDestroy(startTxProd);
    cudaEventDestroy(endTxProd);
    cudaEventDestroy(startProd);
    cudaEventDestroy(endProd);
    cudaEventDestroy(startTxNet);
    cudaEventDestroy(endTxNet);
    cudaEventDestroy(startNet);
    cudaEventDestroy(endNet);
#endif

    //N.B. : This seems like it is pointless, but in testing
    //it improves performance.  Hypothesis is that it keeps
    //the cpu from getting to far ahead
    cudaStreamSynchronize((cudaStream_t) 0);
#ifdef ZERORK_FULL_DEBUG
    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(),"perf_net_cuda::getRatesFromTC_mr_dev(): end of routine");
#endif
}


typedef struct
{
  int x_index;
  int y_index;
} indexPair;

typedef struct
{
  int x_index;
  int y_index;
  double z;
} indexTriple;


// int compareXthenY(void *a, void *b);
//
// A comparison function supplied to qsort() that compares the index pairs
// 'a' and 'b' and returns -1, 0 or 1 if 'a' is less than, equal to, or
// greater than 'b'.  The comparison is first done on the x index and then
// the y index.
int compareXthenY(const void *a, const void *b)
{
  indexPair *aptr = (indexPair *)a;
  indexPair *bptr = (indexPair *)b;
  if(aptr->x_index < bptr->x_index)
    {return -1;}  // a < b
  if(aptr->x_index > bptr->x_index)
    {return 1;}   // a < b

  // x_index is equal, now compare y_index
  if(aptr->y_index < bptr->y_index)
    {return -1;}  // a < b
  if(aptr->y_index > bptr->y_index)
    {return 1;}   // a < b

  return 0;
}

// int compareYthenX(void *a, void *b);
//
// A comparison function supplied to qsort() that compares the index pairs
// 'a' and 'b' and returns -1, 0 or 1 if 'a' is less than, equal to, or
// greater than 'b'.  The comparison is first done on the y index and then
// the x index.
static int compareYthenX(const void *a, const void *b)
{
  indexPair *aptr = (indexPair *)a;
  indexPair *bptr = (indexPair *)b;
  if(aptr->y_index < bptr->y_index)
    {return -1;}  // a < b
  if(aptr->y_index > bptr->y_index)
    {return 1;}   // a < b

  // y_index is equal, now compare x_index
  if(aptr->x_index < bptr->x_index)
    {return -1;}  // a < b
  if(aptr->x_index > bptr->x_index)
    {return 1;}   // a < b

  return 0;
}
// void indexPairSort(const int len,
//                    const char firstComp,
//                    int x[],
//                    int y[])
//
// Sort the index lists x and y as pairs sorting them first by x or y
// depending on the character string provided as firstComp.  If firstComp is
// not equal to 'y' or 'Y', then the x-first sort will be used.
static void indexPairSort(const int len,
		   const char firstComp,
		   int x[],
		   int y[])
{
  int j;
  std::vector<indexPair> tmpList(len);

  // store the values of x and y into the index pair list
  for(j=0; j<len; j++)
    {
      tmpList[j].x_index=x[j];
      tmpList[j].y_index=y[j];
    }

  // perform a quick sort on the index pairs
  if(firstComp == 'y' || firstComp == 'Y')
    { // sort by the y_index first
      qsort((void *)&tmpList[0],
	    len,
	    sizeof(indexPair),
	    compareYthenX);
    }
  else
    { // sort by the x_index first (default)
      qsort((void *)&tmpList[0],
	    len,
	    sizeof(indexPair),
	    compareXthenY);
    }

  // copy the sorted result back in the original x and y arrays
  for(j=0; j<len; j++)
    {
      x[j]=tmpList[j].x_index;
      y[j]=tmpList[j].y_index;
    }

}

int compareXthenYthenZ(const void *a, const void *b)
{
  indexTriple *aptr = (indexTriple *)a;
  indexTriple *bptr = (indexTriple *)b;
  if(aptr->x_index < bptr->x_index)
    {return -1;}  // a < b
  if(aptr->x_index > bptr->x_index)
    {return 1;}   // a < b

  // x_index is equal, now compare y_index
  if(aptr->y_index < bptr->y_index)
    {return -1;}  // a < b
  if(aptr->y_index > bptr->y_index)
    {return 1;}   // a < b

  // y_index is equal, now compare z
  if(aptr->z< bptr->z)
    {return -1;}  // a < b
  if(aptr->z> bptr->z)
    {return 1;}   // a < b

  return 0;
}

// int compareYthenXthenZ(void *a, void *b);
static int compareYthenXthenZ(const void *a, const void *b)
{
  indexTriple *aptr = (indexTriple *)a;
  indexTriple *bptr = (indexTriple *)b;
  if(aptr->y_index < bptr->y_index)
    {return -1;}  // a < b
  if(aptr->y_index > bptr->y_index)
    {return 1;}   // a < b

  // y_index is equal, now compare x_index
  if(aptr->x_index < bptr->x_index)
    {return -1;}  // a < b
  if(aptr->x_index > bptr->x_index)
    {return 1;}   // a < b

  // y_index is equal, now compare x_index
  if(aptr->z< bptr->z)
    {return -1;}  // a < b
  if(aptr->z> bptr->z)
    {return 1;}   // a < b

  return 0;
}

// void indexTripleSort(const int len,
//                      const char firstComp,
//                      int x[],
//                      int y[],
//                      int z[])
//
// Sort the index lists x, y, and z as triplets sorting them first by x or y
// depending on the character string provided as firstComp.  If firstComp is
// not equal to 'y' or 'Y', then the x-first sort will be used. z is always
// sorted last.
static void indexTripleSort(const int len,
		   const char firstComp,
		   int x[],
		   int y[],
		   double z[])
{
  int j;
  std::vector<indexTriple> tmpList(len);

  // store the values of x and y into the index pair list
  for(j=0; j<len; j++)
    {
      tmpList[j].x_index=x[j];
      tmpList[j].y_index=y[j];
      tmpList[j].z=z[j];
    }

  // perform a quick sort on the index pairs
  if(firstComp == 'y' || firstComp == 'Y')
    { // sort by the y_index first
      qsort((void *)&tmpList[0],
	    len,
	    sizeof(indexTriple),
	    compareYthenXthenZ);
    }
  else
    { // sort by the x_index first (default)
      qsort((void *)&tmpList[0],
	    len,
	    sizeof(indexTriple),
	    compareXthenYthenZ);
    }

  // copy the sorted result back in the original x and y arrays
  for(j=0; j<len; j++)
    {
      x[j]=tmpList[j].x_index;
      y[j]=tmpList[j].y_index;
      z[j]=tmpList[j].z;
    }
}

// re-order the source and destination index pairs in n-tuples by identical
// destinations.  Destination addresses with operation counts with remainders
// are aggregated at the end. The starting address in the operation list is
// returned.
int perf_net_cuda::reorderScatterAdd_by_ntuple(const int ntuple,
			     const int nOps,
			     const int srcSize,
			     const int destSize,
			     int srcId[],
			     int destId[],
                             double* srcMult)
{
  int j,k;
  int remId,groupId,nRem,nGroup;
  int ntupleCount;
  int *destCount;
  int *newSrcId,*newDestId;
  double *newSrcMult;
  if(nOps == 0) return 0;

  destCount = (int *)malloc(sizeof(int)*destSize);
  newSrcId  = (int *)malloc(sizeof(int)*nOps);
  newDestId = (int *)malloc(sizeof(int)*nOps);
  newSrcMult = (double *)malloc(sizeof(double)*nOps);

  // sort the oepration pairs (destId, srcId) by the destId first
  if(srcMult != nullptr) {
    indexTripleSort(nOps,'x',&destId[0],&srcId[0],srcMult);
  } else {
    indexPairSort(nOps,'x',&destId[0],&srcId[0]);
  }

  // initialize the counter array for the destination indexes
  for(j=0; j<destSize; j++)
    {destCount[j]=0;}

  // count the distribution of destination indexes
  for(j=0; j<nOps; j++)
    {destCount[destId[j]]++;}

  // count the number of complete ntuples
  ntupleCount=0;
  for(j=0; j<destSize; j++)
    {ntupleCount+=(destCount[j]/ntuple);}

  // regroup the sorted index pairs so that all the complete n-tuples
  // are located at the beginning and all the partial n-tuple remainders
  // are at the end
  j = 0;                      // number of operation index pairs places
  groupId = 0;                // starting address of the n-tuple groups
  remId = ntuple*ntupleCount; // starting address of the remainders
  while(j < nOps)
    {
      // nGroup is the number of index pairs in full n-tuples
      // nRem   is the number of leftover index pairs that fail to fill
      //        an n-tuple
      nGroup  = destCount[destId[j]]/ntuple;
      nGroup *= ntuple;
      nRem    = destCount[destId[j]]-nGroup;

      // place the n-tuple groups
      for(k=0; k<nGroup; k++)
	{
	  newDestId[groupId] = destId[j];
	  newSrcId[groupId]  = srcId[j];
          if(srcMult != nullptr) {
            newSrcMult[groupId] =srcMult[j];
          }
	  ++j; ++groupId;
	}
      // place the remainders
      for(k=0; k<nRem; k++)
	{
	  newDestId[remId] = destId[j];
	  newSrcId[remId]  = srcId[j];
          if(srcMult != nullptr) {
            newSrcMult[remId] =srcMult[j];
          }
	  ++j; ++remId;
	}
    }

  // copy the regrouped index pairsback to the
  for(j=0; j<nOps; j++)
    {
      destId[j]  = newDestId[j];
      srcId[j]   = newSrcId[j];
      if(srcMult != nullptr) {
        srcMult[j] = newSrcMult[j];
      }
    }

  //  for(j=0; j<nOps; j++)
  //{
  //  printf("op list %5d: (dest = %5d, src = %5d)\n",
  // 	     j,destId[j],srcId[j]);
  //}
//  printf("# %d-tuple fraction: %8.6f (count %d)\n",
//	 ntuple,
//	 (double)(ntuple*ntupleCount)/(double)nOps,
//	 ntuple*ntupleCount);
	

  free(newDestId);
  free(newSrcId);
  free(newSrcMult);
  free(destCount);

  return ntuple*ntupleCount;
}


} // namespace zerork

