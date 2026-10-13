
#include <assert.h>

#include <algorithm> //std::min
#include "perf_net_kernels.h"
#include "zerork_cuda_defs.h"
#include "scatter_add_kernels.h"

namespace zerork {

// Kernels here.  Wrapper funcs at bottom

void __global__ cuda_setup_memory
(
    const int nReactors,
    const int nSpc,
    double* C_dev,
    const double* C_out_dev,
    double *createOut_dev,
    double *destroyOut_dev
)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = gridDim.x*blockDim.x;
    while(tid < (nSpc+1)*nReactors)
    {
        //setDoubleArrayVal(C_dev,1.0,nSpc*nReactors,(nSpc+1)*nReactors, scatterAddStreams[1]);
        //cudaMemcpyAsync(C_dev,C_out_dev,cpySpcSize,cudaMemcpyDeviceToDevice,scatterAddStreams[1]);

        //cudaMemsetAsync(createOut_out_dev,0,cpySpcSize,scatterAddStreams[0]);
        if(tid < nSpc*nReactors) {
          C_dev[tid] = C_out_dev[tid]; 
          createOut_dev[tid] = 0.0;
          destroyOut_dev[tid] = 0.0;
        } else {
          C_dev[tid] = 1.0;
        }
        tid += stride;
    }
}


void __global__ cuda_rxn_conc_mult
(
    const int nStep,
    const int nReacsPerStep,
    const int *reactantSpcIdxListUnwrapped_dev,
    const double *C_dev,
    double *stepOut_dev
)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < nStep)
    {
        extern __shared__ double C_shared[];
        int local_tid = threadIdx.x*nReacsPerStep;
        for(int j = 0; j < nReacsPerStep; ++j)
        {
            C_shared[local_tid+j] = C_dev[reactantSpcIdxListUnwrapped_dev[j*nStep+tid]];
        }
//        __syncthreads();

        for(int j = 0; j < nReacsPerStep; ++j)
        {
            stepOut_dev[tid]*=C_shared[local_tid+j];
        }
    }
}

void __global__ cuda_step_limiter_mr
(
    const int nReactors,
    const int nStep,
    const double *stepLimiter_dev,
    double *stepOut_dev
)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  const int reactorid = tid % nReactors;
  const int stepid    = tid / nReactors;
  if(stepid < nStep && reactorid < nReactors)
  {
    //stepOut[j] *= step_limiter[j]/(step_limiter[j]+stepOut[j]);
    const double limit = stepLimiter_dev[stepid];
    const double step = stepOut_dev[stepid*nReactors+reactorid];
    stepOut_dev[stepid*nReactors+reactorid] = step*limit/(limit+step);
  }
}


template<int nReacsPerStep>
void __global__ cuda_rxn_conc_mult_mr
(
    const int nReactors,
    const int nSpc,
    const int nStep,
    const int *reactantSpcIdxListUnwrapped_dev,
    const double *C_dev,
    double *stepOut_dev
)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  const int reactorid = tid % nReactors;
  const int stepid    = tid / nReactors;
  if(stepid < nStep && reactorid < nReactors)
  {
    double mult = 1.0;
    for(int j = 0; j < nReacsPerStep; ++j)
    {
      int currIdx = reactantSpcIdxListUnwrapped_dev[j*nStep+stepid]*nReactors+reactorid;
      mult *= C_dev[currIdx];
    }
    stepOut_dev[stepid*nReactors+reactorid] *= mult;
  }
}

template <int nReacsPerStep>
void __global__ cuda_rxn_conc_mult_ni_mr
(
    const int nReactors,
    const int nSpc,
    const int nStep,
    const int *reactantSpcIdxListUnwrapped_dev,
    const double *rop_concentration_powers_dev,
    const double *C_dev,
    double *stepOut_dev
)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  const int reactorid = tid / nStep;
  const int stepid    = tid % nStep;
  if(stepid < nStep && reactorid < nReactors)
  {
    double mult = 1.0;
    for(int j = 0; j < nReacsPerStep; ++j)
    {
      const int currConcIdx = reactantSpcIdxListUnwrapped_dev[j*nStep+stepid]*nReactors + reactorid;
      const double species_concentration = C_dev[currConcIdx];
      const double power = rop_concentration_powers_dev[j*nStep+stepid];
      double val = pow(fabs(species_concentration),power);
      if(species_concentration < 0) {
        val *= -1; 
      }
      mult *= val;
    }
    stepOut_dev[stepid*nReactors+reactorid] *= mult;
  }
}

template <int nReacsPerStep>
void __global__ cuda_rxn_conc_mult_ni_corr_mr
(
    const int nReactors,
    const int nSpc,
    const int nStep,
    const double ford_correction_k,
    const double ford_correction_c0,
    const int *reactantSpcIdxListUnwrapped_dev,
    const double *rop_concentration_powers_dev,
    const double *C_dev,
    double *stepOut_dev
)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  const int reactorid = tid / nStep;
  const int stepid    = tid % nStep;
  if(stepid < nStep && reactorid < nReactors)
  {
    double mult = 1.0;
    double neg = 1.0;
    for(int j = 0; j < nReacsPerStep; ++j)
    {
      const int currConcIdx = reactantSpcIdxListUnwrapped_dev[j*nStep+stepid]*nReactors + reactorid;
      const double species_concentration = C_dev[currConcIdx];
      const double power = rop_concentration_powers_dev[j*nStep+stepid];

      const double kx = ford_correction_k;
      const double x0 = ford_correction_c0;
      const double x = log(max(1.0e-30,species_concentration));
      const double sigmoid = 1/(1+exp(-kx*(x-x0)));
      double correction = sigmoid + (1-sigmoid)*pow(fabs(species_concentration),1-power);

      double val = pow(fabs(species_concentration),power)*correction;
      if(species_concentration < 0) {
        neg = -1.0;
      }
      mult *= val;
    }
    stepOut_dev[stepid*nReactors+reactorid] *= mult*neg;
  }
}

void __global__ cuda_production_rates
(
    const int nSpc,
    const int nSpcOffset,
    const int nSpcRxns,
    const int listOffset,
    const int *productionSpcRxnList_dev,
    const double *stepOut_dev,
    double *productionOut_dev
)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < nSpc)
    {
        double accum = 0.0;
        for(int j=0; j<nSpcRxns; ++j)
        {
              accum+=stepOut_dev[productionSpcRxnList_dev[listOffset+tid*nSpcRxns+j]];
        }
        productionOut_dev[tid+nSpcOffset]=accum;
    }
}



void __global__ cuda_net_rates
(
    const int nSpc,
    const double *createOut_dev,
    const double *destroyOut_dev,
    double *netOut_dev
)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < nSpc)
    {
          netOut_dev[tid] = createOut_dev[tid] - destroyOut_dev[tid];
    }
}


// Wrapper funcs

void perf_net_cuda_setup_memory(const int nReactors, const int nSpc,
                                double* C_dev, const double* C_out_dev,
                                double* createOut_dev, double* destroyOut_dev,
                                cudaStream_t stream)
{
    int nThreads = MAX_THREADS_PER_BLOCK;
    int nBlocks  = ((nSpc+1)*nReactors+nThreads-1)/nThreads;
    cuda_setup_memory<<<nBlocks, nThreads, 0, stream>>>
    (
        nReactors,
        nSpc,
        C_dev,
        C_out_dev,
        createOut_dev,
        destroyOut_dev
    );
#ifdef ZERORK_FULL_DEBUG
    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(),"perf_net_cuda_setup_memory");
#endif
}

void perf_net_cuda_rxn_conc_mult(const int nStep, const int maxReactants,
                                 const int *reactantSpcIdxListUnwrapped_dev,
                                 const double *C_dev, double *stepOut_dev)
{
    int nThreads = MAX_THREADS_PER_BLOCK;
    int nBlocks  = (nStep+nThreads-1)/nThreads;
    cuda_rxn_conc_mult<<< nBlocks, nThreads, (maxReactants*nThreads)*sizeof(double) >>>
    (
        nStep,
        maxReactants,
        reactantSpcIdxListUnwrapped_dev,
        C_dev,
        stepOut_dev
    );
#ifdef ZERORK_FULL_DEBUG
    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(),"perf_net_cuda_rxn_conc_mult");
#endif
}

void perf_net_cuda_step_limiter_mr(const int nReactors,
        const int nStep,
        const double *stepLimiter_dev,
        double *stepOut_dev)
{
    int nThreads = 512;
    int nBlocks  = (nStep*nReactors+nThreads-1)/nThreads;
    cuda_step_limiter_mr<<< nBlocks, nThreads >>> (
           nReactors, nStep, stepLimiter_dev, stepOut_dev);

#ifdef ZERORK_FULL_DEBUG
    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(),"perf_net_cuda_step_limiter_mr");
#endif
}

void perf_net_cuda_rxn_conc_mult_mr(const int nReactors, const int nSpc,
        const int nStep, const int maxReactants,
	const bool ford_correction_derivative,
	const double ford_correction_k,
	const double ford_correction_c0,
        const int *reactantSpcIdxListUnwrapped_dev,
        const double *rop_concentration_powers_dev,
        const double *C_dev,
        double *stepOut_dev)
{
    assert(maxReactants < 9);
    //N.B. in this call we have stepOut with nStep as leading dimension
 
    int nThreads = 512;
    int nBlocks  = (nStep*nReactors+nThreads-1)/nThreads;

    //Scatter multiplication kernel
    if(rop_concentration_powers_dev == nullptr) {
      switch(maxReactants) {
        case 1:
          cuda_rxn_conc_mult_mr<1><<< nBlocks, nThreads >>> (
            nReactors, nSpc, nStep, reactantSpcIdxListUnwrapped_dev,
            C_dev, stepOut_dev);
          break;
        case 2:
          cuda_rxn_conc_mult_mr<2><<< nBlocks, nThreads >>> (
            nReactors, nSpc, nStep, reactantSpcIdxListUnwrapped_dev,
            C_dev, stepOut_dev);
          break;
        case 3:
          cuda_rxn_conc_mult_mr<3><<< nBlocks, nThreads >>> (
            nReactors, nSpc, nStep, reactantSpcIdxListUnwrapped_dev,
            C_dev, stepOut_dev);
          break;
        case 4:
          cuda_rxn_conc_mult_mr<4><<< nBlocks, nThreads >>> (
            nReactors, nSpc, nStep, reactantSpcIdxListUnwrapped_dev,
            C_dev, stepOut_dev);
          break;
        case 5:
          cuda_rxn_conc_mult_mr<5><<< nBlocks, nThreads >>> (
            nReactors, nSpc, nStep, reactantSpcIdxListUnwrapped_dev,
            C_dev, stepOut_dev);
          break;
        case 6:
          cuda_rxn_conc_mult_mr<6><<< nBlocks, nThreads >>> (
            nReactors, nSpc, nStep, reactantSpcIdxListUnwrapped_dev,
            C_dev, stepOut_dev);
          break;
        case 7:
          cuda_rxn_conc_mult_mr<7><<< nBlocks, nThreads >>> (
            nReactors, nSpc, nStep, reactantSpcIdxListUnwrapped_dev,
            C_dev, stepOut_dev);
          break;
        case 8:
          cuda_rxn_conc_mult_mr<8><<< nBlocks, nThreads >>> (
            nReactors, nSpc, nStep, reactantSpcIdxListUnwrapped_dev,
            C_dev, stepOut_dev);
          break;
        default:
          assert(false);
      }
    } else {
      if(ford_correction_derivative) {
        switch(maxReactants) {
          case 1:
            cuda_rxn_conc_mult_ni_corr_mr<1><<< nBlocks, nThreads >>> (
              nReactors, nSpc, nStep, ford_correction_k, ford_correction_c0,
	      reactantSpcIdxListUnwrapped_dev,
              rop_concentration_powers_dev, C_dev, stepOut_dev);
            break;
          case 2:
            cuda_rxn_conc_mult_ni_corr_mr<2><<< nBlocks, nThreads >>> (
              nReactors, nSpc, nStep, ford_correction_k, ford_correction_c0,
	      reactantSpcIdxListUnwrapped_dev,
              rop_concentration_powers_dev, C_dev, stepOut_dev);
            break;
          case 3:
            cuda_rxn_conc_mult_ni_corr_mr<3><<< nBlocks, nThreads >>> (
              nReactors, nSpc, nStep, ford_correction_k, ford_correction_c0,
	      reactantSpcIdxListUnwrapped_dev,
              rop_concentration_powers_dev, C_dev, stepOut_dev);
            break;
          case 4:
            cuda_rxn_conc_mult_ni_corr_mr<4><<< nBlocks, nThreads >>> (
              nReactors, nSpc, nStep, ford_correction_k, ford_correction_c0,
	      reactantSpcIdxListUnwrapped_dev,
              rop_concentration_powers_dev, C_dev, stepOut_dev);
            break;
          case 5:
            cuda_rxn_conc_mult_ni_corr_mr<5><<< nBlocks, nThreads >>> (
              nReactors, nSpc, nStep, ford_correction_k, ford_correction_c0,
	      reactantSpcIdxListUnwrapped_dev,
              rop_concentration_powers_dev, C_dev, stepOut_dev);
            break;
          case 6:
            cuda_rxn_conc_mult_ni_corr_mr<6><<< nBlocks, nThreads >>> (
              nReactors, nSpc, nStep, ford_correction_k, ford_correction_c0,
	      reactantSpcIdxListUnwrapped_dev,
              rop_concentration_powers_dev, C_dev, stepOut_dev);
            break;
          case 7:
            cuda_rxn_conc_mult_ni_corr_mr<7><<< nBlocks, nThreads >>> (
              nReactors, nSpc, nStep, ford_correction_k, ford_correction_c0,
	      reactantSpcIdxListUnwrapped_dev,
              rop_concentration_powers_dev, C_dev, stepOut_dev);
            break;
          case 8:
            cuda_rxn_conc_mult_ni_corr_mr<8><<< nBlocks, nThreads >>> (
              nReactors, nSpc, nStep, ford_correction_k, ford_correction_c0,
	      reactantSpcIdxListUnwrapped_dev,
              rop_concentration_powers_dev, C_dev, stepOut_dev);
            break;
          default:
            assert(false);
        }
      } else {
        switch(maxReactants) {
          case 1:
            cuda_rxn_conc_mult_ni_mr<1><<< nBlocks, nThreads >>> (
              nReactors, nSpc, nStep, reactantSpcIdxListUnwrapped_dev,
              rop_concentration_powers_dev, C_dev, stepOut_dev);
            break;
          case 2:
            cuda_rxn_conc_mult_ni_mr<2><<< nBlocks, nThreads >>> (
              nReactors, nSpc, nStep, reactantSpcIdxListUnwrapped_dev,
              rop_concentration_powers_dev, C_dev, stepOut_dev);
            break;
          case 3:
            cuda_rxn_conc_mult_ni_mr<3><<< nBlocks, nThreads >>> (
              nReactors, nSpc, nStep, reactantSpcIdxListUnwrapped_dev,
              rop_concentration_powers_dev, C_dev, stepOut_dev);
            break;
          case 4:
            cuda_rxn_conc_mult_ni_mr<4><<< nBlocks, nThreads >>> (
              nReactors, nSpc, nStep, reactantSpcIdxListUnwrapped_dev,
              rop_concentration_powers_dev, C_dev, stepOut_dev);
            break;
          case 5:
            cuda_rxn_conc_mult_ni_mr<5><<< nBlocks, nThreads >>> (
              nReactors, nSpc, nStep, reactantSpcIdxListUnwrapped_dev,
              rop_concentration_powers_dev, C_dev, stepOut_dev);
            break;
          case 6:
            cuda_rxn_conc_mult_ni_mr<6><<< nBlocks, nThreads >>> (
              nReactors, nSpc, nStep, reactantSpcIdxListUnwrapped_dev,
              rop_concentration_powers_dev, C_dev, stepOut_dev);
            break;
          case 7:
            cuda_rxn_conc_mult_ni_mr<7><<< nBlocks, nThreads >>> (
              nReactors, nSpc, nStep, reactantSpcIdxListUnwrapped_dev,
              rop_concentration_powers_dev, C_dev, stepOut_dev);
            break;
          case 8:
            cuda_rxn_conc_mult_ni_mr<8><<< nBlocks, nThreads >>> (
              nReactors, nSpc, nStep, reactantSpcIdxListUnwrapped_dev,
              rop_concentration_powers_dev, C_dev, stepOut_dev);
            break;
          default:
            assert(false);
        }
      }
    }
#ifdef ZERORK_FULL_DEBUG
    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(),"perf_net_cuda_rxn_conc_mult_mr");
#endif
}

void perf_net_scatterAdd_gpu_atomic_global(const int nOps,
    const int srcId[], const int destId[], const int nData,
    const int srcSize, const double src[], const int destSize,
    double dest[], cudaStream_t cuStream)
{
    if( nOps == 0 ) return;
    int nThreads = std::min(256,nData);
    dim3 gridBlk;
    gridBlk.x = (nData+nThreads-1)/nThreads;
    gridBlk.y = nOps;
    gridBlk.z = 1;

    scatterAdd_gpu_atomic_global<<<gridBlk, nThreads, 0, cuStream >>>
        (nOps, srcId, destId, nData, srcSize, src, destSize, dest);
#ifdef ZERORK_FULL_DEBUG
    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(),"perf_net_scatterAdd_gpu_atomic_global");
#endif
}

void perf_net_scatterAdd_gpu_atomic_global_2op(const int nOps,
    const int srcId[], const int destId[], const int nData,
    const int srcSize, const double src[], const int destSize,
    double dest[], cudaStream_t cuStream)
{
    if( nOps == 0 ) return;
    int nThreads = std::min(256,nData);
    dim3 gridBlk;
    gridBlk.x = (nData+nThreads-1)/nThreads;
    gridBlk.y = nOps / 2;
    gridBlk.z = 1;

    scatterAdd_gpu_atomic_global_2op<<<gridBlk, nThreads, 0, cuStream >>>
        (nOps, srcId, destId, nData, srcSize, src, destSize, dest);
#ifdef ZERORK_FULL_DEBUG
    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(),"perf_net_scatterAdd_gpu_atomic_global_2op");
#endif
}

void perf_net_scatterAdd_gpu_atomic_global_4op(const int nOps,
    const int srcId[], const int destId[], const int nData,
    const int srcSize, const double src[], const int destSize,
    double dest[], cudaStream_t cuStream)
{
    if( nOps == 0 ) return;
    int nThreads = std::min(256,nData);
    dim3 gridBlk;
    gridBlk.x = (nData+nThreads-1)/nThreads;
    gridBlk.y = nOps / 4;
    gridBlk.z = 1;

    scatterAdd_gpu_atomic_global_4op<<<gridBlk, nThreads, 0, cuStream >>>
        (nOps, srcId, destId, nData, srcSize, src, destSize, dest);
#ifdef ZERORK_FULL_DEBUG
    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(),"perf_net_scatterAdd_gpu_atomic_global_4op");
#endif
}

void perf_net_scatterAdd_gpu_atomic_global_8op(const int nOps,
    const int srcId[], const int destId[], const int nData,
    const int srcSize, const double src[], const int destSize,
    double dest[], cudaStream_t cuStream)
{
    if( nOps == 0 ) return;
    int nThreads = std::min(256,nData);
    dim3 gridBlk;
    gridBlk.x = (nData+nThreads-1)/nThreads;
    gridBlk.y = nOps / 8;
    gridBlk.z = 1;

    scatterAdd_gpu_atomic_global_8op<<<gridBlk, nThreads, 0, cuStream >>>
        (nOps, srcId, destId, nData, srcSize, src, destSize, dest);
#ifdef ZERORK_FULL_DEBUG
    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(),"perf_net_scatterAdd_gpu_atomic_global_8op");
#endif
}

void perf_net_scatterAdd_gpu_atomic_global_16op(const int nOps,
    const int srcId[], const int destId[], const int nData,
    const int srcSize, const double src[], const int destSize,
    double dest[], cudaStream_t cuStream)
{
    if( nOps == 0 ) return;
    int nThreads = std::min(256,nData);
    dim3 gridBlk;
    gridBlk.x = (nData+nThreads-1)/nThreads;
    gridBlk.y = nOps / 16;
    gridBlk.z = 1;

    scatterAdd_gpu_atomic_global_16op<<<gridBlk, nThreads, 0, cuStream >>>
        (nOps, srcId, destId, nData, srcSize, src, destSize, dest);
#ifdef ZERORK_FULL_DEBUG
    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(),"perf_net_scatterAdd_gpu_atomic_global_16op");
#endif
}

void perf_net_scatterAdd_gpu_atomic_global_32op(const int nOps,
    const int srcId[], const int destId[], const int nData,
    const int srcSize, const double src[], const int destSize,
    double dest[], cudaStream_t cuStream)
{
    if( nOps == 0 ) return;
    int nThreads = std::min(256,nData);
    dim3 gridBlk;
    gridBlk.x = (nData+nThreads-1)/nThreads;
    gridBlk.y = nOps / 32;
    gridBlk.z = 1;

    scatterAdd_gpu_atomic_global_32op<<<gridBlk, nThreads, 0, cuStream >>>
        (nOps, srcId, destId, nData, srcSize, src, destSize, dest);
#ifdef ZERORK_FULL_DEBUG
    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(),"perf_net_scatterAdd_gpu_atomic_global_32op");
#endif
}

void perf_net_scatterAdd_gpu_atomic_global_64op(const int nOps,
    const int srcId[], const int destId[], const int nData,
    const int srcSize, const double src[], const int destSize,
    double dest[], cudaStream_t cuStream)
{
    if( nOps == 0 ) return;
    int nThreads = std::min(256,nData);
    dim3 gridBlk;
    gridBlk.x = (nData+nThreads-1)/nThreads;
    gridBlk.y = nOps / 64;
    gridBlk.z = 1;

    scatterAdd_gpu_atomic_global_64op<<<gridBlk, nThreads, 0, cuStream >>>
        (nOps, srcId, destId, nData, srcSize, src, destSize, dest);
#ifdef ZERORK_FULL_DEBUG
    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(),"perf_net_scatterAdd_gpu_atomic_global_64op");
#endif
}

void perf_net_scatterAdd_gpu_atomic_global_128op(const int nOps,
    const int srcId[], const int destId[], const int nData,
    const int srcSize, const double src[], const int destSize,
    double dest[], cudaStream_t cuStream)
{
    if( nOps == 0 ) return;
    int nThreads = std::min(256,nData);
    dim3 gridBlk;
    gridBlk.x = (nData+nThreads-1)/nThreads;
    gridBlk.y = nOps / 128;
    gridBlk.z = 1;

    scatterAdd_gpu_atomic_global_128op<<<gridBlk, nThreads, 0, cuStream >>>
        (nOps, srcId, destId, nData, srcSize, src, destSize, dest);
#ifdef ZERORK_FULL_DEBUG
    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(),"perf_net_scatterAdd_gpu_atomic_global_128op");
#endif
}

void perf_net_scatterAdd_gpu_atomic_global_fused(const int maxOps,
    const int nOps[],
    const int srcId[], const int destId[], const int nData,
    const int srcSize, const double src[], const int destSize,
    double dest[], cudaStream_t cuStream)
{
    if( maxOps == 0 ) return;
    int nThreads = std::min(MAX_THREADS_PER_BLOCK,nData);
    dim3 gridBlk;
    gridBlk.x = (nData+nThreads-1)/nThreads;
    gridBlk.y = maxOps;
    gridBlk.z = 1;

    scatterAdd_gpu_atomic_global_fused<<<gridBlk, nThreads, 0, cuStream >>>
        (nOps, srcId, destId, nData, srcSize, src, destSize, dest);
#ifdef ZERORK_FULL_DEBUG
    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(),"perf_net_scatterAdd_gpu_atomic_global_fused");
#endif
}

void perf_net_multScatterAdd_gpu_atomic_global_fused(const int maxOps,
    const int nOps[],
    const int srcId[], const int destId[], const double srcMult[], const int nData,
    const int srcSize, const double src[], const int destSize,
    double dest[], cudaStream_t cuStream)
{
    if( maxOps == 0 ) return;
    int nThreads = std::min(MAX_THREADS_PER_BLOCK,nData);
    dim3 gridBlk;
    gridBlk.x = (nData+nThreads-1)/nThreads;
    gridBlk.y = maxOps;
    gridBlk.z = 1;

    multScatterAdd_gpu_atomic_global_fused<<<gridBlk, nThreads, 0, cuStream >>>
        (nOps, srcId, destId, srcMult, nData, srcSize, src, destSize, dest);
#ifdef ZERORK_FULL_DEBUG
    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(),"perf_net_scatterAdd_gpu_atomic_global_fused");
#endif
}

void perf_net_cuda_net_rates(const int nSpc, const double *createOut_dev,
    const double *destroyOut_dev, double *netOut_dev)
{
    int nThreads = THREADS_PER_BLOCK;
    int nBlocks  = (nSpc+nThreads-1)/nThreads;
    cuda_net_rates<<< nBlocks, nThreads >>>
    (
        nSpc,
        createOut_dev,
        destroyOut_dev,
        netOut_dev
    );
#ifdef ZERORK_FULL_DEBUG
    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(),"perf_net_cuda_net_rates");
#endif
}

} // namespace zerork

