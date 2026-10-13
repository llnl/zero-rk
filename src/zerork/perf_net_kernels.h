#ifndef ZERORK_PERF_NET_KERNELS
#define ZERORK_PERF_NET_KERNELS


#include <cuda_runtime.h>

namespace zerork {

void perf_net_cuda_setup_memory(const int nReactors, const int nSpc,
                                double* C_dev, const double* C_out_dev,
                                double* createOut_dev, double* destroyOut_dev,
                                cudaStream_t stream);

void perf_net_cuda_step_limiter_mr(const int nReactors,
        const int nStep,
        const double *stepLimiter_dev,
        double *stepOut_dev);

void perf_net_cuda_rxn_conc_mult(const int nStep, const int maxReactants,
                                 const int *reactantSpcIdxListUnwrapped_dev,
                                 const double *C_dev, double *stepOut_dev);

void perf_net_cuda_rxn_conc_mult_mr(const int nReactors, const int nSpc,
        const int nStep, const int maxReactants,
	const bool ford_correction_derivative,
	const double ford_correction_k,
	const double ford_correction_c0,
        const int *reactantSpcIdxListUnwrapped_dev,
        const double *rop_concentration_powers_dev,
        const double *C_dev,
        double *stepOut_dev);

void perf_net_scatterAdd_gpu_atomic_global(const int nOps,
    const int srcId[], const int destId[], const int nData,
    const int srcSize, const double src[], const int destSize,
    double dest[], cudaStream_t cuStream);

void perf_net_scatterAdd_gpu_atomic_global_2op(const int nOps,
    const int srcId[], const int destId[], const int nData,
    const int srcSize, const double src[], const int destSize,
    double dest[], cudaStream_t cuStream);

void perf_net_scatterAdd_gpu_atomic_global_4op(const int nOps,
    const int srcId[], const int destId[], const int nData,
    const int srcSize, const double src[], const int destSize,
    double dest[], cudaStream_t cuStream);

void perf_net_scatterAdd_gpu_atomic_global_8op(const int nOps,
    const int srcId[], const int destId[], const int nData,
    const int srcSize, const double src[], const int destSize,
    double dest[], cudaStream_t cuStream);

void perf_net_scatterAdd_gpu_atomic_global_16op(const int nOps,
    const int srcId[], const int destId[], const int nData,
    const int srcSize, const double src[], const int destSize,
    double dest[], cudaStream_t cuStream);

void perf_net_scatterAdd_gpu_atomic_global_32op(const int nOps,
    const int srcId[], const int destId[], const int nData,
    const int srcSize, const double src[], const int destSize,
    double dest[], cudaStream_t cuStream);

void perf_net_scatterAdd_gpu_atomic_global_64op(const int nOps,
    const int srcId[], const int destId[], const int nData,
    const int srcSize, const double src[], const int destSize,
    double dest[], cudaStream_t cuStream);

void perf_net_scatterAdd_gpu_atomic_global_128op(const int nOps,
    const int srcId[], const int destId[], const int nData,
    const int srcSize, const double src[], const int destSize,
    double dest[], cudaStream_t cuStream);

void perf_net_scatterAdd_gpu_atomic_global_fused(const int maxOps,
    const int nOps[],
    const int srcId[], const int destId[], const int nData,
    const int srcSize, const double src[], const int destSize,
    double dest[], cudaStream_t cuStream);

void perf_net_multScatterAdd_gpu_atomic_global_fused(const int maxOps,
    const int nOps[],
    const int srcId[], const int destId[], const double srcMult[],
    const int nData, const int srcSize, const double src[],
    const int destSize, double dest[], cudaStream_t cuStream);

void perf_net_cuda_net_rates(const int nSpc, const double *createOut_dev,
    const double *destroyOut_dev, double *netOut_dev);

} // namespace zerork

#endif
