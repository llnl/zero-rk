#include <cassert>
#include "cusolver_rf_manager.h"
#include "../../cuda_err_check.h"


cusolver_rf_manager::cusolver_rf_manager() :
  nnz_(-1),
  n_(-1),
  num_batches_(-1),
  factored_(false)
{
}

cusolver_rf_manager::~cusolver_rf_manager()
{
  if(factored_) {
    FreeDeviceMemory();
    cusolverRfDestroy(cusolverRfHandle_);
  }
}

void cusolver_rf_manager::setup_memory()
{
  if(factored_) {
    FreeDeviceMemory();
  }
  AllocateDeviceMemory();
}

void cusolver_rf_manager::reset() {
  n_ = -1;
  nnz_ = -1;
  num_batches_ = -1;
  slum_.reset();
}

void cusolver_rf_manager::InitializeCusolverRf() {
  status_ = cusolverRfCreate(&cusolverRfHandle_);
  if (status_ != CUSOLVER_STATUS_SUCCESS) {
    printf("[cusovlerRfCreate status %d]\n",status_);
    exit(1);
  }

  status_ = cusolverRfSetResetValuesFastMode(cusolverRfHandle_,
                                             CUSOLVERRF_RESET_VALUES_FAST_MODE_ON);
  if (status_ != CUSOLVER_STATUS_SUCCESS) {
    printf("[cusovlerRfSetResetValuesFastMode status %d]\n",status_);
    exit(1);
  }
  status_ = cusolverRfSetMatrixFormat(cusolverRfHandle_,
                                      CUSOLVERRF_MATRIX_FORMAT_CSR,
                                      CUSOLVERRF_UNIT_DIAGONAL_STORED_L);
  if (status_ != CUSOLVER_STATUS_SUCCESS) {
    printf("[cusovlerRfSetMatrixFormat status %d]\n",status_);
    exit(1);
  }

  status_ = cusolverRfSetAlgs(cusolverRfHandle_,
                              CUSOLVERRF_FACTORIZATION_ALG0,
                              CUSOLVERRF_TRIANGULAR_SOLVE_ALG1);
  if (status_ != CUSOLVER_STATUS_SUCCESS) {
    printf("[cusovlerRfSetAlgs status %d]\n",status_);
    exit(1);
  }

}

void cusolver_rf_manager::AllocateDeviceMemory()
{
  cudaDeviceSynchronize();
  cuda_err_check(cudaGetLastError());

  cuda_err_check(cudaMalloc((void**)&sums_dev_,sizeof(int)*(n_+1)));
  cuda_err_check(cudaMalloc((void**)&indexes_dev_,sizeof(int)*nnz_));
  cuda_err_check(cudaMalloc((void**)&data_dev_,sizeof(double)*nnz_*num_batches_));
  cuda_err_check(cudaMalloc((void**)&col_permutation_dev_,sizeof(int)*n_));
  cuda_err_check(cudaMalloc((void**)&row_permutation_dev_,sizeof(int)*n_));
  cuda_err_check(cudaMalloc((void**)&data_ptrs_dev_,sizeof(double*)*num_batches_));
  cuda_err_check(cudaMalloc((void**)&rhs_ptrs_dev_,sizeof(double*)*num_batches_));
  cuda_err_check(cudaMalloc((void**)&batch_solve_tmp_dev_,sizeof(double)*(2*num_batches_*n_)));
  cuda_err_check(cudaMalloc((void**)&solve_tmp_dev_,sizeof(double)*(n_*num_batches_)));

  data_ptrs_.resize(num_batches_);
  for(int j = 0; j < num_batches_; ++j) {
    data_ptrs_[j] = &solve_tmp_dev_[j*n_];
  }
  cudaMemcpy(rhs_ptrs_dev_, data_ptrs_.data(), sizeof(double*)*num_batches_, cudaMemcpyHostToDevice);

  for(int j = 0; j < num_batches_; ++j) {
    data_ptrs_[j] = &data_dev_[j*nnz_];
  }
  cudaMemcpy(data_ptrs_dev_, data_ptrs_.data(), sizeof(double*)*num_batches_, cudaMemcpyHostToDevice);

  cuda_err_check(cudaGetLastError());
}

void cusolver_rf_manager::FreeDeviceMemory()
{
  cudaFree(sums_dev_);
  cudaFree(indexes_dev_);
  cudaFree(data_dev_);
  cudaFree(col_permutation_dev_);
  cudaFree(row_permutation_dev_);
  cudaFree(data_ptrs_dev_);
  cudaFree(rhs_ptrs_dev_);
  cudaFree(batch_solve_tmp_dev_);
  cudaFree(solve_tmp_dev_);
}

int cusolver_rf_manager::factor(int num_batches,
                                int n,
                                int nnz,
                                const int* indexes,
                                const int* sums,
                                const double* values,
                                matrix_t type)
{
  assert(type == CSR);
  if(!factored_ || nnz != nnz_ || n != n_ || num_batches != num_batches_) {
    n_ = n;
    nnz_ = nnz;
    num_batches_ = num_batches;
    setup_memory();
  }

  if(factored_) {
    cusolverRfDestroy(cusolverRfHandle_);
  }
  InitializeCusolverRf();

  slum_.SetTranspose(true);
  //N.B. : slum ignores CSR right now...
  slum_.factor(n, nnz, indexes, sums, values, superlu_manager::CSR);

  int L_nnz;
  int U_nnz;
  std::vector<int> L_sums;
  std::vector<int> U_sums;
  std::vector<int> L_indexes;
  std::vector<int> U_indexes;
  std::vector<double> L_data;
  std::vector<double> U_data;
  std::vector<int> row_permutation;
  std::vector<int> col_permutation;

  slum_.GetLU(&L_nnz, &L_sums, &L_indexes, &L_data,
              &U_nnz, &U_sums, &U_indexes, &U_data, superlu_manager::CSR, true);
  slum_.GetInversePermutations(&row_permutation, &col_permutation);

  cuda_err_check(cudaMemcpy(row_permutation_dev_, &row_permutation[0], sizeof(int)*n_, cudaMemcpyHostToDevice));
  cuda_err_check(cudaMemcpy(col_permutation_dev_, &col_permutation[0], sizeof(int)*n_, cudaMemcpyHostToDevice));

  cuda_err_check(cudaMemcpy(indexes_dev_, indexes, sizeof(int)*nnz_, cudaMemcpyHostToDevice));
  cuda_err_check(cudaMemcpy(sums_dev_, sums, sizeof(int)*(n_+1), cudaMemcpyHostToDevice));

  for(int j = 0; j < num_batches_; ++j) {
    data_ptrs_[j] = const_cast<double*>(&values[j*nnz_]);
  }
  status_ = cusolverRfBatchSetupHost(num_batches_, n_, nnz_,
                                     const_cast<int*>(sums), const_cast<int*>(indexes),
                                     data_ptrs_.data(),
                                     L_nnz, L_sums.data(), L_indexes.data(), L_data.data(),
                                     U_nnz, U_sums.data(), U_indexes.data(), U_data.data(),
                                     row_permutation.data(), //Left permutation (P)
                                     col_permutation.data(), //Right permutation (Q)
                                     cusolverRfHandle_);
  cudaStatus_ = cudaDeviceSynchronize();
  if ((status_ != CUSOLVER_STATUS_SUCCESS) || (cudaStatus_ != cudaSuccess)) {
    printf("[cusolverRfBatchSetupHost status %d]\n",status_);
    cuda_err_check(cudaStatus_);
    if(status_ == CUSOLVER_STATUS_ALLOC_FAILED) {
       printf("GPU memory allocation failed.  Try with fewer batched reactors.\n");
       exit(1);
    }
  }
  for(int j = 0; j < num_batches_; ++j) {
    data_ptrs_[j] = NULL;
  }

  status_ = cusolverRfBatchAnalyze(cusolverRfHandle_);
  cudaStatus_ = cudaDeviceSynchronize();
  if ((status_ != CUSOLVER_STATUS_SUCCESS) || (cudaStatus_ != cudaSuccess)) {
    printf("[cusovlerRfBatchAnalyze status %d]\n",status_);
    if(status_ == CUSOLVER_STATUS_ALLOC_FAILED) {
       printf("GPU memory allocation failed.  Try with fewer batched reactors.\n");
       exit(1);
    }
  }

  status_ = cusolverRfBatchRefactor(cusolverRfHandle_);
  factored_ = true;
  return status_;
}


int cusolver_rf_manager::refactor(int num_batches, int n, int nnz, const double* values)
{
  if(!factored_ || nnz != nnz_ || num_batches != num_batches_ || n != n_) {
    return 1;
  }

  for(int j = 0; j < num_batches_; ++j) {
    data_ptrs_[j] = const_cast<double*>(&values[j*nnz_]);
  }
  cudaMemcpy(data_ptrs_dev_, data_ptrs_.data(), sizeof(double*)*num_batches_, cudaMemcpyHostToDevice);
  for(int j = 0; j < num_batches_; ++j) {
    data_ptrs_[j] = NULL;
  }

  status_ = cusolverRfBatchResetValues(num_batches,n,nnz,
                                       sums_dev_,
                                       indexes_dev_,
                                       data_ptrs_dev_,
                                       row_permutation_dev_,
                                       col_permutation_dev_,
                                       cusolverRfHandle_);

  status_ = cusolverRfBatchRefactor(cusolverRfHandle_);
  if((status_ != CUSOLVER_STATUS_SUCCESS)) {
    printf("[cusolverRfBatchRefactor status %d]\n",status_);
  }

  std::vector<int> position(num_batches_);
  status_ = cusolverRfBatchZeroPivot(cusolverRfHandle_,position.data());
  if((status_ != CUSOLVER_STATUS_SUCCESS)) {
    printf("[cusolverRfBatchZeroPivot status %d]\n",status_);
    for(int k=0; k<num_batches_; ++k) {
      if(position[k] >= 0) { //zero-pivot in this reactor
        printf("[reactor %d has zero pivot at row,col %d]\n",k,position[k]);
      }
    }
  }

  return status_;
}

//The following modified from cuda sdk-5.0
#define TRANSPOSE_TILE_DIM    32
#define TRANSPOSE_BLOCK_ROWS  8

__global__ void transposeNoBankConflicts(double *odata, const double *idata,
                                         const int width, const int height)
{
  __shared__ double tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM+1];
  int xIndex,yIndex,index_in,index_out;

  xIndex = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;
  yIndex = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;
  index_in = xIndex + (yIndex)*width;

  for (int i=0; i<TRANSPOSE_TILE_DIM; i+=TRANSPOSE_BLOCK_ROWS) {
    if(xIndex < width && yIndex+i < height){
    tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];}
  }

  __syncthreads();

  xIndex = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.x;
  yIndex = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.y;
  index_out = xIndex + (yIndex)*height;

  for (int i=0; i<TRANSPOSE_TILE_DIM; i+=TRANSPOSE_BLOCK_ROWS) {
    if(yIndex+i < width && xIndex < height){
    odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];}
  }
}

int cusolver_rf_manager::solve(int num_batches, int n, const double* rhs, double* soln) {
  if(n != n_ || num_batches != num_batches_) {
    return 1;
  }

  dim3 nBlocks2D,nThreads2D;
  nThreads2D.x = TRANSPOSE_TILE_DIM;
  nThreads2D.y = TRANSPOSE_BLOCK_ROWS;
  nBlocks2D.x = (num_batches+TRANSPOSE_TILE_DIM-1)/TRANSPOSE_TILE_DIM;
  nBlocks2D.y = (n+TRANSPOSE_TILE_DIM-1)/TRANSPOSE_TILE_DIM;
  transposeNoBankConflicts<<<nBlocks2D,nThreads2D>>>(solve_tmp_dev_,rhs,num_batches,n);
  cudaStatus_ = cudaDeviceSynchronize();
  if (cudaStatus_ != cudaSuccess) {
    cuda_err_check(cudaStatus_);
  }

  status_ = cusolverRfBatchSolve(cusolverRfHandle_,
                                 row_permutation_dev_,
                                 col_permutation_dev_,
                                 1, batch_solve_tmp_dev_, n,
                                 rhs_ptrs_dev_, n);
  cudaStatus_ = cudaDeviceSynchronize();

  if ((status_ != CUSOLVER_STATUS_SUCCESS) || (cudaStatus_ != cudaSuccess)) {
    printf ("[cusolverRfBatchSolve status %d]\n",status_);
    cuda_err_check(cudaStatus_);
    if(status_ == CUSOLVER_STATUS_ALLOC_FAILED) {
       printf("GPU memory allocation failed.  Try with fewer batched reactors.\n");
       exit(1);
    }
  }

  nBlocks2D.x = (n+TRANSPOSE_TILE_DIM-1)/TRANSPOSE_TILE_DIM;
  nBlocks2D.y = (num_batches+TRANSPOSE_TILE_DIM-1)/TRANSPOSE_TILE_DIM;
  transposeNoBankConflicts<<<nBlocks2D,nThreads2D>>>(soln,solve_tmp_dev_,n,num_batches);
  return status_;
}


/*// https://stackoverflow.com/questions/15458552/what-is-the-most-efficient-way-to-transpose-a-matrix-in-cuda
  // (accessed 20171220)
    double* dv_ptr_in  = thrust::raw_pointer_cast(data.data());
    double* dv_ptr_out = thrust::raw_pointer_cast(transposed_cuBLAS.data());
    double alpha = 1.;
    double beta  = 0.;
    cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));
    cublasSafeCall(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha, dv_ptr_in, n, &beta, dv_ptr_in, n, dv_ptr_out, m)); 
    print(n, m, transposed_cuBLAS);
*/

