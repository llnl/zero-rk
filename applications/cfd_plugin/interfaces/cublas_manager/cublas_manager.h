#ifndef CUBLAS_MANAGER_H
#define CUBLAS_MANAGER_H

#include <vector>

#include "cublas_v2.h"

#include "../cuda_la_manager/cuda_la_manager.h"

#include "../device_vector/device_vector.h"

template<typename T>
class cublas_manager : public cuda_la_manager<T>
{
 public:
  cublas_manager();
  virtual ~cublas_manager();

  bool factored() { return this->factored_; };

 protected:
  int factor_invert(int num_batches, int n, T* values);
  int factor_lu(int num_batches, int n, T* values);
  int solve_invert(int num_batches, int n, const T* rhs, T* soln);
  int solve_lu(int num_batches, int n, const T* rhs, T* soln);

 private:
  void AllocateDeviceMemory();
  void FreeDeviceMemory();
  void setup_memory();
  int cuda_bdmv(int n, int nbatch, T* A_dev, T* B_dev, T* Y_dev);
  void cuda_transpose(T* odata, const T* idata, const int width, const int height);

  void getrf_batched();
  void getri_batched();
  void getrs_batched();

  bool factored_;
  int n_;
  int num_batches_;

  std::vector<T*> data_ptrs_; //CPU data pointers
  std::vector<T*> tmp_ptrs_; //CPU data pointers
  std::vector<int> info_;

  //device pointers
  zerork::device_vector<int> info_dev_;
  zerork::device_vector<T> tmp_dev_;
  zerork::device_vector<T> matrix_inverse_dev_;
  zerork::device_vector<T*> matrix_inverse_pointers_dev_;
  zerork::device_vector<T*> matrix_pointers_dev_;
  zerork::device_vector<T*> tmp_pointers_dev_;

  cublasHandle_t cublas_handle_;
  cudaError_t cudaStatus_;
};


#endif
