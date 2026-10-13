
#include <cmath> //sqrt
#include <algorithm> //max
#include <cassert>

#include "zerork/zerork_cuda_defs.h"

#include "reactor_nvector_serial_cuda.h"
#include "nvector/nvector_cuda.h"
#include "cuda_transpose.h"

#ifdef ZERORK_HAVE_MAGMA
#include <sunmatrix/sunmatrix_magmadense.h>
#endif

#define ZERORK_ODE_THREADS 512

//#define ZERORK_REACTOR_NON_INTEGER_JACOBIAN_CPU

ReactorNVectorSerialCuda::ReactorNVectorSerialCuda(std::shared_ptr<zerork::mechanism_cuda> mech_ptr) : 
    ReactorBase(),
    mech_ptr_(mech_ptr),
    csrfm_temperature_(),
    csrfm_no_temperature_(),
    csrfm_ptr_(&csrfm_temperature_),
    solve_temperature_(true),
    jacobian_row_indexes_ptr_(&jacobian_row_indexes_temperature_),
    jacobian_column_indexes_ptr_(&jacobian_column_indexes_temperature_),
    jacobian_row_sums_ptr_(&jacobian_row_sums_temperature_),
    jacobian_row_indexes_dev_ptr_(&jacobian_row_indexes_temperature_dev_),
    jacobian_column_indexes_dev_ptr_(&jacobian_column_indexes_temperature_dev_),
    jacobian_row_sums_dev_ptr_(&jacobian_row_sums_temperature_dev_),
    destruction_terms_sparse_indexes_dev_ptr_(&destruction_terms_sparse_indexes_temperature_dev_),
    creation_terms_sparse_indexes_dev_ptr_(&creation_terms_sparse_indexes_temperature_dev_),
    noninteger_sparse_id_ptr_(&noninteger_sparse_id_temperature_),
    noninteger_term_id_dev_ptr_(&noninteger_term_id_temperature_dev_),
    unit_diagonal_dev_ptr_(&unit_diagonal_temperature_dev_)
{
  num_species_ = mech_ptr_->getNumSpecies();
  num_variables_ = num_species_ + 1;
  num_steps_ = mech_ptr_->getNumSteps();
  max_num_reactors_ = mech_ptr_->nReactorsMax();
  num_reactors_ = 1; //place-holder value
  jac_setup_ = false;

  sqrt_unit_round_ = sqrt(UNIT_ROUNDOFF);

  state_data_.assign(num_variables_*max_num_reactors_,0.0);
  state_data_dev_.assign(num_variables_*max_num_reactors_,0.0);
  tmp1_data_.assign(num_variables_*max_num_reactors_,0.0);
  tmp2_data_.assign(num_variables_*max_num_reactors_,0.0);
  tmp3_data_.assign(num_variables_*max_num_reactors_,0.0);
  tmp1_data_dev_.assign(num_variables_*max_num_reactors_,0.0);
  tmp2_data_dev_.assign(num_variables_*max_num_reactors_,0.0);
  tmp3_data_dev_.assign(num_variables_*max_num_reactors_,0.0);

  state_ = N_VMake_Cuda(num_variables_*max_num_reactors_,&state_data_[0], thrust::raw_pointer_cast(&state_data_dev_[0]));
  tmp1_ = N_VMake_Cuda(num_variables_*max_num_reactors_,&tmp1_data_[0], thrust::raw_pointer_cast(&tmp1_data_dev_[0]));
  tmp2_ = N_VMake_Cuda(num_variables_*max_num_reactors_,&tmp2_data_[0], thrust::raw_pointer_cast(&tmp2_data_dev_[0]));
  tmp3_ = N_VMake_Cuda(num_variables_*max_num_reactors_,&tmp3_data_[0], thrust::raw_pointer_cast(&tmp3_data_dev_[0]));

  root_time_ = 0.0;

  inverse_densities_.resize(max_num_reactors_);
  dpdts_.resize(max_num_reactors_);
  mean_cx_mass_.resize(max_num_reactors_);

  mol_wt_.resize(num_species_);
  inv_mol_wt_.resize(num_species_);
  net_production_rates_.resize(num_species_*max_num_reactors_);
  energy_.resize(num_species_*max_num_reactors_);
  cx_mass_.resize(num_species_*max_num_reactors_);
  forward_rates_of_production_.resize(num_steps_*max_num_reactors_);
  creation_rates_.resize(num_species_*max_num_reactors_);
  destruction_rates_.resize(num_species_*max_num_reactors_);
  cx_mass_.resize(max_num_reactors_*num_species_);

  inverse_densities_dev_.resize(max_num_reactors_);
  dpdts_dev_.resize(max_num_reactors_);
  mean_cx_mass_dev_.resize(max_num_reactors_);
  initial_temperatures_dev_.resize(max_num_reactors_);
  initial_energies_dev_.resize(max_num_reactors_);
  e_src_dev_.resize(max_num_reactors_);
  y_src_dev_.resize(max_num_reactors_*num_species_);

  mol_wt_dev_.resize(num_species_);
  inv_mol_wt_dev_.resize(num_species_);
  net_production_rates_dev_.resize(num_species_*max_num_reactors_);
  energy_dev_.resize(num_species_*max_num_reactors_);
  cx_mass_dev_.resize(num_species_*max_num_reactors_);
  forward_rates_of_production_dev_.resize(num_steps_*max_num_reactors_);
  creation_rates_dev_.resize(num_species_*max_num_reactors_);
  destruction_rates_dev_.resize(num_species_*max_num_reactors_);
  cx_mass_dev_.resize(max_num_reactors_*num_species_);
  concentrations_dev_.resize(max_num_reactors_*num_species_);
  temperatures_dev_.resize(max_num_reactors_*num_species_);

  // set constant parameters
  mech_ptr_->getMolWtSpc(&mol_wt_[0]);
  for(int j=0; j < num_species_; ++j) {
    inv_mol_wt_[j]=1.0/mol_wt_[j];
  }

  //Thrust copies to device
  thrust::copy(mol_wt_.begin(), mol_wt_.end(),mol_wt_dev_.begin());
  thrust::copy(inv_mol_wt_.begin(), inv_mol_wt_.end(),inv_mol_wt_dev_.begin());
  //mol_wt_dev_ = mol_wt_;
  //inv_mol_wt_dev_ = inv_mol_wt_;

  step_limiter_.resize(num_steps_);
  thrust::fill(step_limiter_.begin(), step_limiter_.end(), 1.0e22);

  SetupSparseJacobianArrays();

  weights_.assign(max_num_reactors_,1.0);

  int ncmplx_jacs = (7 - 1)/2; //Should query the solver for this number
  cuda_la_manager_z_.resize(ncmplx_jacs);
#ifdef ZERORK_HAVE_MAGMA
  bool use_magma = true;
  int use_cublas_env = 0;
  if(getenv("ZERORK_REACTOR_USE_CUBLAS") != NULL) {
     use_cublas_env = atoi(getenv("ZERORK_REACTOR_USE_CUBLAS"));
     if(use_cublas_env!=0) {
       use_magma = false;
     }
  }

  if(use_magma) {
    cuda_la_manager_ = std::make_unique<magma_manager<double>>();
    for(int i = 0; i < ncmplx_jacs; ++i) {
      cuda_la_manager_z_[i] = std::make_unique<magma_manager<cuDoubleComplex>>();
    }
  } else
#endif
  {
    cuda_la_manager_ = std::make_unique<cublas_manager<double>>();
    for(int i = 0; i < ncmplx_jacs; ++i) {
      cuda_la_manager_z_[i] = std::make_unique<cublas_manager<cuDoubleComplex>>();
    }
  }

  bool use_lu = false;
  int use_lu_env = 0;
  if(getenv("ZERORK_REACTOR_USE_LU") != NULL) {
     use_lu_env = atoi(getenv("ZERORK_REACTOR_USE_LU"));
     if(use_lu_env!=0) {
       use_lu = true;
     }
  }
  cuda_la_manager_->set_lu(use_lu);
  for(int i = 0; i < ncmplx_jacs; ++i) {
    cuda_la_manager_z_[i]->set_lu(use_lu);
  }

}

ReactorNVectorSerialCuda::~ReactorNVectorSerialCuda()
{
  N_VDestroy(state_);
  N_VDestroy(tmp1_);
  N_VDestroy(tmp2_);
  N_VDestroy(tmp3_);
}

N_Vector& ReactorNVectorSerialCuda::GetStateNVectorRef() {
  return state_;
}

void ReactorNVectorSerialCuda::SetSolveTemperature(bool value) {
  if(solve_temperature_ != value) {
    solve_temperature_ = value;
    if(solve_temperature_) {
      num_variables_ = num_species_ + 1;
      nnz_ = nnz_temperature_;

      jacobian_row_sums_ptr_ = &jacobian_row_sums_temperature_;  
      jacobian_row_indexes_ptr_ = &jacobian_row_indexes_temperature_;  
      jacobian_column_indexes_ptr_ = &jacobian_column_indexes_temperature_;  
      noninteger_sparse_id_ptr_ = &noninteger_sparse_id_temperature_;

      jacobian_row_sums_dev_ptr_ = &jacobian_row_sums_temperature_dev_;  
      jacobian_row_indexes_dev_ptr_ = &jacobian_row_indexes_temperature_dev_;  
      jacobian_column_indexes_dev_ptr_ = &jacobian_column_indexes_temperature_dev_;  
      noninteger_term_id_dev_ptr_ = &noninteger_term_id_temperature_dev_;
      destruction_terms_sparse_indexes_dev_ptr_ = &destruction_terms_sparse_indexes_temperature_dev_;
      creation_terms_sparse_indexes_dev_ptr_ = &creation_terms_sparse_indexes_temperature_dev_;
      unit_diagonal_dev_ptr_ = &unit_diagonal_temperature_dev_;

      csrfm_ptr_ = &csrfm_temperature_;
    } else {
      num_variables_ = num_species_;
      nnz_ = nnz_no_temperature_;

      jacobian_row_sums_ptr_ = &jacobian_row_sums_no_temperature_;  
      jacobian_row_indexes_ptr_ = &jacobian_row_indexes_no_temperature_;  
      jacobian_column_indexes_ptr_ = &jacobian_column_indexes_no_temperature_;  
      noninteger_sparse_id_ptr_ = &noninteger_sparse_id_no_temperature_;

      jacobian_row_sums_dev_ptr_ = &jacobian_row_sums_no_temperature_dev_;  
      jacobian_row_indexes_dev_ptr_ = &jacobian_row_indexes_no_temperature_dev_;  
      jacobian_column_indexes_dev_ptr_ = &jacobian_column_indexes_no_temperature_dev_;  
      noninteger_term_id_dev_ptr_ = &noninteger_term_id_no_temperature_dev_;
      destruction_terms_sparse_indexes_dev_ptr_ = &destruction_terms_sparse_indexes_no_temperature_dev_;
      creation_terms_sparse_indexes_dev_ptr_ = &creation_terms_sparse_indexes_no_temperature_dev_;
      unit_diagonal_dev_ptr_ = &unit_diagonal_no_temperature_dev_;

      csrfm_ptr_ = &csrfm_no_temperature_;
    }
    N_VDestroy(state_);
    N_VDestroy(tmp1_);
    N_VDestroy(tmp2_);
    N_VDestroy(tmp3_);
    state_ = N_VMake_Cuda(num_variables_*max_num_reactors_,&state_data_[0], thrust::raw_pointer_cast(&state_data_dev_[0]));
    tmp1_ = N_VMake_Cuda(num_variables_*max_num_reactors_,&tmp1_data_[0], thrust::raw_pointer_cast(&tmp1_data_dev_[0]));
    tmp2_ = N_VMake_Cuda(num_variables_*max_num_reactors_,&tmp2_data_[0], thrust::raw_pointer_cast(&tmp2_data_dev_[0]));
    tmp3_ = N_VMake_Cuda(num_variables_*max_num_reactors_,&tmp3_data_[0], thrust::raw_pointer_cast(&tmp3_data_dev_[0]));
  }
}

static void __global__ RNSC_set_batch_mask(const int num_variables,
                                           const int num_reactors,
                                           const int reactor_idx,
                                           double *batch_mask) {
    int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = gridDim.x*blockDim.x;
    int n = num_variables*num_reactors;
    for( ; tidx < n; tidx += stride) {
      if( tidx > num_variables*reactor_idx && tidx < num_variables*(reactor_idx+1) ){
        batch_mask[tidx] = 1.0;
      } else if (tidx < num_reactors*num_variables) {
        batch_mask[tidx] = 0.0;
      }
    }
}

void ReactorNVectorSerialCuda::SetBatchMaskNVector(int reactor_idx, N_Vector batch_mask) {
  int num_threads = std::min(1024, num_variables_*num_reactors_);
  int num_blocks = (num_reactors_*num_variables_ + num_threads - 1)/num_threads;
  double* tmp_ptr = N_VGetDeviceArrayPointer_Cuda(tmp1_);
  RNSC_set_batch_mask<<<num_blocks,num_threads>>>(num_variables_, num_reactors_, reactor_idx, tmp_ptr);

  double* batch_mask_ptr = N_VGetDeviceArrayPointer_Cuda(batch_mask);
  cuda_transpose(batch_mask_ptr, tmp_ptr, num_reactors_, num_variables_);
}


void ReactorNVectorSerialCuda::SetupSparseJacobianArrays() {
  const int dense_matrix_size = num_variables_*num_variables_;
  std::vector<int> isNonZero(dense_matrix_size);

  // initialize the dense nonzero flags
  for(int j=0; j<num_species_; ++j) {
    isNonZero[j*num_variables_+j]=1;    // mark the diagonal
    isNonZero[j*num_variables_+num_species_]=1; // mark the last row
  }
  for(int k=0; k<num_variables_; ++k) { // mark nSize rows in the last column
    isNonZero[num_species_*num_variables_+k]=1;
  }

  thrust::host_vector<int> destruction_terms_conc_indexes;
  thrust::host_vector<int> destruction_terms_reac_indexes;
  thrust::host_vector<int> destruction_terms_sparse_indexes_temperature;
  thrust::host_vector<int> destruction_terms_sparse_indexes_no_temperature;
  thrust::host_vector<int> creation_terms_conc_indexes;
  thrust::host_vector<int> creation_terms_reac_indexes;
  thrust::host_vector<int> creation_terms_sparse_indexes_temperature;
  thrust::host_vector<int> creation_terms_sparse_indexes_no_temperature;
  // parse the system, filling in the Jacobian term data
  // Jacobian = d ydot(k)/ dy(j)
  for(int j=0; j<num_steps_; ++j) {
    int num_reactants=mech_ptr_->getOrderOfStep(j);
    int num_products=mech_ptr_->getNumProductsOfStep(j);
    for(int k=0; k<num_reactants; ++k) {
      int column_idx=mech_ptr_->getSpecIdxOfStepReactant(j,k); // species being perturbed
      // forward destruction
      for(int m=0; m < num_reactants; ++m) {
        int row_idx=mech_ptr_->getSpecIdxOfStepReactant(j,m); // species being destroyed
        isNonZero[column_idx*num_variables_+row_idx]=1; // mark location in densei

        destruction_terms_conc_indexes.push_back(column_idx);
        destruction_terms_reac_indexes.push_back(j);
        destruction_terms_sparse_indexes_temperature.push_back(row_idx);
        destruction_terms_sparse_indexes_no_temperature.push_back(row_idx);
      }
      // forward creation
      for(int m=0; m < num_products; ++m) {
        int row_idx=mech_ptr_->getSpecIdxOfStepProduct(j,m); // species being created
        isNonZero[column_idx*num_variables_+row_idx]=1; // mark location in dense

        creation_terms_conc_indexes.push_back(column_idx);
        creation_terms_reac_indexes.push_back(j);
        creation_terms_sparse_indexes_temperature.push_back(row_idx);
        creation_terms_sparse_indexes_no_temperature.push_back(row_idx);
      }
    }
  }

  num_noninteger_jacobian_nonzeros_ =
    mech_ptr_->getNonIntegerReactionNetwork()->GetNumJacobianNonzeros();

  std::vector<int> noninteger_row_id;
  std::vector<int> noninteger_column_id;

  // non-integer reaction network
  if(num_noninteger_jacobian_nonzeros_ > 0) {
    noninteger_row_id.assign(num_noninteger_jacobian_nonzeros_, 0);
    noninteger_column_id.assign(num_noninteger_jacobian_nonzeros_, 0);

    mech_ptr_->getNonIntegerReactionNetwork()->GetJacobianPattern(
       &noninteger_row_id[0],&noninteger_column_id[0]);

    for(int j=0; j<num_noninteger_jacobian_nonzeros_; ++j) {
      int dense_id = noninteger_row_id[j]+noninteger_column_id[j]*num_variables_;
      isNonZero[dense_id]=1;
    }
  } // end if(num_noninteger_jacobian_nonzeros_ > 0)

  // count the number of nonzero terms in the dense matrix
  nnz_temperature_=1; // start the count at one so it can still serve as a flag
  jacobian_row_sums_temperature_.assign(num_variables_+1,0);
  jacobian_row_sums_no_temperature_.assign(num_species_+1,0);
  for(int j=0; j<num_variables_; j++) {
    for(int k=0; k<num_variables_; k++) {
      if(isNonZero[k*num_variables_+j]==1) {
        isNonZero[k*num_variables_+j]=nnz_temperature_;
        nnz_temperature_ += 1;
      }
    }
    // after counting column j store the running total in column j+1
    // of the column sum
    jacobian_row_sums_temperature_[j+1]=nnz_temperature_-1;
    if(j < num_species_) {
      jacobian_row_sums_no_temperature_[j+1]=nnz_temperature_-1-(j+1);
    }
  }
  // now at each nonzero term, isNonZero is storing the (address+1) in the
  // actual compressed column storage data array

  nnz_temperature_--; // decrement the count
  nnz_no_temperature_ = nnz_temperature_ - (num_species_+num_variables_);
  nnz_ = nnz_temperature_;

  // allocate column index data
  jacobian_row_indexes_temperature_.assign(nnz_temperature_,0);
  jacobian_row_indexes_no_temperature_.assign(nnz_no_temperature_,0);
  jacobian_column_indexes_temperature_.assign(nnz_temperature_,0);
  jacobian_column_indexes_no_temperature_.assign(nnz_no_temperature_,0);

  // scan the the isNonZero array to determine the row indexes
  // and the special data addresses
  for(int j=0; j<num_variables_; ++j) {
    for(int k=0; k<num_variables_; ++k) {
      int nzAddr=isNonZero[k*num_variables_+j];
      if(nzAddr>0) {
        jacobian_row_indexes_temperature_[nzAddr-1]=j;
        jacobian_column_indexes_temperature_[nzAddr-1]=k;
        if(j < num_species_ && k < num_species_) {
          int nzAddrNoTemp = nzAddr - j; //minus 1 for each temperature elem
          jacobian_row_indexes_no_temperature_[nzAddrNoTemp-1]=j;
          jacobian_column_indexes_no_temperature_[nzAddrNoTemp-1]=k;
        }
      }
    }
  }

  jacobian_row_sums_temperature_dev_ = jacobian_row_sums_temperature_;
  jacobian_row_sums_no_temperature_dev_ = jacobian_row_sums_no_temperature_;
  jacobian_row_indexes_temperature_dev_ = jacobian_row_indexes_temperature_;
  jacobian_row_indexes_no_temperature_dev_ = jacobian_row_indexes_no_temperature_;
  jacobian_column_indexes_temperature_dev_ = jacobian_column_indexes_temperature_;
  jacobian_column_indexes_no_temperature_dev_ = jacobian_column_indexes_no_temperature_;

  // use the isNonZero array as a lookup to store the proper compressed
  // column data storage
  for(int j=0; j<destruction_terms_sparse_indexes_temperature.size(); ++j) {
    int row_idx=destruction_terms_sparse_indexes_temperature[j];
    int column_idx=destruction_terms_conc_indexes[j];
    int nzAddr=isNonZero[column_idx*num_variables_+row_idx];
    destruction_terms_sparse_indexes_temperature[j]=nzAddr-1; // reset to sparse addr
    destruction_terms_sparse_indexes_no_temperature[j]=nzAddr-1 - row_idx; // reset to sparse addr
  }
  destruction_terms_conc_indexes_dev_ = destruction_terms_conc_indexes;
  destruction_terms_reac_indexes_dev_ = destruction_terms_reac_indexes;
  destruction_terms_sparse_indexes_temperature_dev_ = destruction_terms_sparse_indexes_temperature;
  destruction_terms_sparse_indexes_no_temperature_dev_ = destruction_terms_sparse_indexes_no_temperature;

  for(int j=0; j<creation_terms_sparse_indexes_temperature.size(); ++j) {
    int row_idx=creation_terms_sparse_indexes_temperature[j];
    int column_idx=creation_terms_conc_indexes[j];
    int nzAddr=isNonZero[column_idx*num_variables_+row_idx];
    creation_terms_sparse_indexes_temperature[j]=nzAddr-1; // reset to sparse addr
    creation_terms_sparse_indexes_no_temperature[j]=nzAddr-1 - row_idx; // reset to sparse addr
  }
  creation_terms_conc_indexes_dev_ = creation_terms_conc_indexes;
  creation_terms_reac_indexes_dev_ = creation_terms_reac_indexes;
  creation_terms_sparse_indexes_temperature_dev_ = creation_terms_sparse_indexes_temperature;
  creation_terms_sparse_indexes_no_temperature_dev_ = creation_terms_sparse_indexes_no_temperature;



  if(num_noninteger_jacobian_nonzeros_ > 0) {
    noninteger_jacobian_.assign(num_noninteger_jacobian_nonzeros_,0.0);
    noninteger_sparse_id_temperature_.assign(num_noninteger_jacobian_nonzeros_,0);
    noninteger_sparse_id_no_temperature_.assign(num_noninteger_jacobian_nonzeros_,0);

    for(int j=0; j<num_noninteger_jacobian_nonzeros_; ++j) {
      int dense_id = noninteger_row_id[j]+noninteger_column_id[j]*num_variables_;
      int nzAddr=isNonZero[dense_id];
      noninteger_sparse_id_temperature_[j] = nzAddr-1;
      noninteger_sparse_id_no_temperature_[j] = nzAddr-1 - noninteger_row_id[j];
    }

#ifndef ZERORK_REACTOR_NON_INTEGER_JACOBIAN_CPU
    num_noninteger_jacobian_terms_ = mech_ptr_->getNonIntegerReactionNetwork()->GetNumJacobianTerms();
    //Only need temporary copies of these on host
    thrust::host_vector<int> noninteger_internal_term_id(num_noninteger_jacobian_terms_);
    thrust::host_vector<int> noninteger_term_id_temperature(num_noninteger_jacobian_terms_);
    thrust::host_vector<int> noninteger_term_id_no_temperature(num_noninteger_jacobian_terms_);
    thrust::host_vector<int> noninteger_concentration_id(num_noninteger_jacobian_terms_);
    thrust::host_vector<int> noninteger_step_id(num_noninteger_jacobian_terms_);
    thrust::host_vector<double> noninteger_multiplier(num_noninteger_jacobian_terms_);

    mech_ptr_->getNonIntegerReactionNetwork()->GetJacobianParameters(
       &noninteger_internal_term_id[0],
       &noninteger_concentration_id[0],
       &noninteger_step_id[0],
       &noninteger_multiplier[0]);

    for(int j=0; j<num_noninteger_jacobian_terms_; ++j) {
      const int int_idx = noninteger_internal_term_id[j];
      noninteger_term_id_temperature[j] = noninteger_sparse_id_temperature_[int_idx];
      noninteger_term_id_no_temperature[j] = noninteger_sparse_id_no_temperature_[int_idx];
    }

    noninteger_term_id_temperature_dev_ = noninteger_term_id_temperature;
    noninteger_term_id_no_temperature_dev_ = noninteger_term_id_no_temperature;
    noninteger_concentration_id_dev_ = noninteger_concentration_id;
    noninteger_step_id_dev_ = noninteger_step_id;
    noninteger_multiplier_dev_ = noninteger_multiplier;
#endif
  }

  //Make the unit_diagonal array to form preconditioner on device
  thrust::host_vector<double> unit_diagonal_temperature(nnz_temperature_*max_num_reactors_,0.0);
  for(int k = 0;  k < max_num_reactors_; ++k) {
    for(int j = 0; j < num_variables_; ++j) {
       int row = j + k*num_variables_;
       for(int m = jacobian_row_sums_temperature_[j]; m < jacobian_row_sums_temperature_[j+1]; ++m) {
         int col = jacobian_column_indexes_temperature_[m] + k*num_variables_;
         if(row == col) {
           unit_diagonal_temperature[m+k*nnz_temperature_] = 1.0;
         }
       }
    }
  }
  thrust::host_vector<double> unit_diagonal_no_temperature(nnz_no_temperature_*max_num_reactors_,0.0);
  for(int k = 0;  k < max_num_reactors_; ++k) {
    for(int j = 0; j < num_species_; ++j) {
       int row = j + k*num_species_;
       for(int m = jacobian_row_sums_no_temperature_[j]; m < jacobian_row_sums_no_temperature_[j+1]; ++m) {
         int col = jacobian_column_indexes_no_temperature_[m] + k*num_species_;
         if(row == col) {
           unit_diagonal_no_temperature[m+k*nnz_no_temperature_] = 1.0;
         }
       }
    }
  }
  //Thrust copies to device
  unit_diagonal_temperature_dev_ = unit_diagonal_temperature;
  unit_diagonal_no_temperature_dev_ = unit_diagonal_no_temperature;
} 

static __device__ int cuda_kernel_ret;

static void __global__ RNSC_scale_vector(const int n, const double alpha, double *x) {
    int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = gridDim.x*blockDim.x;
    for( ; tidx < n; tidx += stride) {
        x[tidx] *= alpha;
    }
}

static void __global__ RNSC_check_temperatures(const int n, const double check_val_low,
                                               const double check_val_high, double *T_dev) {
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = gridDim.x*blockDim.x;
  for( ; tidx < n; tidx += stride) {
      if(T_dev[tidx] <= check_val_low) {
          cuda_kernel_ret = 1;
      }
      if(T_dev[tidx] > check_val_high) {
          T_dev[tidx] = check_val_high;
      }
  }
}

static void __global__ RNSC_set_temperatures(const int n, const double ref_temp, const double check_val_low,
                                               const double check_val_high, const double* T_scaled_dev, double *T_dev) {
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = gridDim.x*blockDim.x;
  for( ; tidx < n; tidx += stride) {
      double T_scaled_val = T_scaled_dev[tidx];
      double T_val = T_scaled_val*ref_temp;
      if(T_val <= check_val_low) {
          T_val = check_val_low;
      } else if(T_val > check_val_high) {
          T_val = check_val_high;
      }
      T_dev[tidx] = T_val;
  }
}

static void __global__ RNSC_check_mass_fractions(const int n,
                                                 const double check_val,
                                                 const double *y_dev) {
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = gridDim.x*blockDim.x;
  for( ; tidx < n; tidx += stride) {
    if(y_dev[tidx] < check_val) {
        cuda_kernel_ret = 1;
    }
  }
}

static void __global__ RNSC_concentration_derivative(const int num_reactors,
                                                     const int num_species, 
                                                     const double *net_production_rates_dev,
                                                     const double *mol_wt_dev,
                                                     const double *inverse_densities_dev,
                                                     double *ydot_dev) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int reactorid = blockIdx.y*blockDim.y + threadIdx.y;
  if(reactorid < num_reactors) {
    if(tid < num_species) {
      ydot_dev[num_reactors*tid+reactorid] = net_production_rates_dev[num_reactors*tid+reactorid]
                                             *mol_wt_dev[tid]*inverse_densities_dev[reactorid];
    }
  }
}

static void __global__ RNSC_concentration_derivative_with_source(const int num_reactors,
                                                                 const int num_species, 
                                                                 const double *net_production_rates_dev,
                                                                 const double *mol_wt_dev,
                                                                 const double *inverse_densities_dev,
                                                                 const double *y_src,
                                                                 double *ydot_dev) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int reactorid = blockIdx.y*blockDim.y + threadIdx.y;
  if(reactorid < num_reactors) {
    if(tid < num_species) {
      ydot_dev[num_reactors*tid+reactorid] = net_production_rates_dev[num_reactors*tid+reactorid]
                                             *mol_wt_dev[tid]*inverse_densities_dev[reactorid]
                                             +y_src[num_reactors*tid+reactorid];
    }
  }
}

static void __global__ RNSC_temperature_derivative_no_source(const int num_reactors, const int num_species,
                                                   const double gas_constant, const double reference_temperature,
                                                   const double *energy_dev,
                                                   const double *net_production_rates_dev,
                                                   const double *inverse_densities_dev,
                                                   const double *mean_cx_mass_dev,
                                                   const double *dpdts_dev,
                                                   const double *y_dev,
                                                   double *ydot_dev) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid < num_reactors) {
    double tdot = 0.0;
    for(int k=0; k < num_species; ++k) {
      tdot += energy_dev[num_reactors*k+tid]*net_production_rates_dev[num_reactors*k+tid];
    }
    tdot *= -1*gas_constant*y_dev[num_species*num_reactors+tid]
                               *inverse_densities_dev[tid]/mean_cx_mass_dev[tid];
    tdot += (dpdts_dev[tid]*inverse_densities_dev[tid])/(mean_cx_mass_dev[tid]*reference_temperature);
    ydot_dev[num_species*num_reactors+tid] = tdot;
  }
}

static void __global__ RNSC_temperature_derivative(const int num_reactors, const int num_species,
                                                   const double gas_constant, const double reference_temperature,
                                                   const double *energy_dev,
                                                   const double *net_production_rates_dev,
                                                   const double *inverse_densities_dev,
                                                   const double *mean_cx_mass_dev,
                                                   const double *inv_mol_wt_dev,
                                                   const double *dpdts_dev,
                                                   const double *e_src_dev,
                                                   const double *y_src_dev,
                                                   const double *y_dev,
                                                   double *ydot_dev) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid < num_reactors) {
    double tdot = 0.0;
    const double inverse_density_local = inverse_densities_dev[tid];
    const double mean_cx_mass_local = mean_cx_mass_dev[tid];
    if(y_src_dev != nullptr) {
      for(int k=0; k < num_species; ++k) {
        tdot += energy_dev[num_reactors*k+tid]*(net_production_rates_dev[num_reactors*k+tid]*inverse_density_local
                                               +y_src_dev[num_reactors*k+tid]*inv_mol_wt_dev[k]);
      }
    } else {
      for(int k=0; k < num_species; ++k) {
        tdot += energy_dev[num_reactors*k+tid]*net_production_rates_dev[num_reactors*k+tid]*inverse_density_local;
      }
    }
    tdot *= -1*gas_constant*y_dev[num_species*num_reactors+tid]/mean_cx_mass_local;
    if(dpdts_dev != nullptr) {
      tdot += (dpdts_dev[tid]*inverse_density_local)/(mean_cx_mass_local*reference_temperature);
    }
    if(e_src_dev != nullptr) {
      tdot += e_src_dev[tid]/(mean_cx_mass_local*reference_temperature);
    }
    ydot_dev[num_species*num_reactors+tid] = tdot;
  }
}

int ReactorNVectorSerialCuda::CheckMassFractionsDevice(const double* y) {
  //Check for large negative mass fracs and try to get CVODE to recover if found
  int num_threads = std::min(MAX_THREADS_PER_BLOCK,num_reactors_*num_species_);
  int num_blocks = (num_reactors_*num_species_ + num_threads -1)/num_threads;

  int neg_mass_fracs = 0;
  cudaMemcpyToSymbol(cuda_kernel_ret,&neg_mass_fracs,sizeof(int));
  const double CHECK_VAL = -1.0e-10;
  RNSC_check_mass_fractions<<<num_blocks,num_threads>>>(num_reactors_*num_species_,CHECK_VAL,y);
  cudaMemcpyFromSymbol(&neg_mass_fracs,cuda_kernel_ret,sizeof(int));
  if(neg_mass_fracs == 1) {
    if(int_options_["verbosity"] > 4) { 
      printf("WARNING: Significantly negative species in "
             "Zero-RK RHS.  Consider reducing tolerances.\n");
    } 
  }
  return neg_mass_fracs;
}

int ReactorNVectorSerialCuda::SetTemperatures(const double* scaled_temperatures, double* temperatures) {
  const double TLIMIT = 1.0e6;
  const double reference_temperature = double_options_["reference_temperature"];
#if 0
  thrust::device_ptr<double> temperatures_dev_ptr(&(y_ptr_dev[num_species_*num_reactors_]));
  thrust::copy(temperatures_dev_ptr, temperatures_dev_ptr + num_reactors_, temperatures_dev_.begin());

  num_threads = ZERORK_ODE_THREADS;
  num_blocks = (num_reactors_ + num_threads -1)/num_threads;
  RCVG_scale_vector<<<num_blocks,num_threads>>>(num_reactors_, reference_temperature,
                                                thrust::raw_pointer_cast(&temperatures_dev_[0]));

  int bad_temp_flag = 0;
  cudaMemcpyToSymbol(cuda_kernel_ret,&bad_temp_flag,sizeof(int));
  num_threads = ZERORK_ODE_THREADS;
  num_blocks = (num_reactors_ + num_threads-1)/num_threads;
  RCVG_check_temperatures<<<num_blocks,num_threads>>>(num_reactors_, 0.0, TLIMIT, thrust::raw_pointer_cast(&temperatures_dev_[0]));
  cudaMemcpyFromSymbol(&bad_temp_flag,cuda_kernel_ret,sizeof(int));
  if(bad_temp_flag == 1) return 1;
#else
  int num_threads = ZERORK_ODE_THREADS;
  int num_blocks = (num_reactors_ + num_threads-1)/num_threads;
  RNSC_set_temperatures<<<num_blocks,num_threads>>>(num_reactors_, reference_temperature, 0.0, TLIMIT,
                                                    scaled_temperatures, temperatures);
#endif
  return 0;
}

int ReactorNVectorSerialCuda::ConcentrationDerivative(const double* inverse_densities, double *ydot) {
  // ydot = [kmol/m^3/s] * [kg/kmol] * [m^3/kg] = [(kg spec j)/(kg mix)/s]
  dim3 num_threads_2D, num_blocks_2D;
  num_threads_2D.x = 1;
  num_threads_2D.y = std::min((unsigned int) num_reactors_, MAX_THREADS_PER_BLOCK/num_threads_2D.x);
  num_blocks_2D.x = (num_species_+num_threads_2D.x-1)/num_threads_2D.x;
  num_blocks_2D.y = (num_reactors_+num_threads_2D.y-1)/num_threads_2D.y;

  double * net_production_rates = thrust::raw_pointer_cast(&net_production_rates_dev_[0]);
  double * mol_wt = thrust::raw_pointer_cast(&mol_wt_dev_[0]);
  if(y_src_dev_.size() != 0) {
    double* y_src = thrust::raw_pointer_cast(&y_src_dev_[0]);
    RNSC_concentration_derivative_with_source<<<num_blocks_2D,num_threads_2D>>>(num_reactors_, num_species_,
                                                                    net_production_rates, mol_wt,
                                                                    inverse_densities, y_src, ydot);
  } else {
    RNSC_concentration_derivative<<<num_blocks_2D,num_threads_2D>>>(num_reactors_, num_species_,
                                                                    net_production_rates, mol_wt,
                                                                    inverse_densities, ydot);
  }
#ifdef ZERORK_FULL_DEBUG
  cudaDeviceSynchronize();
  zerork::checkCudaError(cudaGetLastError(),"wsr_conc_deriv");
#endif
  return 0;
}


int ReactorNVectorSerialCuda::TemperatureDerivative(const double* inverse_densities, const double* y, double* ydot) {
  int num_threads = ZERORK_ODE_THREADS;
  int num_blocks = (num_reactors_ + num_threads -1)/num_threads;

  double * energy = thrust::raw_pointer_cast(&energy_dev_[0]);
  double * net_production_rates = thrust::raw_pointer_cast(&net_production_rates_dev_[0]);
  double * inv_mol_wt = thrust::raw_pointer_cast(&inv_mol_wt_dev_[0]);
  double * mean_cx_mass = thrust::raw_pointer_cast(&mean_cx_mass_dev_[0]);

  double* e_src_ptr = nullptr;
  if(e_src_dev_.size() != 0) {
    e_src_ptr = thrust::raw_pointer_cast(&e_src_dev_[0]);
  }
  double* dpdts_ptr = nullptr;
  if(dpdts_dev_.size() != 0) {
    dpdts_ptr = thrust::raw_pointer_cast(&dpdts_dev_[0]);
  }
  double* y_src_ptr = nullptr;
  if(y_src_dev_.size() != 0) {
    y_src_ptr = thrust::raw_pointer_cast(&y_src_dev_[0]);
  }
  RNSC_temperature_derivative<<<num_blocks,num_threads,sizeof(double)*num_threads>>>(num_reactors_,num_species_,mech_ptr_->getGasConstant(),
                                                                                     double_options_["reference_temperature"],
                                                                                     energy,net_production_rates,
                                                                                     inverse_densities, mean_cx_mass,
                                                                                     inv_mol_wt,
                                                                                     dpdts_ptr, e_src_ptr, y_src_ptr,
                                                                                     y, ydot);
#ifdef ZERORK_FULL_DEBUG
  cudaDeviceSynchronize();
  zerork::checkCudaError(cudaGetLastError(),"wsr_T_deriv");
#endif
  return 0;
}

#define ZERORK_RNVSC_PERTURB_V2

#ifndef ZERORK_RNVSC_PERTURB_V2
void __global__ RNVSC_perturb_y(const int block_size, const int batch_size,
                                const double uround, const int col,
                                double* y_perturb, double* inv_increments)
{
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = gridDim.x*blockDim.x;
  for( ; tidx < batch_size; tidx += stride)
  {
    // N.B.:  Interleaved data
    int data_idx = batch_size*col+tidx;
    double yval = y_perturb[data_idx];
    double delta=sqrt(uround*max(1.0e-5,fabs(yval)));
    y_perturb[data_idx] = yval+delta;
    delta = 1.0/delta;
    for(int i = 0; i < block_size; ++i)
    {   
      inv_increments[tidx+batch_size*i] = delta;
    }
  }
}
#else
void __global__ RNVSC_perturb_y(const int block_size, const int batch_size,
                                const double uround, const int pcol,
                                double* y, double* y_perturb, double* inv_increments)
{
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = gridDim.x*blockDim.x;
  for( ; tidx < block_size*batch_size; tidx += stride)
  {
    // N.B.:  Interleaved data
    int col = tidx / batch_size; //var
    int row = tidx % batch_size; //block
    double yval = y[tidx];
    if(col == pcol) {
      double delta=sqrt(uround*max(1.0e-5,fabs(yval)));
      yval += delta;
      for(int i = 0; i < block_size; ++i) {
        inv_increments[row+batch_size*i] = 1.0/delta;
      }
    }
    y_perturb[tidx] = yval;
  }
}
#endif


void __global__ distribute_jac_terms_kernel
(
  const int block_size,
  const int batch_size,
  const int col,
  const realtype* jac_compressed_col,
  realtype* jac_full
)
{
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = gridDim.x*blockDim.x;

  for( ; tidx < block_size*batch_size; tidx += stride)
  {
    int local_row   = tidx % block_size;
    int local_block = tidx / block_size;

    int in_idx =  tidx;
    int out_idx = block_size*block_size*local_block + block_size*col + local_row;
    jac_full[out_idx] = jac_compressed_col[in_idx];
  }
}

void distribute_jac_terms
(
  const int block_size,
  const int batch_size,
  const int col,
  const realtype* jac_compressed_col,
  realtype* jac_full
)
{
    int threads = 1024;
    int blocks  = (batch_size*block_size+threads-1)/threads;
    distribute_jac_terms_kernel<<<blocks,threads>>>(block_size,batch_size,col,jac_compressed_col,jac_full);
#ifdef ZERORK_FULL_DEBUG
    cudaErrChk(cudaPeekAtLastError(), 1);
    cudaErrChk(cudaDeviceSynchronize(), 1);
#endif
}

void ReactorNVectorSerialCuda::DividedDifferenceJacobian(double t, N_Vector y, N_Vector fy,
                                                         double* dense_jacobian) {
  long int j;
  const double uround = 1.0e-16;
  int retval = 0;

  double* yd = N_VGetDeviceArrayPointer_Cuda(y);
  double* tmp1d = N_VGetDeviceArrayPointer_Cuda(tmp1_);
  double* tmp2d = N_VGetDeviceArrayPointer_Cuda(tmp2_);
  double* tmp3d = N_VGetDeviceArrayPointer_Cuda(tmp3_);

  const int num_variables = num_variables_;
  const int num_reactors = num_reactors_;

  if(retval == 0) {
    //Loop over num_equations not over N
    for (j = 0; j < num_variables; ++j) {
      /* Generate the jth col of J(tn,y) */

      // Perturb y
#ifndef ZERORK_RNVSC_PERTURB_V2
      // Copy fresh y to the perturbed y (tmp1)
      cudaMemcpy(tmp1d,yd,sizeof(double)*num_variables*num_reactors,cudaMemcpyDeviceToDevice);
      int threads = std::min(1024,num_reactors);
      int blocks  = (num_reactors+threads-1)/threads;
      RNVSC_perturb_y<<<blocks,threads>>>(num_variables, num_reactors, uround, j, tmp1d, tmp3d);
#else
      int threads = std::min(1024,num_reactors*num_variables);
      int blocks  = (num_reactors*num_variables+threads-1)/threads;
      RNVSC_perturb_y<<<blocks,threads>>>(num_variables, num_reactors, uround, j, yd, tmp1d, tmp3d);
#endif
#ifdef ZERORK_FULL_DEBUG
      cudaErrChk(cudaPeekAtLastError(), 1);
      cudaErrChk(cudaDeviceSynchronize(), 1);
#endif
      //tmp1 contains perturbed state

      //Get perturbed derivative
      retval = this->GetTimeDerivative(t, tmp1_, tmp2_);
      if (retval != 0) break;
      //dy1 contains perturbed derivative

      //Take difference of derivatives (store in tmp2_)
      N_VLinearSum(1.0,tmp2_,-1.0,fy,tmp2_);

      // Divide derivative difference by increments (store in tmp3_))
      N_VProd(tmp3_,tmp2_,tmp3_); 
      //tmp3 contains compressed Jacobian data for this column in interleaved/block order

      // Put df/dy in "normal" order (store in tmp2_)
      cuda_transpose(tmp2d,tmp3d,num_reactors,num_variables);

      distribute_jac_terms(num_variables,num_reactors,j,tmp2d,dense_jacobian);
    }
  }

  //return(retval);
}


int ReactorNVectorSerialCuda::GetJacobianDense(double t, N_Vector y, N_Vector fy, SUNMatrix Jac)
{
#ifdef ZERORK_HAVE_MAGMA
  long int N = num_variables_; //N.B. should be unused
  double* Jdata   = SUNMatrix_MagmaDense_Data(Jac);
  return this->GetJacobianDenseRaw(N, t, y, fy, Jdata);
#else
  assert(("Not implemented.",false));
  return 0;
#endif
}

int ReactorNVectorSerialCuda::GetJacobianDenseRaw(long int N, double t, N_Vector y, N_Vector fy, double* Jac)
{
  if(int_options_["analytic"])  {
    SetupJacobianSparseDevice(t,y,fy);
    cudaMemset(Jac, 0, sizeof(double)*num_variables_*num_variables_*num_reactors_);
    SparseToDenseDevice(*jacobian_row_indexes_dev_ptr_, *jacobian_column_indexes_dev_ptr_, jacobian_data_dev_, Jac);
  } else {
    this->DividedDifferenceJacobian(t, y, fy, Jac);
  }
  return 0;
}

int ReactorNVectorSerialCuda::JacobianSetup(realtype t, N_Vector y, N_Vector fy)
{
  int flag = 0;
  if(int_options_["analytic"] == 1)  {
    SetupJacobianSparseDevice(t,y,fy);
    if(int_options_["dense"] == 1) {
      dense_jacobian_dev_.assign(num_variables_*num_variables_*num_reactors_,0);
      SparseToDenseDevice(*jacobian_row_indexes_dev_ptr_, *jacobian_column_indexes_dev_ptr_, jacobian_data_dev_, thrust::raw_pointer_cast(&dense_jacobian_dev_[0]));
    }
  } else {
    dense_jacobian_dev_.resize(num_variables_*num_variables_*num_reactors_);
    this->DividedDifferenceJacobian(t, y, fy, thrust::raw_pointer_cast(&dense_jacobian_dev_[0]));
  }
  jac_setup_ = true;
  return flag;
}

static void __global__ RNSC_AddConstantBlockDiagonalKernel
(
    const int mtx_block_size,
    const int num_mtx_blocks,
    const double constant,
    double* A_dev
)
{
  int tidx   = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = gridDim.x*blockDim.x;
  for( ; tidx < num_mtx_blocks*mtx_block_size; tidx += stride)
  {
      int local_row   = tidx % mtx_block_size;
      int local_block = tidx / mtx_block_size;
      int data_idx = mtx_block_size*mtx_block_size*local_block+local_row*mtx_block_size+local_row;
      A_dev[data_idx] += constant;
  }
}


int ReactorNVectorSerialCuda::AddConstantBlockDiagonal(int n, int nbatch,
                                          double constant, double* A_dev)
{
  int threads=1024;
  int blocks=(n*nbatch + threads - 1)/threads;
  RNSC_AddConstantBlockDiagonalKernel<<<blocks,threads>>>(n,nbatch,constant,A_dev);
#ifdef ZERORK_FULL_DEBUG
  cudaErrChk(cudaPeekAtLastError(), 1);
  cudaErrChk(cudaDeviceSynchronize(), 1);
#endif
  return 0;  
}


int ReactorNVectorSerialCuda::JacobianFactor(realtype gamma)
{
  int flag;
  assert(jac_setup_);
  if(int_options_["dense"] == 1) {
    dense_preconditioner_dev_.resize(num_reactors_*num_variables_*num_variables_);
    //TODO: Fuse all of these into a single kernel
    thrust::fill(dense_preconditioner_dev_.begin(), dense_preconditioner_dev_.end(), -gamma);
    thrust::transform(dense_jacobian_dev_.begin(), dense_jacobian_dev_.end(),
                      dense_preconditioner_dev_.begin(),
                      dense_preconditioner_dev_.begin(),
                      thrust::multiplies<double>());

    // compute M = 1.0 - gamma*numerical_jacobian
    AddConstantBlockDiagonal(num_variables_, num_reactors_, 1.0, thrust::raw_pointer_cast(&dense_preconditioner_dev_[0]));
    flag = cuda_la_manager_->factor(num_reactors_, num_variables_, thrust::raw_pointer_cast(&dense_preconditioner_dev_[0]));
  } else {
    //TODO: Sparse numeric
    if(int_options_["analytic"] != 1) {
      assert(false);
    }
    FormPreconditionerDevice(gamma);
    //try refactor and fall back to full factor on fail
    flag = csrfm_ptr_->refactor(num_reactors_, num_variables_, nnz_,
                           thrust::raw_pointer_cast(&preconditioner_data_dev_[0]));
    if(flag != 0) {
      preconditioner_column_indexes_ = *jacobian_column_indexes_ptr_;
      preconditioner_row_sums_ = *jacobian_row_sums_ptr_;
      preconditioner_data_ = preconditioner_data_dev_; //copies device to host
      flag = csrfm_ptr_->factor(num_reactors_, num_variables_, nnz_,
                           thrust::raw_pointer_cast(&preconditioner_column_indexes_[0]),
                           thrust::raw_pointer_cast(&preconditioner_row_sums_[0]),
                           thrust::raw_pointer_cast(&preconditioner_data_[0]),
                           cusolver_rf_manager::CSR);
    }
  }

  // CVODE will retry if return value is positive
  return flag;
}

//TODO: This could use thrust functor to simplify the code
void __global__ RNSC_FormPreconditionerDevice(
    const int n, const int nnz, const double A[],
    const double B[], double C[], const double gamma)
{
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = gridDim.x*blockDim.x;
  for( ; tidx < nnz; tidx += stride) {
    C[tidx] = A[tidx] - gamma*B[tidx];
  }
}

int ReactorNVectorSerialCuda::FormPreconditionerDevice(double gamma) {
  preconditioner_data_dev_.resize(nnz_*num_reactors_);
  int num_threads = std::min(MAX_THREADS_PER_BLOCK,nnz_*num_reactors_);
  int num_blocks = (nnz_*num_reactors_ + num_threads - 1)/num_threads;
  RNSC_FormPreconditionerDevice<<<num_blocks,num_threads>>>
                        (num_variables_*num_reactors_,
                         nnz_*num_reactors_,
                         thrust::raw_pointer_cast(&(*unit_diagonal_dev_ptr_)[0]),
                         thrust::raw_pointer_cast(&jacobian_data_dev_[0]),
                         thrust::raw_pointer_cast(&preconditioner_data_dev_[0]),
                         gamma);
  return 0;
}

int ReactorNVectorSerialCuda::JacobianSolve(double t, N_Vector y, N_Vector fy,
                            N_Vector r, N_Vector z)
{
  double * r_ptr = N_VGetDeviceArrayPointer_Cuda(r);
  double * z_ptr = N_VGetDeviceArrayPointer_Cuda(z);
  int flag = 0;
  if(int_options_["dense"] == 1) {
    flag = cuda_la_manager_->solve(num_reactors_, num_variables_, r_ptr, z_ptr);
  } else {
    flag = csrfm_ptr_->solve(num_reactors_, num_variables_, r_ptr, z_ptr);
  }
  return flag;
}


static void __global__ RNSC_AddComplexConstantBlockDiagonalKernel
(
    const int mtx_block_size,
    const int num_mtx_blocks,
    const cuDoubleComplex constant,
    cuDoubleComplex* A_dev
)
{
  int tidx   = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = gridDim.x*blockDim.x;
  for( ; tidx < num_mtx_blocks*mtx_block_size; tidx += stride)
  {
      int local_row   = tidx % mtx_block_size;
      int local_block = tidx / mtx_block_size;
      int data_idx = mtx_block_size*mtx_block_size*local_block+local_row*mtx_block_size+local_row;
      A_dev[data_idx] = cuCadd(A_dev[data_idx],constant);
  }
}


int ReactorNVectorSerialCuda::AddComplexConstantBlockDiagonal(int n, int nbatch,
                                          cuDoubleComplex constant, cuDoubleComplex* A_dev)
{
  int threads=1024;
  int blocks=(n*nbatch + threads - 1)/threads;
  RNSC_AddComplexConstantBlockDiagonalKernel<<<blocks,threads>>>(n,nbatch,constant,A_dev);
#ifdef ZERORK_FULL_DEBUG
  cudaErrChk(cudaPeekAtLastError(), 1);
  cudaErrChk(cudaDeviceSynchronize(), 1);
#endif
  return 0;  
}

void __global__ RNSC_InitializeDenseComplexPreconditioner(
    const int n, const double A[], cuDoubleComplex B[])
{
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = gridDim.x*blockDim.x;
  for( ; tidx < n; tidx += stride) {
    B[tidx] = make_cuDoubleComplex(-A[tidx],0.0);
  }
}

int ReactorNVectorSerialCuda::FormDenseComplexPreconditioner(int k, double alpha, double beta) {
  const int num_data_total = num_reactors_*num_variables_*num_variables_;
  dense_preconditioner_dev_z_.resize(num_data_total);
  int num_threads = std::min(MAX_THREADS_PER_BLOCK,num_data_total);
  int num_blocks = (num_data_total + num_threads - 1)/num_threads;
  RNSC_InitializeDenseComplexPreconditioner<<<num_blocks,num_threads>>>
                        (num_data_total,
                         thrust::raw_pointer_cast(&dense_jacobian_dev_[0]),
                         thrust::raw_pointer_cast(&dense_preconditioner_dev_z_[0]));
#ifdef ZERORK_FULL_DEBUG
  cudaErrChk(cudaPeekAtLastError(), 1);
  cudaErrChk(cudaDeviceSynchronize(), 1);
#endif

  //preconditioner is equal to negative jacobian (in cuDoubleComplex)
  const int num_data_diagonal = num_reactors_*num_variables_;
  num_threads = std::min(MAX_THREADS_PER_BLOCK,num_data_diagonal);
  num_blocks = (num_data_diagonal + num_threads - 1)/num_threads;
  cuDoubleComplex constant = make_cuDoubleComplex(alpha, beta);
  RNSC_AddComplexConstantBlockDiagonalKernel<<<num_blocks,num_threads>>>(num_variables_,num_reactors_,constant,
                         thrust::raw_pointer_cast(&dense_preconditioner_dev_z_[0]));
#ifdef ZERORK_FULL_DEBUG
  cudaErrChk(cudaPeekAtLastError(), 1);
  cudaErrChk(cudaDeviceSynchronize(), 1);
#endif
  return 0;
}

int ReactorNVectorSerialCuda::ComplexJacobianFactor(int k, double alpha, double beta)
{
  if(int_options_["dense"] != 1) {
    assert(("Not implemented.",false));
  }
  // compute M = -numerical_jacobian+alpha + beta*i
  FormDenseComplexPreconditioner(k, alpha, beta);
  int flag = cuda_la_manager_z_[k]->factor(num_reactors_, num_variables_, thrust::raw_pointer_cast(&dense_preconditioner_dev_z_[0]));

  return flag;
}

void __global__ RNSC_DoubleArraysToComplex(
    const int n, const double a[], const double b[], cuDoubleComplex c[])
{
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = gridDim.x*blockDim.x;
  for( ; tidx < n; tidx += stride) {
    c[tidx] = make_cuDoubleComplex(a[tidx],b[tidx]);
  }
}

void __global__ RNSC_ComplexArrayToDouble(
    const int n, cuDoubleComplex c[], double a[], double b[])
{
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = gridDim.x*blockDim.x;
  for( ; tidx < n; tidx += stride) {
    a[tidx] = c[tidx].x;
    b[tidx] = c[tidx].y;
  }
}

int ReactorNVectorSerialCuda::ComplexJacobianSolve(int k, N_Vector ax, N_Vector bx)
{
  if(int_options_["dense"] != 1) {
    assert(("Not implemented.",false));
  }
  const int num_data = num_variables_*num_reactors_;
  complex_workspace_dev_.resize(num_data);
  double * a_ptr = N_VGetDeviceArrayPointer_Cuda(ax);
  double * b_ptr = N_VGetDeviceArrayPointer_Cuda(bx);
  cuDoubleComplex* tmp_ptr = thrust::raw_pointer_cast(&complex_workspace_dev_[0]);
  int num_threads = std::min(MAX_THREADS_PER_BLOCK,num_data);
  int num_blocks = (num_data + num_threads - 1)/num_threads;
  RNSC_DoubleArraysToComplex<<<num_blocks,num_threads>>>(num_data,a_ptr, b_ptr, tmp_ptr);
#ifdef ZERORK_FULL_DEBUG
  cudaErrChk(cudaPeekAtLastError(), 1);
  cudaErrChk(cudaDeviceSynchronize(), 1);
#endif

  int flag = cuda_la_manager_z_[k]->solve(num_reactors_, num_variables_, tmp_ptr, tmp_ptr);

  RNSC_ComplexArrayToDouble<<<num_blocks,num_threads>>>(num_data, tmp_ptr, a_ptr, b_ptr);
#ifdef ZERORK_FULL_DEBUG
  cudaErrChk(cudaPeekAtLastError(), 1);
  cudaErrChk(cudaDeviceSynchronize(), 1);
#endif
  return flag;
}

void __global__ RNSC_SparseToDenseDevice(
    const int num_reactors, const int num_vars,
    const int nnz, const int* row_idxs, const int* col_idxs,
    const double* vals, double* dense)
{
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  const int stride = gridDim.x*blockDim.x;
  for( ; tidx < num_reactors*nnz; tidx += stride) {
    const double data = vals[tidx];
    const int k = tidx / nnz; //reactor id
    const int j = tidx % nnz; //sparse id
    const int row = row_idxs[j];
    const int col = col_idxs[j];
    const int offset = k*num_vars*num_vars;
    dense[offset+col*num_vars+row] = data;
  }
}

int ReactorNVectorSerialCuda::SparseToDenseDevice(const zerork::device_vector<int>& row_idxs, const zerork::device_vector<int>& col_idxs,
                                                   const zerork::device_vector<double>& vals, double* dense) {
  int num_threads = std::min(MAX_THREADS_PER_BLOCK,nnz_*num_reactors_);
  int num_blocks = (nnz_*num_reactors_ + num_threads - 1)/num_threads;
  RNSC_SparseToDenseDevice<<<num_blocks, num_threads>>>(num_reactors_, num_variables_, nnz_,
                                                        thrust::raw_pointer_cast(&row_idxs[0]),
                                                        thrust::raw_pointer_cast(&col_idxs[0]),
                                                        thrust::raw_pointer_cast(&vals[0]),
                                                        dense);
  return 0;
}



static void __global__ kernel_setupJac1
(
    const int num_reactors,
    const int num_species,
    const double min_mass_frac,
    const double *mol_wt,
    const double *inv_dens,
    const double *y,
    double *pos_frac,
    double *inv_conc
)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  const int stride = gridDim.x*blockDim.x;
  for( ; tid < num_species*num_reactors; tid += stride) {
    int speciesid = tid / num_reactors;
    int reactorid = tid % num_reactors;
    double pos_val=  y[tid];
    if(fabs(pos_val) < min_mass_frac) {
      if(pos_val < 0) {
        pos_val= -min_mass_frac;
      } else {
        pos_val= min_mass_frac;
      }
    }
    pos_frac[tid] = pos_val;
    inv_conc[tid] = mol_wt[speciesid]*inv_dens[reactorid]/pos_val;
  }
}


void __global__ kernel_fwd_destroy_fused
(
  const int num_reactors,
  const int num_terms,
  const int nnz,
  const int* sparse_indexes,
  const int* conc_indexes,
  const int* rxn_indexes,
  const double *inv_conc,
  const double *rop,
  double *jacobian_data
)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  const int stride = gridDim.x*blockDim.x;
  for( ; tid < num_reactors*num_terms; tid += stride) {
     const int termid =    tid % num_terms;
     const int reactorid = tid / num_terms;

     int conc_idx_loc   = conc_indexes[termid]*num_reactors + reactorid;
     int rxn_idx_loc  = rxn_indexes[termid]*num_reactors + reactorid;
     int sparse_idx_loc  = reactorid*nnz + sparse_indexes[termid];
     double val = -1*rop[rxn_idx_loc]*inv_conc[conc_idx_loc];
     atomicAdd(&jacobian_data[sparse_idx_loc], val);
  }
}

void __global__ kernel_fwd_create_fused
(
  const int num_reactors,
  const int num_terms,
  const int nnz,
  const int* sparse_indexes,
  const int* conc_indexes,
  const int* rxn_indexes,
  const double *inv_conc,
  const double *rop,
  double *jacobian_data
)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  const int stride = gridDim.x*blockDim.x;
  for( ; tid < num_reactors*num_terms; tid += stride) {
     const int termid =    tid % num_terms;
     const int reactorid = tid / num_terms;

     int conc_idx_loc   = conc_indexes[termid]*num_reactors + reactorid;
     int rxn_idx_loc  = rxn_indexes[termid]*num_reactors + reactorid;
     int sparse_idx_loc  = reactorid*nnz + sparse_indexes[termid];
     double val = 1*rop[rxn_idx_loc]*inv_conc[conc_idx_loc];
     atomicAdd(&jacobian_data[sparse_idx_loc], val);
  }
}

void __global__ kernel_noninteger_terms
(
  const int num_reactors,
  const int num_terms,
  const int nnz,
  const int* sparse_indexes,
  const int* conc_indexes,
  const int* rxn_indexes,
  const double* multipliers,
  const double *inv_conc,
  const double *rop,
  double *jacobian_data
)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  const int stride = gridDim.x*blockDim.x;
  for( ; tid < num_reactors*num_terms; tid += stride) {
     const int termid =    tid % num_terms;
     const int reactorid = tid / num_terms;

     const int conc_idx_loc    = conc_indexes[termid]*num_reactors + reactorid;
     const int rxn_idx_loc     = rxn_indexes[termid]*num_reactors + reactorid;
     const int sparse_idx_loc  = reactorid*nnz + sparse_indexes[termid];
     double val = multipliers[termid]*rop[rxn_idx_loc]*inv_conc[conc_idx_loc];
     atomicAdd(&jacobian_data[sparse_idx_loc], val);
  }
}

void __global__ kernel_convert_concentration_to_mf
(
  const int num_reactors,
  const int num_species,
  const int nnz,
  const int *jacobian_row_indexes,
  const int *jacobian_column_indexes,
  const double * inv_mol_wt,
  double * jacobian_data
)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  const int stride = gridDim.x*blockDim.x;

  for( ; tid < num_reactors*nnz; tid += stride) {
    const int sparseid = tid%nnz;
    const int row = jacobian_row_indexes[sparseid];
    const int col = jacobian_column_indexes[sparseid];
    if(row < num_species && col < num_species) {
      jacobian_data[tid] *= inv_mol_wt[col] / inv_mol_wt[row];
    }
  }
}

void __global__ kernel_jac_temperatureTermsA
(
  const int num_reactors,
  const int num_species,
  const int nnz,
  const int jacobian_temp_idx,
  const double Ru,
  const double sqrt_unit_round,
  const int *jacobian_row_indexes,
  const int *jacobian_column_indexes,
  const double * y_ptr,
  const double * tmp3_ptr,
  const double * energy,
  const double * inv_mol_wt,
  const double * cx_mass,
  const double * mean_cx_mass,
  double * jacobian_data,
  double * tmp1_ptr
)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  const int stride = gridDim.x*blockDim.x;

  for( ; tid < num_reactors*nnz; tid += stride) {
    const int sparseid = tid%nnz;
    const int row = jacobian_row_indexes[sparseid];
    const int col = jacobian_column_indexes[sparseid];
    if(row < num_species && col < num_species) {
      const int reactorid = tid/nnz;
      const int energyid = num_reactors*row + reactorid;
      const int temperature_sparse_idx = jacobian_temp_idx + col + reactorid * nnz;
      atomicAdd(&jacobian_data[temperature_sparse_idx],jacobian_data[tid]*energy[energyid]);
    }
  }
}


void __global__ kernel_jac_temperatureTermsB
(
  const int num_reactors,
  const int num_species,
  const int nnz,
  const double Ru,
  const double sqrt_unit_round,
  const int *jacobian_row_sums,
  const int *jacobian_column_indexes,
  const double * y_ptr,
  const double * tmp3_ptr,
  const double * energy,
  const double * inv_mol_wt,
  const double * cx_mass,
  const double * mean_cx_mass,
  double * jacobian_data,
  double * tmp1_ptr
)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  const int stride = gridDim.x*blockDim.x;

  for( ; tid < num_reactors*num_species; tid += stride) {
    const int speciesid = tid / num_reactors;
    const int reactorid = tid % num_reactors;

    int temperature_sparse_idx = jacobian_row_sums[num_species] + speciesid + reactorid * nnz;
    double RuTemp = Ru*y_ptr[num_species*num_reactors+reactorid];
    jacobian_data[temperature_sparse_idx] *= RuTemp;

    // step 2: add the Tdot*Cv[j] term to d(Tdot)/dy[j]
    jacobian_data[temperature_sparse_idx]+=tmp3_ptr[num_species*num_reactors+reactorid]*cx_mass[tid]/inv_mol_wt[speciesid];

    // step 3: divide by -1/(molWt[j]*meanCvMass)
    jacobian_data[temperature_sparse_idx] *= -inv_mol_wt[speciesid]/mean_cx_mass[reactorid];

    // At this point Mtx stores d(Tdot[k])/dy[j] ignoring the contribution
    // of perturbations in the third body species

    // calculate the derivatives with the temperature perturbed
    // step 1: perturb the temperature
    if(speciesid == 0) {
      double delta_temp=y_ptr[num_species*num_reactors+reactorid]*sqrt_unit_round;
      tmp1_ptr[num_species*num_reactors+reactorid] = y_ptr[num_species*num_reactors+reactorid] + delta_temp;
    }

    tmp1_ptr[tid]=y_ptr[tid];
  }
}


void __global__ kernel_jac_temperatureTermsC
(
  const int num_reactors,
  const int num_variables,
  const int nnz,
  const int *jacobian_row_sums,
  const double * y_ptr,
  const double * ydot_ptr,
  const double * tmp1_ptr,
  const double * tmp3_ptr,
  double * jacobian_data
)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  const int stride = gridDim.x*blockDim.x;
  for( ; tid < num_reactors*num_variables; tid += stride) {
    const int varid = tid / num_reactors;
    const int reactorid = tid % num_reactors;
    // ---------------------------------------------------------------------
    // calculate d(ydot[k])/dT
    double multiplier=1.0/(tmp1_ptr[(num_variables-1)*num_reactors+reactorid]-y_ptr[(num_variables-1)*num_reactors+reactorid]);

    // step 3: approximate d(ydot[k])/dT with finite difference
    int temperature_sparse_idx = jacobian_row_sums[varid+1] - 1 + reactorid*nnz;
    jacobian_data[temperature_sparse_idx]=(tmp3_ptr[tid]-ydot_ptr[tid])*multiplier;
  }
}


//N.B. we work with CSR (not CSC) to be compatible with cusolverRf
int ReactorNVectorSerialCuda::SetupJacobianSparseDevice(realtype t, N_Vector y, N_Vector fy) {
  const double min_mass_frac = double_options_["min_mass_fraction"];

  double *y_ptr_dev = N_VGetDeviceArrayPointer_Cuda(y);
  double *ydot_ptr_dev = N_VGetDeviceArrayPointer_Cuda(fy);
  double *tmp1_ptr_dev = N_VGetDeviceArrayPointer_Cuda(tmp1_);
  double *tmp2_ptr_dev = N_VGetDeviceArrayPointer_Cuda(tmp2_);
  double *tmp3_ptr_dev = N_VGetDeviceArrayPointer_Cuda(tmp3_);

  // set tmp1 to the strictly positive mass fraction array
  // set tmp2 to 1/C where C is the concentration for the strictly positive
  // mass fraction array
  int nthreads = std::min(num_reactors_*num_species_, 512);
  int nblocks = (num_reactors_*num_species_ + nthreads - 1)/nthreads;
  kernel_setupJac1<<<nblocks,nthreads>>>
                   (num_reactors_, num_species_, min_mass_frac, thrust::raw_pointer_cast(&mol_wt_dev_[0]),
                    thrust::raw_pointer_cast(&inverse_densities_dev_[0]), y_ptr_dev, tmp1_ptr_dev, tmp2_ptr_dev);

  if(solve_temperature_) {
    //Copy reactor temperatures to tmp1 //TODO: Async
    cudaMemcpy(&(tmp1_ptr_dev[num_species_*num_reactors_]),&(y_ptr_dev[num_species_*num_reactors_]),
               sizeof(double)*num_reactors_,cudaMemcpyDeviceToDevice);
  }

  GetTimeDerivative(t,tmp1_,tmp3_);

  // set the full sparse array to zero //TODO: Async?
  jacobian_data_dev_.assign(nnz_*num_reactors_,0);

  // process the forward destruction terms
  int n_destruction_terms = destruction_terms_sparse_indexes_dev_ptr_->size(); 
  if(n_destruction_terms > 0) {
    nthreads = std::min(num_reactors_*n_destruction_terms, 512);
    nblocks = (num_reactors_*n_destruction_terms + nthreads - 1)/nthreads;
    kernel_fwd_destroy_fused<<<nblocks, nthreads>>>(num_reactors_, n_destruction_terms, nnz_, 
                                              thrust::raw_pointer_cast(&(*destruction_terms_sparse_indexes_dev_ptr_)[0]),
                                              thrust::raw_pointer_cast(&destruction_terms_conc_indexes_dev_[0]),
                                              thrust::raw_pointer_cast(&destruction_terms_reac_indexes_dev_[0]),
                                              tmp2_ptr_dev, thrust::raw_pointer_cast(&forward_rates_of_production_dev_[0]),
                                              thrust::raw_pointer_cast(&jacobian_data_dev_[0]));
  }

  // process the forward creation terms
  int n_creation_terms = creation_terms_sparse_indexes_dev_ptr_->size(); 
  if(n_creation_terms > 0) {
    nthreads = std::min(num_reactors_*n_creation_terms, 512);
    nblocks = (num_reactors_*n_creation_terms + nthreads - 1)/nthreads;
    kernel_fwd_create_fused<<<nblocks, nthreads>>>(num_reactors_, n_creation_terms, nnz_, 
                                              thrust::raw_pointer_cast(&(*creation_terms_sparse_indexes_dev_ptr_)[0]),
                                              thrust::raw_pointer_cast(&creation_terms_conc_indexes_dev_[0]),
                                              thrust::raw_pointer_cast(&creation_terms_reac_indexes_dev_[0]),
                                              tmp2_ptr_dev, thrust::raw_pointer_cast(&forward_rates_of_production_dev_[0]),
                                              thrust::raw_pointer_cast(&jacobian_data_dev_[0]));
  }

  // process the non-integer Jacobian information
  if(num_noninteger_jacobian_nonzeros_ > 0) {
#ifdef ZERORK_REACTOR_NON_INTEGER_JACOBIAN_CPU
    //copy inverse concentrations and rates of progress to host
    forward_rates_of_production_ = forward_rates_of_production_dev_;
    N_VCopyFromDevice_Cuda(tmp2_);
    double* tmp2_ptr = N_VGetHostArrayPointer_Cuda(tmp2_);

    //These hold transposed data for call to host non-integer routine
    std::vector<double> fwd_rop_trans(num_steps_);
    std::vector<double> inv_concentrations_trans(num_species_);

    jacobian_data_nonint_.assign(nnz_*num_reactors_, 0.0);   
    const int num_noninteger_jacobian_nonzeros =
      num_noninteger_jacobian_nonzeros_;

    for(int k=0; k<num_reactors_; ++k) { 
      for(int j=0; j<num_noninteger_jacobian_nonzeros; ++j) {
        noninteger_jacobian_[j] = 0.0;
      }
      for(int j = 0; j < num_steps_; ++j) {
        fwd_rop_trans[j] = forward_rates_of_production_[j*num_reactors_+k];
      }
      for(int j = 0; j < num_species_; ++j) {
        inv_concentrations_trans[j] = tmp2_ptr[j*num_reactors_+k];
      }

      mech_ptr_->getNonIntegerReactionNetwork()->GetSpeciesJacobian(
        &inv_concentrations_trans[0],
        &fwd_rop_trans[0],
        &noninteger_jacobian_[0]);

      for(int j=0; j<num_noninteger_jacobian_nonzeros; ++j) {
        jacobian_data_nonint_[k*nnz_+ (*noninteger_sparse_id_ptr_)[j]] += noninteger_jacobian_[j];
      }
    }
    //copy host non_integer terms to device
    jacobian_data_nonint_dev_ = jacobian_data_nonint_;
    //add non_integer to full jacobian
    thrust::transform(jacobian_data_dev_.begin(), jacobian_data_dev_.end(),
                      jacobian_data_nonint_dev_.begin(),
                      jacobian_data_dev_.begin(),
                      thrust::plus<double>());
    
#else
    nthreads = std::min(num_reactors_*num_noninteger_jacobian_terms_, 512);
    nblocks = (num_reactors_*num_noninteger_jacobian_terms_ + nthreads - 1)/nthreads;
    kernel_noninteger_terms<<<nblocks, nthreads>>>(num_reactors_, num_noninteger_jacobian_terms_, nnz_,
                                                   thrust::raw_pointer_cast(&(*noninteger_term_id_dev_ptr_)[0]),
                                                   thrust::raw_pointer_cast(&noninteger_concentration_id_dev_[0]),
                                                   thrust::raw_pointer_cast(&noninteger_step_id_dev_[0]),
                                                   thrust::raw_pointer_cast(&noninteger_multiplier_dev_[0]),
                                                   tmp2_ptr_dev, thrust::raw_pointer_cast(&forward_rates_of_production_dev_[0]),
                                                   thrust::raw_pointer_cast(&jacobian_data_dev_[0]));
#endif
  }

  if(solve_temperature_) {
    // At this point sMptr stores d(wdot[k])/dC[j] ignoring the contribution
    // of perturbations in the third body species
    // ---------------------------------------------------------------------
    // compute d(Tdot)/dy[j]
    nthreads = std::min(num_reactors_*nnz_, 512);
    nblocks = (num_reactors_*nnz_ + nthreads - 1)/nthreads;
    kernel_jac_temperatureTermsA<<<nblocks, nthreads>>>(num_reactors_, num_species_, nnz_, (*jacobian_row_sums_ptr_)[num_species_], mech_ptr_->getGasConstant(),
                                sqrt_unit_round_, thrust::raw_pointer_cast(&(*jacobian_row_indexes_dev_ptr_)[0]), 
                                thrust::raw_pointer_cast(&(*jacobian_column_indexes_dev_ptr_)[0]),
                                y_ptr_dev, tmp3_ptr_dev, thrust::raw_pointer_cast(&energy_dev_[0]),
                                thrust::raw_pointer_cast(&inv_mol_wt_dev_[0]), thrust::raw_pointer_cast(&cx_mass_dev_[0]),
                                thrust::raw_pointer_cast(&mean_cx_mass_dev_[0]), thrust::raw_pointer_cast(&jacobian_data_dev_[0]),
                                tmp1_ptr_dev);

    nthreads = std::min(num_reactors_*num_species_, 512);
    nblocks = (num_reactors_*num_species_ + nthreads - 1)/nthreads;
    kernel_jac_temperatureTermsB<<<nblocks, nthreads>>>(num_reactors_, num_species_, nnz_, mech_ptr_->getGasConstant(),
                                sqrt_unit_round_, thrust::raw_pointer_cast(&(*jacobian_row_sums_dev_ptr_)[0]), 
                                thrust::raw_pointer_cast(&(*jacobian_column_indexes_dev_ptr_)[0]),
                                y_ptr_dev, tmp3_ptr_dev, thrust::raw_pointer_cast(&energy_dev_[0]),
                                thrust::raw_pointer_cast(&inv_mol_wt_dev_[0]), thrust::raw_pointer_cast(&cx_mass_dev_[0]),
                                thrust::raw_pointer_cast(&mean_cx_mass_dev_[0]), thrust::raw_pointer_cast(&jacobian_data_dev_[0]),
                                tmp1_ptr_dev);

    // step 2: calculate ydot at Temp+dTemp (result in tmp3)
    GetTimeDerivative(t,tmp1_,tmp3_);

    kernel_jac_temperatureTermsC<<<nblocks, nthreads>>>(num_reactors_, num_variables_, nnz_, 
                                                        thrust::raw_pointer_cast(&(*jacobian_row_sums_dev_ptr_)[0]),
                                                        y_ptr_dev,
                                                        ydot_ptr_dev,
                                                        tmp1_ptr_dev,
                                                        tmp3_ptr_dev,
                                                        thrust::raw_pointer_cast(&jacobian_data_dev_[0]));
  }

  nthreads = std::min(num_reactors_*nnz_, 512);
  nblocks = (num_reactors_*nnz_ + nthreads - 1)/nthreads;
  kernel_convert_concentration_to_mf<<<nblocks, nthreads>>>(num_reactors_,
      num_species_, nnz_, thrust::raw_pointer_cast(&(*jacobian_row_indexes_dev_ptr_)[0]),
      thrust::raw_pointer_cast(&(*jacobian_column_indexes_dev_ptr_)[0]),
      thrust::raw_pointer_cast(&inv_mol_wt_dev_[0]),
      thrust::raw_pointer_cast(&jacobian_data_dev_[0]));
  return 0;
}

void ReactorNVectorSerialCuda::SetStepLimiter(double value) {
  thrust::fill(step_limiter_.begin(), step_limiter_.end(), value);
}

void ReactorNVectorSerialCuda::Reset() {
  csrfm_temperature_.reset();
  csrfm_no_temperature_.reset();
}


int ReactorNVectorSerialCuda::SetRootTime(double t)
{
  root_time_ = t;
  return 0;
}

double ReactorNVectorSerialCuda::GetRootTime()
{
  return root_time_;
}

