
#include <cmath> //sqrt
#include <algorithm> //max
#include <cassert>

#include "utility_funcs.h"
#include "reactor_nvector_serial.h"

#include "nvector/nvector_serial.h"

ReactorNVectorSerial::ReactorNVectorSerial(std::shared_ptr<zerork::mechanism> mech_ptr) : 
  ReactorBase(),
  mech_ptr_(mech_ptr),
  solve_temperature_(true),
  jacobian_row_indexes_ptr_(&jacobian_row_indexes_temperature_),
  jacobian_column_sums_ptr_(&jacobian_column_sums_temperature_),  
  noninteger_sparse_id_ptr_(&noninteger_sparse_id_temperature_),
  destruction_sparse_indexes_ptr_(&destruction_terms_.sparse_indexes_temperature),
  creation_sparse_indexes_ptr_(&creation_terms_.sparse_indexes_temperature)
{
  num_species_ = mech_ptr_->getNumSpecies();
  num_variables_ = num_species_ + 1;
  num_steps_ = mech_ptr_->getNumSteps();

  sqrt_unit_round_ = sqrt(UNIT_ROUNDOFF);
  root_time_ = 0.0;

  state_data_.assign(num_variables_,0);
  tmp1_data_.assign(num_variables_,0);
  tmp2_data_.assign(num_variables_,0);
  tmp3_data_.assign(num_variables_,0);

  state_ = N_VMake_Serial(num_variables_, &state_data_[0]);
  tmp1_ = N_VMake_Serial(num_variables_, &tmp1_data_[0]);
  tmp2_ = N_VMake_Serial(num_variables_, &tmp2_data_[0]);
  tmp3_ = N_VMake_Serial(num_variables_, &tmp3_data_[0]);

  mol_wt_.assign(num_species_,0.0);
  inv_mol_wt_.assign(num_species_,0.0);
  net_production_rates_.assign(num_species_,0.0);
  energy_.assign(num_species_,0.0);
  cx_mass_.assign(num_species_,0.0);
  forward_rates_of_production_.assign(num_steps_,0.0);
  creation_rates_.assign(num_species_,0.0);
  destruction_rates_.assign(num_species_,0.0);
  concentrations_.assign(num_species_,0.0);

  // set constant parameters
  mech_ptr_->getMolWtSpc(&mol_wt_[0]);
  for(int j=0; j < num_species_; ++j) {
    inv_mol_wt_[j]=1.0/mol_wt_[j];
  }

  SetupSparseJacobianArrays();

  weights_.assign(1,1.0);
  step_limiter_.assign(num_steps_, 1.0e22);
}

ReactorNVectorSerial::~ReactorNVectorSerial()
{
  N_VDestroy(state_);
  N_VDestroy(tmp1_);
  N_VDestroy(tmp2_);
  N_VDestroy(tmp3_);
}

N_Vector& ReactorNVectorSerial::GetStateNVectorRef() {
  return state_;
}

void ReactorNVectorSerial::SetSolveTemperature(bool value) {
  if(solve_temperature_ != value) {
    solve_temperature_ = value;
    if(solve_temperature_) {
      num_variables_ = num_species_ + 1;
      nnz_ = nnz_temperature_;
      jacobian_row_indexes_ptr_ = &jacobian_row_indexes_temperature_;  
      jacobian_column_sums_ptr_ = &jacobian_column_sums_temperature_;  
      noninteger_sparse_id_ptr_ = &noninteger_sparse_id_temperature_;
      destruction_sparse_indexes_ptr_ = &(destruction_terms_.sparse_indexes_temperature);
      creation_sparse_indexes_ptr_ = &(creation_terms_.sparse_indexes_temperature);
    } else {
      num_variables_ = num_species_;
      nnz_ = nnz_no_temperature_;
      jacobian_row_indexes_ptr_ = &jacobian_row_indexes_no_temperature_;  
      jacobian_column_sums_ptr_ = &jacobian_column_sums_no_temperature_;  
      noninteger_sparse_id_ptr_ = &noninteger_sparse_id_no_temperature_;
      destruction_sparse_indexes_ptr_ = &(destruction_terms_.sparse_indexes_no_temperature);
      creation_sparse_indexes_ptr_ = &(creation_terms_.sparse_indexes_no_temperature);
    }
    N_VDestroy(state_);
    N_VDestroy(tmp1_);
    N_VDestroy(tmp2_);
    N_VDestroy(tmp3_);
    state_ = N_VMake_Serial(num_variables_, &state_data_[0]);
    tmp1_ = N_VMake_Serial(num_variables_, &tmp1_data_[0]);
    tmp2_ = N_VMake_Serial(num_variables_, &tmp2_data_[0]);
    tmp3_ = N_VMake_Serial(num_variables_, &tmp3_data_[0]);
  }
}

void ReactorNVectorSerial::SetBatchMaskNVector(int reactor_idx, N_Vector batch_mask) {
  assert(reactor_idx == 0);
  N_VConst(1.0, batch_mask);
}

void ReactorNVectorSerial::GetAbsoluteToleranceCorrection(N_Vector correction) {
  if(int_options_["abstol_dens"]) {
    double reactor_density = 1.0/inverse_density_;
    for(int j=0; j < num_species_; ++j) {
      double molar_density = reactor_density*inv_mol_wt_[j]*1.0e-3; //mks->cgs
      NV_Ith_S(correction,j) = 1.0/molar_density;
    }
    if(solve_temperature_) {
      NV_Ith_S(correction,num_species_) = 1.0;
    }
  } else {
    N_VConst(1.0,correction);
  }
}

void ReactorNVectorSerial::SetupSparseJacobianArrays() {
  assert(num_variables_ == num_species_+1);
  const int dense_matrix_size = num_variables_*num_variables_;
  std::vector<int> isNonZero(dense_matrix_size, 0);

  // initialize the dense nonzero flags
  for(int j=0; j<num_species_; ++j) {
      isNonZero[j*num_variables_+j]=1;    // mark the diagonal
      isNonZero[j*num_variables_+num_species_]=1; // mark the last row (temperature)
  }
  for(int k=0; k<num_variables_; ++k) { // mark the last column (temperature)
    isNonZero[num_species_*num_variables_+k]=1;
  }

  destruction_terms_.concentration_indexes.clear();
  destruction_terms_.reaction_indexes.clear();
  destruction_terms_.sparse_indexes_temperature.clear();
  destruction_terms_.sparse_indexes_no_temperature.clear();
  creation_terms_.concentration_indexes.clear();
  creation_terms_.reaction_indexes.clear();
  creation_terms_.sparse_indexes_temperature.clear();
  creation_terms_.sparse_indexes_no_temperature.clear();
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
        isNonZero[column_idx*num_variables_+row_idx]=1; // mark location in dense

        destruction_terms_.concentration_indexes.push_back(column_idx);
        destruction_terms_.reaction_indexes.push_back(j);
        destruction_terms_.sparse_indexes_temperature.push_back(row_idx);
        destruction_terms_.sparse_indexes_no_temperature.push_back(row_idx);
      }
      // forward creation
      for(int m=0; m < num_products; ++m) {
        int row_idx=mech_ptr_->getSpecIdxOfStepProduct(j,m); // species being created
        isNonZero[column_idx*num_variables_+row_idx]=1; // mark location in dense

        creation_terms_.concentration_indexes.push_back(column_idx);
        creation_terms_.reaction_indexes.push_back(j);
        creation_terms_.sparse_indexes_temperature.push_back(row_idx);
        creation_terms_.sparse_indexes_no_temperature.push_back(row_idx);
      }
    }
  }

  num_noninteger_jacobian_nonzeros_ =
    mech_ptr_->getNonIntegerReactionNetwork()->GetNumJacobianNonzeros();

  std::vector<int> noninteger_row_id;
  std::vector<int> noninteger_column_id;

  // non-integer reaction network
  if(num_noninteger_jacobian_nonzeros_ > 0) {
    noninteger_jacobian_.assign(num_noninteger_jacobian_nonzeros_,0.0);
    noninteger_sparse_id_temperature_.assign(num_noninteger_jacobian_nonzeros_,0);
    noninteger_sparse_id_no_temperature_.assign(num_noninteger_jacobian_nonzeros_,0);

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
  jacobian_column_sums_temperature_.assign(num_variables_+1,0);
  jacobian_column_sums_no_temperature_.assign(num_species_+1,0);
  for(int j=0; j<num_variables_; j++) {
    for(int k=0; k<num_variables_; k++) {
      if(isNonZero[j*num_variables_+k]==1) {
        isNonZero[j*num_variables_+k]=nnz_temperature_;
        nnz_temperature_ += 1;
      }
    }
    // after counting column j store the running total in column j+1
    // of the column sum
    jacobian_column_sums_temperature_[j+1]=nnz_temperature_-1;
    if(j < num_species_) {
      jacobian_column_sums_no_temperature_[j+1]=nnz_temperature_-1-(j+1);
    }
  }
  // now at each nonzero term, isNonZero is storing the (address+1) in the
  // actual compressed column storage data array

  nnz_temperature_--; // decrement the count
  nnz_no_temperature_ = nnz_temperature_ - (num_species_+num_variables_);
  nnz_ = nnz_temperature_;

  // allocate matrix data
  jacobian_row_indexes_temperature_.assign(nnz_temperature_,0);
  jacobian_row_indexes_no_temperature_.assign(nnz_no_temperature_,0);
  last_row_indexes_.assign(num_variables_,0);

  // scan the the isNonZero array to determine the row indexes
  // and the special data addresses
  for(int j=0; j<num_variables_; ++j) {
    for(int k=0; k<num_variables_; ++k) {
      int nzAddr=isNonZero[j*num_variables_+k];
      if(nzAddr>0) {
        jacobian_row_indexes_temperature_[nzAddr-1]=k;
        if(j < num_species_ && k < num_species_) {
          int nzAddrNoTemp = nzAddr - j; //minus 1 for each temperature elem
          jacobian_row_indexes_no_temperature_[nzAddrNoTemp-1]=k;
        }
      }
    }
    last_row_indexes_[j] = isNonZero[(j+1)*num_variables_-1]-1;
  }

  // use the isNonZero array as a lookup to store the proper compressed
  // column data storage
  for(int j=0; j<destruction_terms_.concentration_indexes.size(); ++j) {
    int row_idx=destruction_terms_.sparse_indexes_temperature[j];
    int column_idx=destruction_terms_.concentration_indexes[j];
    int nzAddr=isNonZero[column_idx*num_variables_+row_idx];
    int sparse_idx = nzAddr-1;
    destruction_terms_.sparse_indexes_temperature[j]=sparse_idx; // reset to sparse addr
    destruction_terms_.sparse_indexes_no_temperature[j]=sparse_idx-column_idx; // reset to sparse addr
  }

  for(int j=0; j<creation_terms_.concentration_indexes.size(); ++j) {
    int row_idx=creation_terms_.sparse_indexes_temperature[j];
    int column_idx=creation_terms_.concentration_indexes[j];
    int nzAddr=isNonZero[column_idx*num_variables_+row_idx];
    int sparse_idx = nzAddr-1;
    creation_terms_.sparse_indexes_temperature[j]=sparse_idx; // reset to sparse addr
    creation_terms_.sparse_indexes_no_temperature[j]=sparse_idx-column_idx; // reset to sparse addr
  }

  for(int j=0; j<num_noninteger_jacobian_nonzeros_; ++j) {
    int dense_id = noninteger_row_id[j]+noninteger_column_id[j]*num_variables_;
    int nzAddr=isNonZero[dense_id];
    noninteger_sparse_id_temperature_[j] = nzAddr-1;
    noninteger_sparse_id_no_temperature_[j] = nzAddr-1 - noninteger_column_id[j];
  }


  // complex J for radau
  lpmz_.resize(ncmplx_jacs_);
  slumz_.resize(ncmplx_jacs_);
  preconditioner_column_sums_z_.resize(ncmplx_jacs_);
  preconditioner_row_indexes_z_.resize(ncmplx_jacs_);
  preconditioner_data_z_.resize(ncmplx_jacs_);

}


#ifdef SUNDIALS2
int ReactorNVectorSerial::GetJacobianDense(long int N, double t, N_Vector y, N_Vector fy,
                                               DlsMat Jac)
{
  SetupJacobianSparse(t,y,fy);
  int flag = SparseToDense(Jac);
  return flag;
}
#elif defined SUNDIALS3 || defined SUNDIALS4
int ReactorNVectorSerial::GetJacobianDense(double t, N_Vector y, N_Vector fy,
                                               SUNMatrix Jac)
{
  SetupJacobianSparse(t,y,fy);
  int flag = SparseToDense(Jac);
  return flag;
}
#else
#error "Unsupported SUNDIALS version"
#endif


int ReactorNVectorSerial::GetJacobianDenseRaw(long int N, double t, N_Vector y, N_Vector fy,
                                                  double* Jac)
{
  ASSERT(false,"Not implemented.");
  //Used for GPU
  return 0;
}

#ifdef SUNDIALS2
int ReactorNVectorSerial::SparseToDense(DlsMat Jac) {
  const int num_vars = GetNumStateVariables();
  for(int j=0; j < num_vars; ++j) {
    realtype* col = DENSE_COL(Jac,j);
    for(int k = (*jacobian_column_sums_ptr_)[j]; k < (*jacobian_column_sums_ptr_)[j+1]; ++k) {
      col[(*jacobian_row_indexes_ptr_)[k]] = jacobian_data_[k];
    }
  }
  return 0;
}
#elif defined SUNDIALS3 || defined SUNDIALS4
int ReactorNVectorSerial::SparseToDense(SUNMatrix Jac) {
  const int num_vars = GetNumStateVariables();
  for(int j=0; j < num_vars; ++j) {
    realtype* col = SUNDenseMatrix_Column(Jac,j);
    for(int k = (*jacobian_column_sums_ptr_)[j]; k < (*jacobian_column_sums_ptr_)[j+1]; ++k) {
      col[(*jacobian_row_indexes_ptr_)[k]] = jacobian_data_[k];
    }
  }
  return 0;
}
#else
#error "Unsupported SUNDIALS version"
#endif


int ReactorNVectorSerial::SparseToDense(const std::vector<int>& sums, const std::vector<int>& idxs,
                                        const std::vector<double>& vals, std::vector<double>* dense) {
  const int num_vars = GetNumStateVariables();
  dense->assign(num_vars*num_vars, 0.0);
  for(int j=0; j < num_vars; ++j) {
    for(int k = sums[j]; k < sums[j+1]; ++k) {
      (*dense)[j*num_vars + idxs[k]] = vals[k];
    }
  }
  return 0;
}

int ReactorNVectorSerial::SparseToDenseComplex(const std::vector<int>& sums, const std::vector<int>& idxs,
                                               const std::vector<doublecomplex>& vals, std::vector<std::complex<double>>* dense) {
  const int num_vars = GetNumStateVariables();
  for(int j=0; j < num_vars; ++j) {
    for(int k = sums[j]; k < sums[j+1]; ++k) {
      (*dense)[j*num_vars + idxs[k]] = std::complex<double>(vals[k].r, vals[k].i);
    }
  }
  return 0;
}

int ReactorNVectorSerial::JacobianSetup(double t, N_Vector y, N_Vector fy)
{
 if(int_options_["analytic"] == 1) {
   SetupJacobianSparse(t,y,fy);
 } else {
   this->DividedDifferenceJacobian(t, y, fy, &dense_jacobian_);
 }
 return 0;
}

int ReactorNVectorSerial::JacobianFactor(double gamma)
{
  bool dense_numeric = (int_options_["dense"] == 1) && (int_options_["analytic"] == 0);
  int mismatch = 0;
  if( !dense_numeric ) {
    mismatch = FormPreconditioner(gamma);
  }

  int flag = 0;
  if(int_options_["dense"] == 1) {
    if(int_options_["analytic"] == 1) {
      SparseToDense(preconditioner_column_sums_, preconditioner_row_indexes_, preconditioner_data_,
                    &dense_preconditioner_);
    } else {
      dense_preconditioner_.assign(num_variables_*num_variables_,0.0);
      for(int col = 0; col < num_variables_; ++col) {
        for(int row = 0; row < num_variables_; ++row) {
          const int idx = row*num_variables_ + col;
          dense_preconditioner_[idx] = -gamma*dense_jacobian_[idx];
        }
        dense_preconditioner_[col*num_variables_ + col] += 1.0;
      }
    }
    flag = lpm_.factor(num_variables_, num_variables_, dense_preconditioner_);
    dense_preconditioner_.clear();
  } else {
    //try refactor and fall back to full factor on fail
    //flag = 1;
    //if(mismatch == 0) {
      flag = slum_.refactor(preconditioner_data_);
    //}
    if(flag != 0) {
      flag = slum_.factor(preconditioner_row_indexes_,
                          preconditioner_column_sums_,
                          preconditioner_data_,
                          superlu_manager::CSC);
    }
  }
  return flag;
}

int ReactorNVectorSerial::FormPreconditioner(double gamma)
{
  int preconditioner_nnz = 0;
  double preconditioner_threshold = double_options_["preconditioner_threshold"];
  if(int_options_["iterative"] == 0) {
    preconditioner_threshold = 0.0;
  }

  std::vector<int> prev_preconditioner_column_sums = preconditioner_column_sums_;
  std::vector<int> prev_preconditioner_row_indexes = preconditioner_row_indexes_;

  preconditioner_column_sums_.assign(num_variables_+1,0);
  preconditioner_data_.assign(nnz_,0.0);
  preconditioner_row_indexes_.assign(nnz_,0);
  preconditioner_column_sums_[0] = 0;
  const int num_vars = num_variables_;

  if(int_options_["analytic"] == 0) {
    for(int j=0; j<num_vars; ++j) {
      preconditioner_column_sums_[j+1] = preconditioner_column_sums_[j];
      for(int k=(*jacobian_column_sums_ptr_)[j]; k<(*jacobian_column_sums_ptr_)[j+1]; ++k) {
        int row = (*jacobian_row_indexes_ptr_)[k];
        bool diag = row == j;
        double element_value = -gamma*dense_jacobian_[j*num_variables_ + row];
        if(diag) {
          element_value += 1.0;
        }

        if(diag || fabs(element_value) > preconditioner_threshold) { // directly copy diagonal and any 
          //off-diagonal terms larger than  tol
          preconditioner_data_[preconditioner_nnz]=element_value;
          preconditioner_row_indexes_[preconditioner_nnz] = row;
          preconditioner_column_sums_[j+1] += 1;
          preconditioner_nnz += 1;
        }
      }
    }
  } else {
    for(int j=0; j<num_vars; ++j) {
      preconditioner_column_sums_[j+1] = preconditioner_column_sums_[j];
      for(int k=(*jacobian_column_sums_ptr_)[j]; k<(*jacobian_column_sums_ptr_)[j+1]; ++k) {
        int row = (*jacobian_row_indexes_ptr_)[k];
        bool diag = row == j;
        double element_value = -gamma*jacobian_data_[k];
        if(diag) {
          element_value += 1.0;
        }

        if(diag || fabs(element_value) > preconditioner_threshold) { // directly copy diagonal and any 
          //off-diagonal terms larger than  tol
          preconditioner_data_[preconditioner_nnz]=element_value;
          preconditioner_row_indexes_[preconditioner_nnz] = row;
          preconditioner_column_sums_[j+1] += 1;
          preconditioner_nnz += 1;
        }
      }
    }
  }
  preconditioner_data_.resize(preconditioner_nnz);
  preconditioner_row_indexes_.resize(preconditioner_nnz,0);

  //Check for mismatch
  if(prev_preconditioner_column_sums.size() != preconditioner_column_sums_.size()) {
    return 1;
  }
  if(prev_preconditioner_row_indexes.size() != preconditioner_row_indexes_.size()) {
    return 1;
  }
  for(int j=1; j<=num_vars; ++j) {
    if(prev_preconditioner_column_sums[j] != preconditioner_column_sums_[j]) {
      return 1;
    }
  }
  for(int j=0; j<preconditioner_nnz; ++j) {
    if(prev_preconditioner_row_indexes[j] != preconditioner_row_indexes_[j]) {
      return 1;
    }
  }
  return 0;
}

int ReactorNVectorSerial::JacobianSolve(double t, N_Vector y, N_Vector fy,
                                            N_Vector r, N_Vector z)
{
  double * r_ptr = NV_DATA_S(r);
  double * z_ptr = NV_DATA_S(z);
  int flag = 0;
  if(int_options_["dense"] == 1) {
    flag = lpm_.solve(num_variables_, r_ptr, z_ptr);
  } else {
    flag = slum_.solve(num_variables_, r_ptr, z_ptr);
  }
  return flag;
}

int ReactorNVectorSerial::ComplexJacobianFactor(int k, double alpha, double beta)
{
  int flag = 0;
  bool dense_numeric = (int_options_["dense"] == 1) && (int_options_["analytic"] == 0);
  if( !dense_numeric ) {
    int preconditioner_nnz = 0;

    preconditioner_column_sums_z_[k].assign(num_variables_+1,0);
    preconditioner_row_indexes_z_[k].assign(nnz_,0);
    doublecomplex zero;
    zero.r = 0.0;
    zero.i = 0.0;
    preconditioner_data_z_[k].assign(nnz_,zero);
    for(int j=0; j<num_variables_; j++) { // column number
      preconditioner_column_sums_z_[k][j+1] = preconditioner_column_sums_z_[k][j];
      for(int i=(*jacobian_column_sums_ptr_)[j]; i<(*jacobian_column_sums_ptr_)[j+1]; ++i) { //row
        int row = (*jacobian_row_indexes_ptr_)[i];
        bool diag = row == j;
        double element_value = -jacobian_data_[i];

        preconditioner_data_z_[k][preconditioner_nnz].r = element_value;
        preconditioner_data_z_[k][preconditioner_nnz].i = 0.0;
        if(diag) {
          preconditioner_data_z_[k][preconditioner_nnz].r += alpha;
          preconditioner_data_z_[k][preconditioner_nnz].i = beta;
        }
        preconditioner_row_indexes_z_[k][preconditioner_nnz] = row;
        preconditioner_column_sums_z_[k][j+1] += 1;
        preconditioner_nnz += 1;
      }
    }
    preconditioner_row_indexes_z_[k].resize(preconditioner_nnz);
    preconditioner_data_z_[k].resize(preconditioner_nnz);
  }


  if(int_options_["dense"] == 1) {
    std::complex<double> czero(0.0, 0.0);
    std::vector<std::complex<double>> dense_preconditioner_z(num_variables_*num_variables_, czero);
    if(int_options_["analytic"] == 1) {
      SparseToDenseComplex(preconditioner_column_sums_z_[k], preconditioner_row_indexes_z_[k], preconditioner_data_z_[k],
                           &dense_preconditioner_z);
    } else {
      for(int col = 0; col < num_variables_; ++col) {
        for(int row = 0; row < num_variables_; ++row) {
          const int idx = row*num_variables_ + col;
          dense_preconditioner_z[idx].real(-dense_jacobian_[idx]);
	  if(row == col) {
            dense_preconditioner_z[idx] += std::complex<double>(alpha , beta);
	  }
        }
      }
    }
    flag = lpmz_[k].factor(num_variables_, num_variables_, dense_preconditioner_z);
  } else {
    flag = slumz_[k].refactor(preconditioner_data_z_[k]);
    if(flag != 0) {
      flag = slumz_[k].factor(preconditioner_row_indexes_z_[k],
                              preconditioner_column_sums_z_[k],
                              preconditioner_data_z_[k],
                              superlu_manager_z::CSC);
    }
  }
  return flag;
}


int ReactorNVectorSerial::ComplexJacobianSolve(int k, N_Vector ax, N_Vector bx)
{
  double * a_ptr = NV_DATA_S(ax); //real
  double * b_ptr = NV_DATA_S(bx); //imaginary
  int flag = 0;
  if(int_options_["dense"] == 1) {
    std::vector<std::complex<double>> rhs(num_variables_);
    std::vector<std::complex<double>> sol(num_variables_);
    for(int i=0; i<num_variables_; ++i) {
      rhs[i] = std::complex<double>(a_ptr[i], b_ptr[i]);
    }
    flag = lpmz_[k].solve(rhs, &sol);
    for(int i=0; i<num_variables_; ++i) {
      a_ptr[i] = sol[i].real();
      b_ptr[i] = sol[i].imag();
    }
  } else {
    std::vector<doublecomplex> rhs(num_variables_);
    std::vector<doublecomplex> sol(num_variables_);

    for(int i=0; i<num_variables_; ++i) {
      rhs[i].r = a_ptr[i];
      rhs[i].i = b_ptr[i];
    }
    flag = slumz_[k].solve(rhs, &sol);
    for(int i=0; i<num_variables_; ++i) {
      a_ptr[i] = sol[i].r;
      b_ptr[i] = sol[i].i;
    }
  }

  return flag;
}

int ReactorNVectorSerial::SetupJacobianSparse(realtype t, N_Vector y,N_Vector fy)
{
  double *y_ptr = NV_DATA_S(y);
  double *fy_ptr = NV_DATA_S(fy);
  double *tmp1_ptr = NV_DATA_S(tmp1_);
  double *tmp2_ptr = NV_DATA_S(tmp2_);
  double *tmp3_ptr = NV_DATA_S(tmp3_);

  //double startTime = getHighResolutionTime();

  const int num_vars = num_variables_;
  const int num_spec = num_species_;
  const int num_destruction_terms = destruction_terms_.concentration_indexes.size();
  const int num_creation_terms = creation_terms_.concentration_indexes.size();
  const double min_mass_frac = double_options_["min_mass_fraction"];

  // set tmp1_ to the strictly positive mass fraction array
  // set tmp2_ to 1/C where C is the concentration for the strictly positive
  // mass fraction array
  for(int j=0; j < num_spec; j++) {
    tmp1_ptr[j]=y_ptr[j];
    if(fabs(tmp1_ptr[j]) < min_mass_frac) {
      if(tmp1_ptr[j] < 0.0) {
        tmp1_ptr[j]= -min_mass_frac;
      } else {
        tmp1_ptr[j]=  min_mass_frac;
      }
    }
    tmp2_ptr[j]=mol_wt_[j]/tmp1_ptr[j]*inverse_density_;
  }

  if(solve_temperature_) {
    tmp1_ptr[num_spec]=y_ptr[num_spec];
  }

  // calculate the reaction info for the strictly positive case
  //double start_time_deriv = getHighResolutionTime();
  GetTimeDerivative(t,tmp1_,tmp3_);
  //double deriv_time = getHighResolutionTime() - start_time_deriv;

  // set the full sparse array
  jacobian_data_.assign(nnz_,0.0);

  // process the forward destruction terms
  for(int j=0; j < num_destruction_terms; ++j) {
      int conc_idx = destruction_terms_.concentration_indexes[j];
      int rxn_idx    = destruction_terms_.reaction_indexes[j];
      int sparse_idx = (*destruction_sparse_indexes_ptr_)[j];
      jacobian_data_[sparse_idx]-=forward_rates_of_production_[rxn_idx]*tmp2_ptr[conc_idx];
  }

  // process the forward creation terms
  for(int j=0; j < num_creation_terms; ++j) {
      int conc_idx = creation_terms_.concentration_indexes[j];
      int rxn_idx    = creation_terms_.reaction_indexes[j];
      int sparse_idx = (*creation_sparse_indexes_ptr_)[j];
      jacobian_data_[sparse_idx]+=forward_rates_of_production_[rxn_idx]*tmp2_ptr[conc_idx];
  }

  // process the non-integer Jacobian information
  if(num_noninteger_jacobian_nonzeros_ > 0) {
    const int num_noninteger_jacobian_nonzeros =
      num_noninteger_jacobian_nonzeros_;

    for(int j=0; j<num_noninteger_jacobian_nonzeros; ++j) {
      noninteger_jacobian_[j] = 0.0;
    }

    mech_ptr_->getNonIntegerReactionNetwork()->GetSpeciesJacobian(
      tmp2_ptr,
      &forward_rates_of_production_[0],
      &noninteger_jacobian_[0]);

    for(int j=0; j<num_noninteger_jacobian_nonzeros; ++j) {
      jacobian_data_[(*noninteger_sparse_id_ptr_)[j]] += noninteger_jacobian_[j];
    }
  }

  // At this point sMptr stores d(wdot[k])/dC[j] ignoring the contribution
  // of perturbations in the third body species

  if(solve_temperature_) {
    // ---------------------------------------------------------------------
    // compute d(Tdot)/dy[j]
    const double RuTemp= mech_ptr_->getGasConstant() * y_ptr[num_spec];

    // SPARSE:
    // step 1: compute matrix vector product (d(wdot[k])/dC[j])*E[k]
    for(int j=0; j<num_spec; j++) { // column number
      int last_row_idx = last_row_indexes_[j];

      jacobian_data_[last_row_idx]=0.0;
      //N.B. skip last row in this loop
      for(int k=(*jacobian_column_sums_ptr_)[j]; k<(*jacobian_column_sums_ptr_)[j+1]-1; k++) {
        int row=(*jacobian_row_indexes_ptr_)[k];
        jacobian_data_[last_row_idx]+=jacobian_data_[k]*energy_[row];
      }
      jacobian_data_[last_row_idx] *= RuTemp;
    }

    // step 2: add the Tdot*Cv[j] term to d(Tdot)/dy[j]
    for(int j=0; j<num_spec; j++) {
      int last_row_idx = last_row_indexes_[j];
      jacobian_data_[last_row_idx]+=tmp3_ptr[num_spec]*cx_mass_[j]*mol_wt_[j];
    }

    // step 3: divide by -1/(molWt[j]*meanCvMass)
    double multFact = -1.0/(mean_cx_mass_);
    for(int j=0; j<num_spec; j++) {
      int last_row_idx = last_row_indexes_[j];
      jacobian_data_[last_row_idx] *= inv_mol_wt_[j]*multFact;
    }

    // At this point Mtx stores d(Tdot[k])/dy[j] ignoring the contribution
    // of perturbations in the third body species

    // --------------------------------------------------------------------- 
    // calculate d(ydot[k])/dT

    // step 1: perturb the temperature
    double delta_temp=y_ptr[num_spec]*sqrt_unit_round_;
    tmp1_ptr[num_spec]+=delta_temp;
    delta_temp=tmp1_ptr[num_spec]-y_ptr[num_spec];
    double multiplier = 1.0/delta_temp;

    for(int j=0; j<num_spec; ++j) {
      tmp1_ptr[j]=y_ptr[j];
    }

    // step 2: calculate ydot at Temp+dTemp
    //start_time_deriv = getHighResolutionTime();
    GetTimeDerivative(t,tmp1_,tmp3_);
    //deriv_time += getHighResolutionTime() - start_time_deriv;

    // step 3: approximate d(ydot[k])/dT with finite difference
    for(int k=(*jacobian_column_sums_ptr_)[num_spec]; k<(*jacobian_column_sums_ptr_)[num_vars]; ++k) {
      int row=(*jacobian_row_indexes_ptr_)[k];
      jacobian_data_[k]=(tmp3_ptr[row]-fy_ptr[row])*multiplier;
    }
  } //if(solve_temperature_)

  //Convert concentration to mass fraction
  for(int j=0; j<num_spec; j++) { // jth column
    for(int k=(*jacobian_column_sums_ptr_)[j]; k<(*jacobian_column_sums_ptr_)[j+1]; k++) {
      const int row=(*jacobian_row_indexes_ptr_)[k];
      if(row < num_spec) {
        jacobian_data_[k] *= mol_wt_[row]*inv_mol_wt_[j];
      }
    }
  }

  //jacobian_setup_time += getHighResolutionTime() - start_time - deriv_time;
  //jacobian_setups += 1;
  //bool print_jacobian = false;
  //if(print_jacobian) {
  //  print_sp_matrix(num_variables_, num_variables_, &(*jacobian_column_sums_ptr_)[0], &(*jacobian_row_indexes_ptr_)[0], &jacobian_data_[0]);
  //  std::vector<double> dense;
  //  SparseToDense(*jacobian_column_sums_ptr_, *jacobian_row_indexes_ptr_, jacobian_data_, &dense);
  //  printf("\n");
  //}
  return 0;
}


int ReactorNVectorSerial::RootFunction(double t, N_Vector y, double *root_function)
{
  if(solve_temperature_) {
    double ignition_temperature = initial_temperature_ + double_options_["delta_temperature_ignition"];
    double current_temperature = NV_Ith_S(y,num_species_)*double_options_["reference_temperature"];
    root_function[0] = ignition_temperature - current_temperature;
  } else {
    root_function[0] = -1;
  }
  return 0;
}

int ReactorNVectorSerial::SetRootTime(double t)
{
  root_time_ = t;
  return 0;
}

double ReactorNVectorSerial::GetRootTime()
{
  return root_time_;
}

int ReactorNVectorSerial::GetNumStateVariables()
{
  return num_variables_;
}


int ReactorNVectorSerial::GetNumRootFunctions()
{
  if(double_options_["delta_temperature_ignition"] > 0) {
    return 1;
  } else {
    return 0;
  }
}

void ReactorNVectorSerial::SetStepLimiter(double value) {
  step_limiter_.assign(num_steps_, value);
}

void ReactorNVectorSerial::DividedDifferenceJacobian(double t, N_Vector y, N_Vector fy,
                                                     std::vector<double>* dense_jacobian) {
  static const int n = num_variables_;
  static const double uround = 1.0e-16;
  double* yd = NV_DATA_S(y);
  double* dy0d = NV_DATA_S(tmp1_);
  double* dy1d = NV_DATA_S(tmp2_);

  this->GetTimeDerivative(t,y,tmp1_);
  dense_jacobian->assign(num_variables_*num_variables_,0.0);
  for(int i = 0; i < n; ++i) {
     double ysafe=yd[i];
     double delta=sqrt(uround*std::max(1.0e-5,fabs(ysafe)));
     yd[i]=ysafe+delta;
     this->GetTimeDerivative(t,y,tmp2_);
     for(int j = 0; j < n; ++j) {
       (*dense_jacobian)[i*n + j] = (dy1d[j] - dy0d[j])/delta;
     }
     yd[i] = ysafe;
  }
}

void ReactorNVectorSerial::Reset() {
  slum_.reset();
}

