#include "superlu_manager.h"

#include <algorithm>
#include "slu_ddefs.h"
#include "slu_data.h"

superlu_manager::superlu_manager() :
  m_last_factor_nnz(-1),
  m_factored(false),
  m_slu_container(std::make_unique<slu_data>())
{
  dCreate_CompCol_Matrix(&m_slu_container->M, 1, 1, 1, NULL, NULL, NULL, SLU_NC,SLU_D,SLU_GE);
  dCreate_Dense_Matrix(&m_slu_container->B,1,1,NULL,1,SLU_DN,SLU_D,SLU_GE);
  dCreate_Dense_Matrix(&m_slu_container->X,1,1,NULL,1,SLU_DN,SLU_D,SLU_GE);
  set_default_options(&m_slu_container->options);
  m_slu_container->options.ColPerm = MY_PERMC;
  m_slu_container->options.RowPerm = LargeDiag_MC64;
  m_slu_container->options.Equil   = YES;
  m_slu_container->options.DiagPivotThresh = 0.0;

  StatInit(&m_slu_container->stats);
}

superlu_manager::~superlu_manager()
{
  StatFree(&m_slu_container->stats);
  Destroy_SuperMatrix_Store(&m_slu_container->B);
  Destroy_SuperMatrix_Store(&m_slu_container->X);
  if(m_factored) {
    Destroy_SuperNode_Matrix(&m_slu_container->L);
    Destroy_CompCol_Matrix(&m_slu_container->U);
  }
  Destroy_SuperMatrix_Store(&m_slu_container->M);
}

void superlu_manager::setup_memory(int n, int nnz, const int* indexes, const int* sums, const double* values) {
  m_slu_container->M.nrow = n;
  m_slu_container->M.ncol = n;

  m_colPermutation.assign(n,0);
  m_rowPermutation.assign(n,0);
  m_colElimTree.assign(n,0);
  m_R.assign(n,0.0);
  m_C.assign(n,0.0);

  m_slu_container->B.nrow=n;
  m_slu_container->X.nrow=n;
  m_slu_container->B.ncol=1;
  m_slu_container->X.ncol=1;
  ((DNformat *)m_slu_container->B.Store)->lda = n;
  ((DNformat *)m_slu_container->X.Store)->lda = n;
}


int superlu_manager::factor(const std::vector<int>& indexes,
                            const std::vector<int>& sums,
                            const std::vector<double>& values,
                            matrix_t type)
{
  int n = sums.size()-1;
  int nnz = indexes.size();

  if(values.size() != nnz) {
    return 1;
  }

  return this->factor(n, nnz, &(indexes[0]), &(sums[0]), &(values[0]), type);
}

int superlu_manager::factor(int n,
                            int nnz,
                            const int* indexes,
                            const int* sums,
                            const double* values,
                            matrix_t type)
{
  if(nnz != m_last_factor_nnz || n != m_last_factor_n) {
    setup_memory(n, nnz, indexes, sums, values);
  }

  ((NCformat *)m_slu_container->M.Store)->nnz = nnz;
  ((NCformat *)m_slu_container->M.Store)->nzval = const_cast<double*>(values);
  ((NCformat *)m_slu_container->M.Store)->rowind = const_cast<int*>(indexes);
  ((NCformat *)m_slu_container->M.Store)->colptr = const_cast<int*>(sums);

  if(m_factored) {
    Destroy_SuperNode_Matrix(&m_slu_container->L);
    Destroy_CompCol_Matrix(&m_slu_container->U);
  }

  m_slu_container->B.ncol=0;
  m_slu_container->options.Fact=DOFACT;
  get_perm_c(MMD_AT_PLUS_A, &m_slu_container->M, &m_colPermutation[0]);

  int lwork=0;      // SuperLU allocates its own memeory
  void *work=NULL;  // ''
  double rpg,rcond; // reciprocal pivot growth, recip. condition number
  double *ferr = NULL; //Not factoring so un-used
  double *berr = NULL;
  int flag;
  dgssvx(&m_slu_container->options,&m_slu_container->M, &m_colPermutation[0],&m_rowPermutation[0],
         &m_colElimTree[0],m_equed,&m_R[0],&m_C[0],&m_slu_container->L, &m_slu_container->U, work, lwork,
         &m_slu_container->B, &m_slu_container->X, &rpg, &rcond, ferr, berr,
         &m_slu_container->Glu,
         &m_slu_container->mem_usage, &m_slu_container->stats, &flag);

  m_factored=true;
  m_last_factor_n = n;
  m_last_factor_nnz = nnz;

  return flag;
}


int superlu_manager::refactor(const std::vector<double>& values) {
  int nnz = values.size();
  return this->refactor(nnz, &(values[0]));
}

int superlu_manager::refactor(int nnz, const double* values)
{
  if(nnz != m_last_factor_nnz) {
    return 1;
  }
  ((NCformat *)m_slu_container->M.Store)->nzval = const_cast<double*>(values);
  Destroy_SuperNode_Matrix(&m_slu_container->L);
  Destroy_CompCol_Matrix(&m_slu_container->U);

  m_slu_container->B.ncol=0;
  m_slu_container->options.Fact=SamePattern;

  int lwork=0;      // SuperLU allocates its own memeory
  void *work=NULL;  // ''
  double rpg,rcond; // reciprocal pivot growth, recip. condition number
  double *ferr = NULL; //Not factoring so un-used
  double *berr = NULL;
  int flag;
  dgssvx(&m_slu_container->options,&m_slu_container->M, &m_colPermutation[0],&m_rowPermutation[0],
         &m_colElimTree[0],m_equed,&m_R[0],&m_C[0],&m_slu_container->L, &m_slu_container->U, work, lwork,
         &m_slu_container->B, &m_slu_container->X, &rpg, &rcond, ferr, berr,
         &m_slu_container->Glu,
         &m_slu_container->mem_usage, &m_slu_container->stats, &flag);

  //if(m_slu_container->options.DiagPivotThresh > 0.0) {
    for(int j = 0; j < m_last_factor_n; ++j) {
      if(m_rowPermutation[j] == -1 ) {
        flag = 1;
        break;
      }
    }
  //}

  return flag;
}


int superlu_manager::solve(const std::vector<double> rhs, std::vector<double>* soln) {
  if(rhs.size() != soln->size()) {
    return 1;
  }
  int n = rhs.size();
  return this->solve(n, &(rhs[0]), &((*soln)[0]));
}

int superlu_manager::solve(int n, const double* rhs, double* soln) {
  if(n != m_last_factor_n) {
    return 1;
  }
  int flag;
  int lwork=0;
  void *work=NULL;
  double rpg,rcond;
  double ferr[1],berr[1]; // length is the number of RHS

  m_slu_container->options.Fact=FACTORED;
  m_slu_container->B.ncol=1;

  ((DNformat *)m_slu_container->B.Store)->nzval = const_cast<double*>(rhs);
  ((DNformat *)m_slu_container->X.Store)->nzval = soln;

  //backsolve with expert driver function
  dgssvx(&m_slu_container->options,&m_slu_container->M,&m_colPermutation[0],&m_rowPermutation[0],
         &m_colElimTree[0], m_equed, &m_R[0], &m_C[0], &m_slu_container->L, &m_slu_container->U,
	 work, lwork, &m_slu_container->B, &m_slu_container->X, &rpg, &rcond, ferr, berr,
         &m_slu_container->Glu,
         &m_slu_container->mem_usage, &m_slu_container->stats, &flag);

  return flag;
}

int superlu_manager::GetLU(int* L_nnz,
                           std::vector<int>* L_sums,
                           std::vector<int>* L_indexes,
                           std::vector<double>* L_values,
                           int* U_nnz,
                           std::vector<int>* U_sums,
                           std::vector<int>* U_indexes,
                           std::vector<double>* U_values,
                           matrix_t output_format,
                           bool transpose)
{
  if(!m_factored) {
    return 1;
  }

  const int n = m_last_factor_n;
  // Convert SuperLU L and U to CSR format
  if(transpose) {
    *L_nnz = ((SCformat *)m_slu_container->U.Store)->nnz;
    *U_nnz = ((NCformat *)m_slu_container->L.Store)->nnz;
  } else {
    *L_nnz = ((SCformat *)m_slu_container->L.Store)->nnz;
    *U_nnz = ((NCformat *)m_slu_container->U.Store)->nnz;
  }

  L_sums->resize(n+1);
  L_indexes->resize(*L_nnz);
  L_values->resize(*L_nnz);

  U_sums->resize(n+1);
  U_indexes->resize(*U_nnz);
  U_values->resize(*U_nnz);

  //This converts L and U from SuperLU supernodal to true upper
  //and lower triangular matrices.

  //  (don't store implicit unit diagonal in intermediate form)
  const int num_elems = *L_nnz + *U_nnz - n;
  typedef std::pair<std::pair<int,int>, double>  coo_unit;
  std::vector<coo_unit> scattered_vals(num_elems);
  int scatter_counter = 0;

  //U matrix
  NCformat *Ustore ((NCformat *)m_slu_container->U.Store);
  for(int j = 0; j < n; ++j) {
      for (int i = Ustore->colptr[j]; i < Ustore->colptr[j+1]; ++i) {
         scattered_vals[scatter_counter].first.first = Ustore->rowind[i]; // row
         scattered_vals[scatter_counter].first.second = j; //col
         scattered_vals[scatter_counter].second = ((double*)Ustore->nzval)[i]; //sparse_idx
         ++scatter_counter;
      }
  }

  //L matrix
  SCformat *Lstore = ((SCformat *)m_slu_container->L.Store);
  double* dp = (double *) Lstore->nzval;
  int* sup_to_col = Lstore->sup_to_col;
  int* rowind_colptr = Lstore->rowind_colptr;
  int* rowind = Lstore->rowind;
  for (int k = 0; k <= Lstore->nsuper; ++k) {
    int c = sup_to_col[k];
    for (int j = c; j < sup_to_col[k+1]; ++j) {
      int d = Lstore->nzval_colptr[j];
      for (int i = rowind_colptr[c]; i < rowind_colptr[c+1]; ++i) {
        scattered_vals[scatter_counter].first.first = rowind[i];
        scattered_vals[scatter_counter].first.second = j;
        scattered_vals[scatter_counter].second = dp[d];
        ++scatter_counter;
        ++d;
      }
    }
  }
  assert(scatter_counter == num_elems);

  // Sorted combined matrix.
  if( (output_format == CSC) != transpose ) {
    std::sort(scattered_vals.begin(),scattered_vals.end(),
              [](coo_unit a, coo_unit b) {
                if((a.first.second) == (b.first.second)) {    //by column
                  return ((a.first.first) < (b.first.first)); //by row
                } else {
                  return ((a.first.second) < (b.first.second));
                }
             });
  } else {
    std::sort(scattered_vals.begin(),scattered_vals.end(),
              [](coo_unit a, coo_unit b) {
                if((a.first.first) == (b.first.first)) {        //by row
                  return ((a.first.second) < (b.first.second)); //by column
                } else {
                  return ((a.first.first) < (b.first.first));
                }
             });
  }

  // This L stores unit diagonal.
  (*U_sums)[0] = 0; // by convention
  (*L_sums)[0] = 0;
  (*U_sums)[1] = 0; // start the counter on column 0
  (*L_sums)[1] = 0;

  int Lctr = 0;
  int Uctr = 0;
  int row = 0;
  int col = 0;
  int idx = 0;
  int maj = 0;
  int prev_maj = 0;
  for(int j=0; j<scatter_counter; ++j) {
    if(transpose) {
      col = scattered_vals[j].first.first;
      row = scattered_vals[j].first.second;
    } else {
      row = scattered_vals[j].first.first;
      col = scattered_vals[j].first.second;
    }
    if(output_format == CSR) {
      maj = row;
      idx = col;
    } else {
      maj = col;
      idx = row;
    }
    double val = scattered_vals[j].second;
    //printf("L+U[%d, %d] : %g\n", row, col, val);
    if(maj != prev_maj) {
      assert(prev_maj == (maj-1));
      (*U_sums)[maj+1] = (*U_sums)[maj];
      (*L_sums)[maj+1] = (*L_sums)[maj];
    }
    prev_maj = maj;
    if(row == col) { //diagonal
      (*U_indexes)[Uctr] = idx;
      (*U_values)[Uctr] = val;
      (*U_sums)[maj+1] += 1;
      ++Uctr;
      (*L_indexes)[Lctr] = idx;
      (*L_values)[Lctr] = 1.0;
      (*L_sums)[maj+1] += 1;
      ++Lctr;
    } else if ( row < col ) {
      (*U_indexes)[Uctr] = idx;
      (*U_values)[Uctr] = val;
      (*U_sums)[maj+1] += 1;
      ++Uctr;
    } else {
      (*L_indexes)[Lctr] = idx;
      (*L_values)[Lctr] = val;
      (*L_sums)[maj+1] += 1;
      ++Lctr;
    }
  }
  assert(Lctr == *L_nnz);
  assert(Uctr == *U_nnz);

  *U_nnz = (*U_sums)[n];
  *L_nnz = (*L_sums)[n];
  return 0;
}

int superlu_manager::GetInversePermutations(std::vector<int> *invRowPerm, std::vector<int> *invColPerm)
{
  if(!m_factored) {
    return 1;
  }
  const int n = m_last_factor_n;
  invRowPerm->resize(n);
  invColPerm->resize(n);
  for(int j=0; j<n; ++j) {
    (*invRowPerm)[m_rowPermutation[j]] = j;
    (*invColPerm)[m_colPermutation[j]] = j;
  }
  return 0;
}


void superlu_manager::SetTranspose(const bool transpose)
{
  if(transpose) {
    m_slu_container->options.Trans = TRANS;
  } else {
    m_slu_container->options.Trans = NOTRANS;
  }
}

void superlu_manager::reset()
{
  m_last_factor_nnz = -1;
}
