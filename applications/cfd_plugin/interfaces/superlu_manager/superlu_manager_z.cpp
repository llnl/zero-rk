#include "superlu_manager_z.h"

#include "slu_zdefs.h"
#include "slu_data.h"


superlu_manager_z::superlu_manager_z() :
  m_last_factor_nnz(-1),
  m_factored(false),
  m_slu_container(std::make_unique<slu_data>())
{
  zCreate_CompCol_Matrix(&m_slu_container->M, 1, 1, 1, NULL, NULL, NULL, SLU_NC,SLU_Z,SLU_GE);
  zCreate_Dense_Matrix(&m_slu_container->B,1,1,NULL,1, SLU_DN,SLU_Z,SLU_GE);
  zCreate_Dense_Matrix(&m_slu_container->X,1,1,NULL,1, SLU_DN,SLU_Z,SLU_GE);
  set_default_options(&m_slu_container->options);
  m_slu_container->options.ColPerm = MY_PERMC;
  m_slu_container->options.RowPerm = LargeDiag_MC64;
  m_slu_container->options.Equil   = YES;

  StatInit(&m_slu_container->stats);
}


superlu_manager_z::~superlu_manager_z()
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

//These are necessary to be able to construct a vector of superlu_manager_z's
superlu_manager_z::superlu_manager_z(superlu_manager_z&& other) noexcept = default;
superlu_manager_z& superlu_manager_z::operator=(superlu_manager_z&& other) noexcept = default;

void superlu_manager_z::setup_memory(int n, int nnz, const int* indexes, const int* sums, const doublecomplex* values) {
  m_slu_container->M.nrow = n;
  m_slu_container->M.ncol = n;
  ((NCformat *)m_slu_container->M.Store)->nnz = nnz;
  ((NCformat *)m_slu_container->M.Store)->nzval = const_cast<doublecomplex*>(values);
  ((NCformat *)m_slu_container->M.Store)->rowind = const_cast<int*>(indexes);
  ((NCformat *)m_slu_container->M.Store)->colptr = const_cast<int*>(sums);

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


int superlu_manager_z::factor(const std::vector<int>& indexes,
                              const std::vector<int>& sums,
                              const std::vector<doublecomplex>& values,
                              matrix_t type)
{
  int n = sums.size()-1;
  int nnz = indexes.size();

  if(values.size() != nnz) {
    return 1;
  }
  return this->factor(n, nnz, &(indexes[0]), &(sums[0]), &(values[0]), type);
}

int superlu_manager_z::factor(int n,
                              int nnz,
                              const int* indexes,
                              const int* sums,
                              const doublecomplex* values,
                              matrix_t type)
{
  if(nnz != m_last_factor_nnz || n != m_last_factor_n) {
    setup_memory(n,nnz, indexes, sums, values);
  }

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
  zgssvx(&m_slu_container->options,&m_slu_container->M, &m_colPermutation[0],&m_rowPermutation[0],
         &m_colElimTree[0],m_equed,&m_R[0],&m_C[0],&m_slu_container->L, &m_slu_container->U, work, lwork,
         &m_slu_container->B, &m_slu_container->X, &rpg, &rcond, ferr, berr,
         &m_slu_container->Glu,
         &m_slu_container->mem_usage, &m_slu_container->stats, &flag);

  m_factored=true;
  m_last_factor_n = n;
  m_last_factor_nnz = nnz;

  return flag;
}


int superlu_manager_z::refactor(const std::vector<doublecomplex>& values) {
  int nnz = values.size();
  return this->refactor(nnz, &(values[0]));
}

int superlu_manager_z::refactor(int nnz, const doublecomplex* values)
{
  if(nnz != m_last_factor_nnz) {
    return 1;
  }
  ((NCformat *)m_slu_container->M.Store)->nzval = const_cast<doublecomplex*>(values);
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
  zgssvx(&m_slu_container->options,&m_slu_container->M, &m_colPermutation[0],&m_rowPermutation[0],
         &m_colElimTree[0],m_equed,&m_R[0],&m_C[0],&m_slu_container->L, &m_slu_container->U, work, lwork,
         &m_slu_container->B, &m_slu_container->X, &rpg, &rcond, ferr, berr,
         &m_slu_container->Glu,
         &m_slu_container->mem_usage, &m_slu_container->stats, &flag);

  if(m_slu_container->options.DiagPivotThresh > 0.0) {
    for(int j = 0; j < m_last_factor_n; ++j) {
      if(m_rowPermutation[j] == -1 ) {
        flag = 1;
      }
    }
  }

  return flag;
}


int superlu_manager_z::solve(const std::vector<doublecomplex> rhs, std::vector<doublecomplex>* soln) {
  if(rhs.size() != soln->size()) {
    return 1;
  }
  int n = rhs.size();
  return this->solve(n, &(rhs[0]), &((*soln)[0]));
}

int superlu_manager_z::solve(int n, const doublecomplex* rhs, doublecomplex* soln) {
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

  ((DNformat *)m_slu_container->B.Store)->nzval = const_cast<doublecomplex*>(rhs);
  ((DNformat *)m_slu_container->X.Store)->nzval = soln;

  //backsolve with expert driver function
  zgssvx(&m_slu_container->options,&m_slu_container->M,&m_colPermutation[0],&m_rowPermutation[0],
         &m_colElimTree[0], m_equed, &m_R[0], &m_C[0], &m_slu_container->L, &m_slu_container->U,
	 work, lwork, &m_slu_container->B, &m_slu_container->X, &rpg, &rcond, ferr, berr,
         &m_slu_container->Glu,
         &m_slu_container->mem_usage, &m_slu_container->stats, &flag);

  return flag;
}
