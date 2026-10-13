#ifndef SUPERLU_MANAGER_Z_H
#define SUPERLU_MANAGER_Z_H

#include <vector>
#include <memory>

#include "slu_dcomplex.h"

struct slu_data;

class superlu_manager_z
{
 public:
  superlu_manager_z();
  virtual ~superlu_manager_z();

  superlu_manager_z(superlu_manager_z&& other) noexcept;
  superlu_manager_z& operator=(superlu_manager_z&& other) noexcept;

  superlu_manager_z(const superlu_manager_z&) = delete;
  superlu_manager_z& operator=(const superlu_manager_z&) = delete;

  enum matrix_t { CSC, CSR };
  int factor(const std::vector<int>& indexs, const std::vector<int>& sums,
             const std::vector<doublecomplex>& values, superlu_manager_z::matrix_t type);
  int factor(int n, int nnz, const int* indexs, const int* sums,
             const doublecomplex* values, superlu_manager_z::matrix_t type);
  int refactor(const std::vector<doublecomplex>& values);
  int refactor(int nnz, const doublecomplex* values);
  int solve(const std::vector<doublecomplex> rhs, std::vector<doublecomplex>* soln);
  int solve(int n, const doublecomplex* rhs, doublecomplex* soln);

  bool factored() { return this->m_factored; };

 private:
  void setup_memory(int n, int nnz, const int* indexs, const int* sums,
                    const doublecomplex* values);

  std::unique_ptr<slu_data> m_slu_container;
  char m_equed[1];
  std::vector<double> m_R;
  std::vector<double> m_C;
  std::vector<int> m_rowPermutation;
  std::vector<int> m_colPermutation;
  std::vector<int> m_colElimTree; // etree

  bool m_factored;
  int m_last_factor_n;
  int m_last_factor_nnz;
};


#endif
