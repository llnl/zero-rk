#ifndef SUPERLU_MANAGER_H
#define SUPERLU_MANAGER_H

#include <vector>
#include <memory> //std::unique_ptr

struct slu_data;

class superlu_manager
{
 public:
  superlu_manager();
  virtual ~superlu_manager();

  enum matrix_t { CSC, CSR };
  int factor(const std::vector<int>& indexs, const std::vector<int>& sums,
             const std::vector<double>& values, superlu_manager::matrix_t type);
  int factor(int n, int nnz, const int* indexs, const int* sums,
             const double* values, superlu_manager::matrix_t type);
  int refactor(const std::vector<double>& values);
  int refactor(int nnz, const double* values);
  int solve(const std::vector<double> rhs, std::vector<double>* soln);
  int solve(int n, const double* rhs, double* soln);

  bool factored() { return this->m_factored; };

  void reset();

  void SetTranspose(const bool transpose);

  int GetLU(int* L_nnz,
            std::vector<int>* L_sums,
            std::vector<int>* L_indexes,
            std::vector<double>* L_values,
            int* U_nnz,
            std::vector<int>* U_sums,
            std::vector<int>* U_indexes,
            std::vector<double>* U_values,
            matrix_t output_format,
            bool transpose);

  int GetInversePermutations(std::vector<int> *invRowPerm,
                             std::vector<int> *invColPerm);

 private:
  void setup_memory(int n, int nnz, const int* indexes, const int* sums,
                    const double* values);

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
