#ifndef SUPERLU_MANAGER_SLU_DATA_H
#define SUPERLU_MANAGER_SLU_DATA_H

struct slu_data {
  SuperMatrix M;
  SuperMatrix L;
  SuperMatrix U;
  SuperMatrix B;
  SuperMatrix X;
  superlu_options_t options;
  SuperLUStat_t stats;
  mem_usage_t mem_usage;
  GlobalLU_t Glu;
};

#endif
