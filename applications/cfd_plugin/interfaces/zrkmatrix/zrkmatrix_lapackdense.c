/* -----------------------------------------------------------------
 * Zero-RK Modification of Sundials Dense Matrix Module to use
 * BLAS routines
 *
 * Programmer(s): Daniel Reynolds @ SMU
 *                David Gardner @ LLNL
 * Based on code sundials_dense.c by: Scott D. Cohen,
 *     Alan C. Hindmarsh and Radu Serban @ LLNL
 * -----------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2021, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 * -----------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>

#include "zrkmatrix_lapackdense.h"
#include <sundials/sundials_types.h>

#if defined(SUNDIALS_DOUBLE_PRECISION)
extern int dscal_(int *n, double *alpha, double *x, int *incx);
extern int dcopy_(int *n, double *x, int* incx, double *y, int* incy);
extern int daxpy_(int *n, double *alpha, double* x, int* incx, double* y, int* incy);
extern int dgemv_(const char* trans, int* m, int* n, double* alpha, double* a, int* lda, double* x, int* incx, double* beta, double* y, int* incy);
#define xscal dscal_
#define xcopy dcopy_
#define xaxpy daxpy_
#define xgemv dgemv_
#elif defined(SUNDIALS_SINGLE_PRECISION)
extern int sscal_(int *n, float *alpha, float *x, int *incx);
extern int scopy_(int *n, float *x, int* incx, float *y, int* incy);
extern int saxpy_(int *n, float *alpha, float* x, int* incx, float* y, int* incy);
extern int sgemv_(const char* trans, int* m, int* n, float* alpha, float* a, int* lda, float* x, int* incx, float* beta, float* y, int* incy);
#define xscal sscal_
#define xcopy scopy_
#define xaxpy saxpy_
#define xgemv sgemv_
#else
#error  Incompatible realtype for ZRKLapackDense Matrix
#endif

#define ZERO RCONST(0.0)
#define ONE  RCONST(1.0)


/* Private function prototypes */
static booleantype SMCompatible_ZRKLapackDense(SUNMatrix A, SUNMatrix B);
static booleantype SMCompatible2_ZRKLapackDense(SUNMatrix A, N_Vector x, N_Vector y);

/*
 * -----------------------------------------------------------------
 * exported functions
 * -----------------------------------------------------------------
 */

/* ----------------------------------------------------------------------------
 * Function to create a new dense matrix
 */

SUNMatrix ZRKDenseLapackMatrix(sunindextype M, sunindextype N)
{
  SUNMatrix A;
  SUNMatrixContent_ZRKLapackDense content;
  sunindextype j;

  /* return with NULL matrix on illegal dimension input */
  if ( (M <= 0) || (N <= 0) ) return(NULL);

  /* Create an empty matrix object */
  A = NULL;
  A = SUNMatNewEmpty();
  if (A == NULL) return(NULL);

  /* Attach operations */
  A->ops->getid     = SUNMatGetID_ZRKLapackDense;
  A->ops->clone     = SUNMatClone_ZRKLapackDense;
  A->ops->destroy   = SUNMatDestroy_ZRKLapackDense;
  A->ops->zero      = SUNMatZero_ZRKLapackDense;
  A->ops->copy      = SUNMatCopy_ZRKLapackDense;
  A->ops->scaleadd  = SUNMatScaleAdd_ZRKLapackDense;
  A->ops->scaleaddi = SUNMatScaleAddI_ZRKLapackDense;
  A->ops->matvec    = SUNMatMatvec_ZRKLapackDense;
  A->ops->space     = SUNMatSpace_ZRKLapackDense;

  /* Create content */
  content = NULL;
  content = (SUNMatrixContent_ZRKLapackDense) malloc(sizeof *content);
  if (content == NULL) { SUNMatDestroy(A); return(NULL); }

  /* Attach content */
  A->content = content;

  /* Fill content */
  content->M     = M;
  content->N     = N;
  content->ldata = M*N;
  content->data  = NULL;
  content->cols  = NULL;

  /* Allocate content */
  content->data = (realtype *) calloc(M * N, sizeof(realtype));
  if (content->data == NULL) { SUNMatDestroy(A); return(NULL); }

  content->cols = (realtype **) malloc(N * sizeof(realtype *));
  if (content->cols == NULL) { SUNMatDestroy(A); return(NULL); }
  for (j=0; j<N; j++) content->cols[j] = content->data + j * M;

  return(A);
}


/* ----------------------------------------------------------------------------
 * Function to print the dense matrix 
 */
 
void ZRKDenseLapackMatrix_Print(SUNMatrix A, FILE* outfile)
{
  sunindextype i, j;
  
  /* should not be called unless A is a dense matrix; 
     otherwise return immediately */
  if (SUNMatGetID(A) != SUNMATRIX_DENSE)
    return;

  /* perform operation */
  fprintf(outfile,"\n");
  for (i=0; i<ZM_ROWS_D(A); i++) {
    for (j=0; j<ZM_COLUMNS_D(A); j++) {
#if defined(SUNDIALS_EXTENDED_PRECISION)
      fprintf(outfile,"%12Lg  ", ZM_ELEMENT_D(A,i,j));
#elif defined(SUNDIALS_DOUBLE_PRECISION)
      fprintf(outfile,"%12g  ", ZM_ELEMENT_D(A,i,j));
#else
      fprintf(outfile,"%12g  ", ZM_ELEMENT_D(A,i,j));
#endif
    }
    fprintf(outfile,"\n");
  }
  fprintf(outfile,"\n");
  return;
}


/* ----------------------------------------------------------------------------
 * Functions to access the contents of the dense matrix structure
 */

sunindextype ZRKDenseLapackMatrix_Rows(SUNMatrix A)
{
  if (SUNMatGetID(A) == SUNMATRIX_DENSE)
    return ZM_ROWS_D(A);
  else
    return SUNMAT_ILL_INPUT;
}

sunindextype ZRKDenseLapackMatrix_Columns(SUNMatrix A)
{
  if (SUNMatGetID(A) == SUNMATRIX_DENSE)
    return ZM_COLUMNS_D(A);
  else
    return SUNMAT_ILL_INPUT;
}

sunindextype ZRKDenseLapackMatrix_LData(SUNMatrix A)
{
  if (SUNMatGetID(A) == SUNMATRIX_DENSE)
    return ZM_LDATA_D(A);
  else
    return SUNMAT_ILL_INPUT;
}

realtype* ZRKDenseLapackMatrix_Data(SUNMatrix A)
{
  if (SUNMatGetID(A) == SUNMATRIX_DENSE)
    return ZM_DATA_D(A);
  else
    return NULL;
}

realtype** ZRKDenseLapackMatrix_Cols(SUNMatrix A)
{
  if (SUNMatGetID(A) == SUNMATRIX_DENSE)
    return ZM_COLS_D(A);
  else
    return NULL;
}

realtype* ZRKDenseLapackMatrix_Column(SUNMatrix A, sunindextype j)
{
  if (SUNMatGetID(A) == SUNMATRIX_DENSE)
    return ZM_COLUMN_D(A,j);
  else
    return NULL;
}


/*
 * -----------------------------------------------------------------
 * implementation of matrix operations
 * -----------------------------------------------------------------
 */

SUNMatrix_ID SUNMatGetID_ZRKLapackDense(SUNMatrix A)
{
  return SUNMATRIX_DENSE;
}

SUNMatrix SUNMatClone_ZRKLapackDense(SUNMatrix A)
{
  SUNMatrix B = ZRKDenseLapackMatrix(ZM_ROWS_D(A), ZM_COLUMNS_D(A));
  return(B);
}

void SUNMatDestroy_ZRKLapackDense(SUNMatrix A)
{
  if (A == NULL) return;

  /* free content */
  if (A->content != NULL) {
    /* free data array */
    if (ZM_DATA_D(A) != NULL) {
      free(ZM_DATA_D(A));
      ZM_DATA_D(A) = NULL;
    }
    /* free column pointers */
    if (ZM_CONTENT_D(A)->cols != NULL) {
      free(ZM_CONTENT_D(A)->cols);
      ZM_CONTENT_D(A)->cols = NULL;
    }
    /* free content struct */
    free(A->content);
    A->content = NULL;
  }

  /* free ops and matrix */
  if (A->ops) { free(A->ops); A->ops = NULL; }
  free(A); A = NULL;

  return;
}

int SUNMatZero_ZRKLapackDense(SUNMatrix A)
{
  sunindextype ntot, inc;
  realtype *Adata, zero;

  /* Perform operation */
  ntot = ZM_LDATA_D(A);
  Adata = ZM_DATA_D(A);
  inc = 1;
  zero = ZERO;
  xscal(&ntot, &zero, Adata, &inc); //scale A to zero
//  for (i=0; i<ZM_LDATA_D(A); i++)
//    Adata[i] = ZERO;
  return SUNMAT_SUCCESS;
}

int SUNMatCopy_ZRKLapackDense(SUNMatrix A, SUNMatrix B)
{
  sunindextype ntot, inc;
  realtype *Adata, *Bdata;

  /* Verify that A and B are compatible */
  if (!SMCompatible_ZRKLapackDense(A, B))
    return SUNMAT_ILL_INPUT;

  /* Perform operation */
  ntot = ZM_LDATA_D(A);
  Adata = ZM_DATA_D(A);
  Bdata = ZM_DATA_D(B);
  inc = 1;
  xcopy(&ntot, Adata, &inc, Bdata, &inc);
//  for (j=0; j<ZM_COLUMNS_D(A); j++)
//    for (i=0; i<ZM_ROWS_D(A); i++)
//      ZM_ELEMENT_D(B,i,j) = ZM_ELEMENT_D(A,i,j);
  return SUNMAT_SUCCESS;
}

int SUNMatScaleAddI_ZRKLapackDense(realtype c, SUNMatrix A)
{
  sunindextype j, ntot, nrows, ncols, nlimit, inc;
  realtype *Adata;

  /* Perform operation */
  ntot = ZM_LDATA_D(A);
  nrows = ZM_ROWS_D(A);
  ncols = ZM_COLUMNS_D(A);
  nlimit = (nrows > ncols) ? nrows : ncols;
  Adata = ZM_DATA_D(A);
  inc = 1;
  xscal(&ntot, &c, Adata, &inc); //scale A
  for (j = 0; j < nlimit; ++j) {
    Adata[j*nrows+j] += 1.0;
  }
//  for (j=0; j<ZM_COLUMNS_D(A); j++)
//    for (i=0; i<ZM_ROWS_D(A); i++) {
//      ZM_ELEMENT_D(A,i,j) *= c;
//      if (i == j) 
//        ZM_ELEMENT_D(A,i,j) += ONE;
//    }
  return SUNMAT_SUCCESS;
}

int SUNMatScaleAdd_ZRKLapackDense(realtype c, SUNMatrix A, SUNMatrix B)
{
  sunindextype ntot, inc;
  realtype one, *Adata, *Bdata;

  /* Verify that A and B are compatible */
  if (!SMCompatible_ZRKLapackDense(A, B))
    return SUNMAT_ILL_INPUT;

  /* Perform operation */
  ntot = ZM_LDATA_D(A);
  Adata = ZM_DATA_D(A);
  Bdata = ZM_DATA_D(B);
  one = 1.0;
  inc = 1;
  xscal(&ntot, &c, Adata, &inc); //scale A
  xaxpy(&ntot, &one, Bdata, &inc, Adata, &inc); //add B to A
//  for (j=0; j<ZM_COLUMNS_D(A); j++)
//    for (i=0; i<ZM_ROWS_D(A); i++)
//      ZM_ELEMENT_D(A,i,j) = c*ZM_ELEMENT_D(A,i,j) + ZM_ELEMENT_D(B,i,j);
  return SUNMAT_SUCCESS;
}

int SUNMatMatvec_ZRKLapackDense(SUNMatrix A, N_Vector x, N_Vector y)
{
  sunindextype nrows, ncols, inc;
  realtype zero, one, *xd, *yd, *Adata;
  
  /* Verify that A, x and y are compatible */
  if (!SMCompatible2_ZRKLapackDense(A, x, y))
    return SUNMAT_ILL_INPUT;

  /* access vector data (return if failure) */
  xd = N_VGetArrayPointer(x);
  yd = N_VGetArrayPointer(y);
  if ((xd == NULL) || (yd == NULL) || (xd == yd))
    return SUNMAT_MEM_FAIL;

  /* Perform operation */
  nrows = ZM_ROWS_D(A);
  ncols = ZM_COLUMNS_D(A);
  Adata = ZM_DATA_D(A);
  one = 1.0;
  zero = ZERO;
  inc = 1;
  xgemv("N", &nrows, &ncols, &one, Adata, &nrows, xd, &inc, &zero, yd, &inc);
  return SUNMAT_SUCCESS;
}

int SUNMatSpace_ZRKLapackDense(SUNMatrix A, long int *lenrw, long int *leniw)
{
  *lenrw = ZM_LDATA_D(A);
  *leniw = 3 + ZM_COLUMNS_D(A);
  return SUNMAT_SUCCESS;
}


/*
 * -----------------------------------------------------------------
 * private functions
 * -----------------------------------------------------------------
 */

static booleantype SMCompatible_ZRKLapackDense(SUNMatrix A, SUNMatrix B)
{
  /* both matrices must be SUNMATRIX_DENSE */
  if (SUNMatGetID(A) != SUNMATRIX_DENSE)
    return SUNFALSE;
  if (SUNMatGetID(B) != SUNMATRIX_DENSE)
    return SUNFALSE;

  /* both matrices must have the same shape */
  if (ZM_ROWS_D(A) != ZM_ROWS_D(B))
    return SUNFALSE;
  if (ZM_COLUMNS_D(A) != ZM_COLUMNS_D(B))
    return SUNFALSE;

  return SUNTRUE;
}


static booleantype SMCompatible2_ZRKLapackDense(SUNMatrix A, N_Vector x, N_Vector y)
{
  /*   vectors must be one of {SERIAL, OPENMP, PTHREADS} */ 
  if ( (N_VGetVectorID(x) != SUNDIALS_NVEC_SERIAL) &&
       (N_VGetVectorID(x) != SUNDIALS_NVEC_OPENMP) &&
       (N_VGetVectorID(x) != SUNDIALS_NVEC_PTHREADS) )
    return SUNFALSE;

  /* Optimally we would verify that the dimensions of A, x and y agree, 
   but since there is no generic 'length' routine for N_Vectors we cannot */

  return SUNTRUE;
}

