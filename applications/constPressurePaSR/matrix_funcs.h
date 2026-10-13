#ifndef MATRIX_FUNCS_H
#define MATRIX_FUNCS_H

#include <stdio.h>
#include <nvector/nvector_serial.h> // serial N_Vector types, fcts., and macros
#include "cv_param_sparse.h"


#if defined SUNDIALS2
int jac_full_prec_setup(realtype t, N_Vector y, N_Vector fy,
			booleantype jok, booleantype *jcurPtr,
			realtype gamma, void *user_data,
			N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

int jac_full_prec_solveV3(realtype t, N_Vector y, N_Vector fy,
			  N_Vector r, N_Vector z, realtype gamma,
			  realtype delta, int lr, void *user_data,
			  N_Vector tmp);

#elif defined SUNDIALS3 || defined SUNDIALS4
int jac_full_prec_setup(realtype t, N_Vector y, N_Vector fy,
                        booleantype jok, booleantype *jcurPtr,
                        realtype gamma, void *user_data);

int jac_full_prec_solveV3(realtype t, N_Vector y, N_Vector fy,
                          N_Vector r, N_Vector z, realtype gamma,
                          realtype delta, int lr, void *user_data);

int jac_full_prec_setup_mix(realtype t, N_Vector y, N_Vector fy,
                        booleantype jok, booleantype *jcurPtr,
                        realtype gamma, void *user_data);

int jac_full_prec_solveV3_mix(realtype t, N_Vector y, N_Vector fy,
                          N_Vector r, N_Vector z, realtype gamma,
                          realtype delta, int lr, void *user_data);
#endif

void setupJacobianSparse(realtype t,N_Vector y,N_Vector fy,cv_param* cvp,N_Vector tmp1,N_Vector tmp2,N_Vector tmp3);


void sparseOffDiagThreshCopy(const double tol, const int nsize, const int nnzA,
			     const double A[], const int ArowId[],
			     const int AcolSum[], int *nnzB, double B[],
			     int BrowId[], int BcolSum[]);

void sparseOffDiagThreshCopyGamma(const double tol, const int nsize, const int nnzA,
			     const double A[], const int ArowId[],
			     const int AcolSum[], int *nnzB, double B[],
			     int BrowId[], int BcolSum[], const double gamma);

void sparseOffDiagThreshCopy_rnorm(const double tol, const int nsize,
			     const int nnzA,
			     const double A[], const int ArowId[],
			     const int AcolSum[], int *nnzB, double B[],
				   int BrowId[], int BcolSum[]);

void sparseOffDiagThreshCopyGamma_rnorm(const double tol, const int nsize,
			     const int nnzA,
			     const double A[], const int ArowId[],
			     const int AcolSum[], int *nnzB, double B[],
				   int BrowId[], int BcolSum[], const double gamma);

void sparseOffDiagThreshCopy_cnorm(const double tol, const int nsize,
			     const int nnzA,
			     const double A[], const int ArowId[],
			     const int AcolSum[], int *nnzB, double B[],
				   int BrowId[], int BcolSum[]);

void sparseOffDiagThreshCopyGamma_cnorm(const double tol, const int nsize,
			     const int nnzA,
			     const double A[], const int ArowId[],
			     const int AcolSum[], int *nnzB, double B[],
				   int BrowId[], int BcolSum[], const double gamma);

void sparseOffDiagThreshCopy_rcmin(const double tol, const int nsize,
			     const int nnzA,
			     const double A[], const int ArowId[],
			     const int AcolSum[], int *nnzB, double B[],
				   int BrowId[], int BcolSum[]);

void sparseOffDiagThreshCopyGamma_rcmin(const double tol, const int nsize,
			     const int nnzA,
			     const double A[], const int ArowId[],
			     const int AcolSum[], int *nnzB, double B[],
				   int BrowId[], int BcolSum[], const double gamma);

void sparseOffDiagMagCopy(const double frac, const int nsize, const int nnzA,
			  const double A[], const int ArowId[],
			  const int AcolSum[], int *nnzB, double B[],
			  int BrowId[], int BcolSum[]);

void sparseOffDiagMagCopyGamma(const double frac, const int nsize, const int nnzA,
			  const double A[], const int ArowId[],
			  const int AcolSum[], int *nnzB, double B[],
			  int BrowId[], int BcolSum[], const double gamma);

void sparseUpdateGamma(const int nsize, const int nnzA,
			     const double A[], const int ArowId[],
			     const int AcolSum[], int nnzB, double B[],
			     int BrowId[], int BcolSum[], const double gamma);

bool reThreshCheck(const double tol, const int nsize, const int nnzA,
		   const double A[], const int ArowId[],
 	           const int AcolSum[], const int nnzB, const double B[],
		   const int BrowId[], const int BcolSum[], const double gamma);

int calcMismatch(const double tol, const int nsize, const int nnzA,
		   const double A[], const int ArowId[],
 	           const int AcolSum[], const int nnzB, const double B[],
		   const int BrowId[], const int BcolSum[], const double gamma);

int sparse_jac_v(N_Vector v, N_Vector Jv, realtype t, N_Vector y, N_Vector fy,
                 void *user_data, N_Vector tmp);
//void jacobianStats

#endif
