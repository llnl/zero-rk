#include "matrix_funcs.h"
#include "cv_param_sparse.h"
#include "ode_funcs.h"

//Prototype for at_plus_a borrowed from get_perm_c.c of SuperLU
void at_plus_a(const int n, const int nz, int *colptr, int *rowind, int *bnz, int **b_colptr, int **b_rowind);
int get_fillfactor(int n, int* colSum, int* rowIdx, double* data, int * perm_c,
#if SUPERLU_MAJOR_VERSION > 4
  GlobalLU_t Glu,
#endif
  superlu_options_t options);

#if defined SUNDIALS2
int jac_full_prec_setup(realtype t, N_Vector y, N_Vector fy,
			booleantype jok, booleantype *jcurPtr,
			realtype gamma, void *user_data,
			N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
#elif defined SUNDIALS3 || defined SUNDIALS4
int jac_full_prec_setup(realtype t, N_Vector y, N_Vector fy,
			booleantype jok, booleantype *jcurPtr,
                        realtype gamma, void *user_data)
{
#endif
  cv_param *cvp=(cv_param *)user_data;
  double startTime;

  static int numReUses = 0;
  int i,j;
  int flag;

  if(!jok) // if the Jacobian is not ok, process a new jacobian
  {
#if defined SUNDIALS2
    setupJacobianSparse(t,y,fy,cvp,tmp1,tmp2,tmp3);
#elif defined SUNDIALS3 || defined SUNDIALS4
    int nsize = cvp->sparseMtx->nSize;
    N_Vector tmp1, tmp2, tmp3;
    tmp1 = N_VNew_Serial(nsize);
    tmp2 = N_VNew_Serial(nsize);
    tmp3 = N_VNew_Serial(nsize);
    setupJacobianSparse(t,y,fy,cvp,tmp1,tmp2,tmp3);
    N_VDestroy_Serial(tmp1);
    N_VDestroy_Serial(tmp2);
    N_VDestroy_Serial(tmp3);
#endif

    int misMatch=0;
    if(!cvp->sparseMtx->isFirstFactor)
    {
      misMatch = calcMismatch(cvp->sparseMtx->offDiagThreshold,
                              cvp->sparseMtx->nSize,
                              cvp->sparseMtx->nNonZero,
                              cvp->sparseMtx->mtxData,
                              cvp->sparseMtx->mtxRowIdx,
                              cvp->sparseMtx->mtxColSum,
                              cvp->sparseMtx->reduceNNZ,
                              cvp->sparseMtx->reduceData,
                              cvp->sparseMtx->reduceRowIdx,
                              cvp->sparseMtx->reduceColSum,
                              gamma);
    }

    long int currNumErrTestFails;
    CVodeGetNumErrTestFails(cvp->cvodeMemPtr, &currNumErrTestFails);
    if(1)
    {
      numReUses = 0;
      cvp->prevNumErrTestFails = currNumErrTestFails;

      // copy only the off diagonal terms larger than the threshold
      if(cvp->sparseMtx->threshType == 1)
      {
        sparseOffDiagThreshCopyGamma(cvp->sparseMtx->offDiagThreshold,
                                     cvp->sparseMtx->nSize,
                                     cvp->sparseMtx->nNonZero,
                                     cvp->sparseMtx->mtxData,
                                     cvp->sparseMtx->mtxRowIdx,
                                     cvp->sparseMtx->mtxColSum,
                                     &(cvp->sparseMtx->reduceNNZ),
                                     cvp->sparseMtx->reduceData,
                                     cvp->sparseMtx->reduceRowIdx,
                                     cvp->sparseMtx->reduceColSum,
                                     gamma);
      } else if(cvp->sparseMtx->threshType == 2) {
        sparseOffDiagThreshCopyGamma_cnorm(cvp->sparseMtx->offDiagThreshold,
                                           cvp->sparseMtx->nSize,
                                           cvp->sparseMtx->nNonZero,
                                           cvp->sparseMtx->mtxData,
                                           cvp->sparseMtx->mtxRowIdx,
                                           cvp->sparseMtx->mtxColSum,
                                           &(cvp->sparseMtx->reduceNNZ),
                                           cvp->sparseMtx->reduceData,
                                           cvp->sparseMtx->reduceRowIdx,
                                           cvp->sparseMtx->reduceColSum,
                                           gamma);
      } else if(cvp->sparseMtx->threshType == 3) {
        sparseOffDiagThreshCopyGamma_rnorm(cvp->sparseMtx->offDiagThreshold,
                                           cvp->sparseMtx->nSize,
                                           cvp->sparseMtx->nNonZero,
                                           cvp->sparseMtx->mtxData,
                                           cvp->sparseMtx->mtxRowIdx,
                                           cvp->sparseMtx->mtxColSum,
                                           &(cvp->sparseMtx->reduceNNZ),
                                           cvp->sparseMtx->reduceData,
                                           cvp->sparseMtx->reduceRowIdx,
                                           cvp->sparseMtx->reduceColSum,
                                           gamma);
      } else if(cvp->sparseMtx->threshType == 4) {
        sparseOffDiagThreshCopyGamma_rcmin(cvp->sparseMtx->offDiagThreshold,
                                           cvp->sparseMtx->nSize,
                                           cvp->sparseMtx->nNonZero,
                                           cvp->sparseMtx->mtxData,
                                           cvp->sparseMtx->mtxRowIdx,
                                           cvp->sparseMtx->mtxColSum,
                                           &(cvp->sparseMtx->reduceNNZ),
                                           cvp->sparseMtx->reduceData,
                                           cvp->sparseMtx->reduceRowIdx,
                                           cvp->sparseMtx->reduceColSum,
                                           gamma);
      } else if(cvp->sparseMtx->threshType == 5) {
        sparseOffDiagMagCopyGamma(cvp->sparseMtx->offDiagThreshold,
                                  cvp->sparseMtx->nSize,
                                  cvp->sparseMtx->nNonZero,
                                  cvp->sparseMtx->mtxData,
                                  cvp->sparseMtx->mtxRowIdx,
                                  cvp->sparseMtx->mtxColSum,
                                  &(cvp->sparseMtx->reduceNNZ),
                                  cvp->sparseMtx->reduceData,
                                  cvp->sparseMtx->reduceRowIdx,
                                  cvp->sparseMtx->reduceColSum,
                                  gamma);
      } else {
        printf("Invalid threshType: %d.\n",cvp->sparseMtx->threshType);
        return -1;
      }
      cvp->sparseMtx->optionSLU.Fact=DOFACT;
      if(!cvp->sparseMtx->isFirstFactor)
      {
        // You must destroy the L and U before each new factorization, or
        // you will have a memory leak
        Destroy_SuperNode_Matrix(&(cvp->sparseMtx->Lslu));
        Destroy_CompCol_Matrix(&(cvp->sparseMtx->Uslu));
        //if(cvp->sparseMtx->ILU) cvp->sparseMtx->optionSLU.Fact=SamePattern;
        if(misMatch < cvp->sparseMtx->nSize/3) cvp->sparseMtx->optionSLU.Fact=SamePattern; // 2ma opt: 4000
      }
    } else {
      ++numReUses;
      sparseUpdateGamma(cvp->sparseMtx->nSize,
                        cvp->sparseMtx->nNonZero,
                        cvp->sparseMtx->mtxData,
                        cvp->sparseMtx->mtxRowIdx,
                        cvp->sparseMtx->mtxColSum,
                        cvp->sparseMtx->reduceNNZ,
                        cvp->sparseMtx->reduceData,
                        cvp->sparseMtx->reduceRowIdx,
                        cvp->sparseMtx->reduceColSum,
                        gamma);
      cvp->sparseMtx->optionSLU.Fact=SamePattern_SameRowPerm;
    }

    (*jcurPtr)=TRUE; //indicate that Jacobian data was recomputed
  }
  else //jok
  {
    (*jcurPtr)=FALSE; //indicate that Jacobian data was not recomputed
    if(cvp->sparseMtx->fakeUpdate) {return 0;}

    sparseUpdateGamma(cvp->sparseMtx->nSize,
                      cvp->sparseMtx->nNonZero,
                      cvp->sparseMtx->mtxData,
                      cvp->sparseMtx->mtxRowIdx,
                      cvp->sparseMtx->mtxColSum,
                      cvp->sparseMtx->reduceNNZ,
                      cvp->sparseMtx->reduceData,
                      cvp->sparseMtx->reduceRowIdx,
                      cvp->sparseMtx->reduceColSum,
                      gamma);
    cvp->sparseMtx->optionSLU.Fact=SamePattern_SameRowPerm;
  }

  int lwork=0;
  void *work=NULL;
  double rpg,rcond;
  double ferr[1],berr[1]; // length is the number of RHS

  ((NCformat *)cvp->sparseMtx->Mslu.Store)->nnz=cvp->sparseMtx->reduceNNZ;

  cvp->sparseMtx->Bslu.ncol=0; // in dlinsolx1.c example this is supposed to
                                 // indicate that a solution is not needed only
                                 // the factorization

  // Perform permutation
  startTime = zerork::getHighResolutionTime();
  if(cvp->sparseMtx->optionSLU.Fact == DOFACT)
  {
    // Need to generate column permutation
    if(cvp->sparseMtx->permutationType == 1)
    {
      get_perm_c(MMD_AT_PLUS_A, &(cvp->sparseMtx->Mslu), cvp->sparseMtx->colPermutation);
    }
    else if(cvp->sparseMtx->permutationType == 2) // Metis
    {
#ifdef CVIDT_USE_METIS
      int bnz;
      int *b_colptr, *b_rowind;
      int metisPermC[cvp->sparseMtx->nSize];
      at_plus_a(cvp->sparseMtx->nSize,
                cvp->sparseMtx->reduceNNZ,
                cvp->sparseMtx->reduceColSum,
                cvp->sparseMtx->reduceRowIdx,
                &bnz, &b_colptr, &b_rowind);
      METIS_NodeND(&cvp->sparseMtx->nSize,
                   b_colptr,b_rowind,NULL,
                   cvp->sparseMtx->metis_options,
                   metisPermC,
                   cvp->sparseMtx->colPermutation);
#else
      printf("ERROR : Metis not enabled.  Re-compile with CVIDT_USE_METIS defined.\n");
      exit(-1);
#endif
    } else {
      // should have read permutation from file perm_c.txt in setup of cvp
    }
  }
  ++(cvp->nColPerm);
  cvp->colPermTime += zerork::getHighResolutionTime() - startTime;

  startTime = zerork::getHighResolutionTime();
  if(!cvp->sparseMtx->ILU)
  {
    dgssvx(&(cvp->sparseMtx->optionSLU),&(cvp->sparseMtx->Mslu),
           cvp->sparseMtx->colPermutation,cvp->sparseMtx->rowPermutation,
           cvp->sparseMtx->colElimTree,cvp->sparseMtx->equed,cvp->sparseMtx->Rvec,
           cvp->sparseMtx->Cvec,&(cvp->sparseMtx->Lslu),&(cvp->sparseMtx->Uslu),
           work,lwork,&(cvp->sparseMtx->Bslu),&(cvp->sparseMtx->Xslu),&rpg,&rcond,
           ferr,berr,
#if SUPERLU_MAJOR_VERSION > 4
           &cvp->sparseMtx->Glu,
#endif
           &(cvp->sparseMtx->mem_usage),&(cvp->sparseMtx->statSLU), &flag);
  } else {
    dgsisx(&(cvp->sparseMtx->optionSLU),&(cvp->sparseMtx->Mslu),
           cvp->sparseMtx->colPermutation,cvp->sparseMtx->rowPermutation,
           cvp->sparseMtx->colElimTree,cvp->sparseMtx->equed,cvp->sparseMtx->Rvec,
           cvp->sparseMtx->Cvec,&(cvp->sparseMtx->Lslu),&(cvp->sparseMtx->Uslu),
           work,lwork,&(cvp->sparseMtx->Bslu),&(cvp->sparseMtx->Xslu),&rpg,&rcond,
#if SUPERLU_MAJOR_VERSION > 4
           &cvp->sparseMtx->Glu,
#endif
           &(cvp->sparseMtx->mem_usage),&(cvp->sparseMtx->statSLU),
           &flag);
  }
  ++(cvp->nJacFactor);

  cvp->sparseMtx->LUnnz = ((SCformat *)cvp->sparseMtx->Lslu.Store)->nnz + ((NCformat *)cvp->sparseMtx->Uslu.Store)->nnz;
  cvp->sparseMtx->fillFactor = cvp->sparseMtx->LUnnz/cvp->sparseMtx->reduceNNZ;

  cvp->jacFactorTime += zerork::getHighResolutionTime() - startTime;

  if(cvp->sparseMtx->isFirstFactor==1)
  {
    cvp->sparseMtx->isFirstFactor=0;
  }

  if(flag != 0)
  {return flag;}
  // flag > 0, singular matrix, zero diagonal at row,col = flag
  // flag < 0, illegal input
  // positive return values

  return 0;
}


#if defined SUNDIALS3 || defined SUNDIALS4
int jac_full_prec_setup_mix(realtype t, N_Vector y, N_Vector fy,
			booleantype jok, booleantype *jcurPtr,
                        realtype gamma, void *user_data)
{
#endif
  cv_param *cvp=(cv_param *)user_data;
  double* yptr = NV_DATA_S(y);
  double* fyptr = NV_DATA_S(fy);
  double startTime;

  static int numReUses = 0;
  int i,j,k;
  int flag;

  //Reactor 1
  if(!jok) // if the Jacobian is not ok, process a new jacobian
  {
    int nsize = cvp->sparseMtx->nSize;
    N_Vector tmp1, tmp2, tmp3;
    tmp1 = N_VNew_Serial(nsize);
    tmp2 = N_VNew_Serial(nsize);
    tmp3 = N_VNew_Serial(nsize);
    N_Vector y1;
    N_Vector fy1;
    y1 = N_VMake_Serial(nsize, yptr);
    fy1 = N_VMake_Serial(nsize, fyptr);
    cvp->ymix = &yptr[nsize];
    setupJacobianSparse(t, y1, fy1, cvp, tmp1, tmp2, tmp3);
    N_VDestroy_Serial(y1);
    N_VDestroy_Serial(fy1);

    //Merge chemical jacobians into mixing jacobian
    for(j=0; j<nsize; j++) // jth column
    {
      int kStart = cvp->sparseMtx->mtxColSum[j];
      int kEnd = cvp->sparseMtx->mtxColSum[j+1];
      for(k=kStart; k<kEnd; ++k)
      {
	int row = cvp->sparseMtx->mtxRowIdx[k];
        cvp->sparseMtx->mtxDataMix[k+j] = cvp->sparseMtx->mtxData[k];
	if(j == row) {
          cvp->sparseMtx->mtxDataMix[k+j] -= 1/cvp->tau_mix;
	}
      }
      cvp->sparseMtx->mtxDataMix[kEnd+j] = 1/cvp->tau_mix;
    }

    N_Vector y2;
    N_Vector fy2;
    y2 = N_VMake_Serial(nsize, &yptr[nsize]);
    fy2 = N_VMake_Serial(nsize, &fyptr[nsize]);
    cvp->ymix = yptr;
    setupJacobianSparse(t, y2, fy2, cvp, tmp1, tmp2, tmp3);
    N_VDestroy_Serial(y2);
    N_VDestroy_Serial(fy2);
    N_VDestroy_Serial(tmp1);
    N_VDestroy_Serial(tmp2);
    N_VDestroy_Serial(tmp3);

    //Merge chemical jacobians into mixing jacobian
    int kOffset = cvp->sparseMtx->nNonZero+nsize;
    for(j=0; j<nsize; j++) // jth column
    {
      int kStart = cvp->sparseMtx->mtxColSum[j];
      int kEnd = cvp->sparseMtx->mtxColSum[j+1];
      for(k=kStart; k<kEnd; ++k)
      {
	int row = cvp->sparseMtx->mtxRowIdx[k];
        cvp->sparseMtx->mtxDataMix[kOffset+k+j+1] =  cvp->sparseMtx->mtxData[k];
	if(j == row) {
          cvp->sparseMtx->mtxDataMix[kOffset+k+j+1] -= 1/cvp->tau_mix;
	}
      }
      cvp->sparseMtx->mtxDataMix[kOffset+kStart+j] = 1/cvp->tau_mix;
    }


    // copy only the off diagonal terms larger than the threshold
    if(cvp->sparseMtx->threshType == 1)
    {
      sparseOffDiagThreshCopyGamma(cvp->sparseMtx->offDiagThreshold,
                                   cvp->sparseMtx->nSizeMix,
                                   cvp->sparseMtx->nNonZeroMix,
                                   cvp->sparseMtx->mtxDataMix,
                                   cvp->sparseMtx->mtxRowIdxMix,
                                   cvp->sparseMtx->mtxColSumMix,
                                   &(cvp->sparseMtx->reduceNNZMix),
                                   cvp->sparseMtx->reduceDataMix,
                                   cvp->sparseMtx->reduceRowIdxMix,
                                   cvp->sparseMtx->reduceColSumMix,
                                   gamma);
    } else if(cvp->sparseMtx->threshType == 2) {
      sparseOffDiagThreshCopyGamma_cnorm(cvp->sparseMtx->offDiagThreshold,
                                         cvp->sparseMtx->nSizeMix,
                                         cvp->sparseMtx->nNonZeroMix,
                                         cvp->sparseMtx->mtxDataMix,
                                         cvp->sparseMtx->mtxRowIdxMix,
                                         cvp->sparseMtx->mtxColSumMix,
                                         &(cvp->sparseMtx->reduceNNZMix),
                                         cvp->sparseMtx->reduceDataMix,
                                         cvp->sparseMtx->reduceRowIdxMix,
                                         cvp->sparseMtx->reduceColSumMix,
                                         gamma);
    } else if(cvp->sparseMtx->threshType == 3) {
      sparseOffDiagThreshCopyGamma_rnorm(cvp->sparseMtx->offDiagThreshold,
                                         cvp->sparseMtx->nSizeMix,
                                         cvp->sparseMtx->nNonZeroMix,
                                         cvp->sparseMtx->mtxDataMix,
                                         cvp->sparseMtx->mtxRowIdxMix,
                                         cvp->sparseMtx->mtxColSumMix,
                                         &(cvp->sparseMtx->reduceNNZMix),
                                         cvp->sparseMtx->reduceDataMix,
                                         cvp->sparseMtx->reduceRowIdxMix,
                                         cvp->sparseMtx->reduceColSumMix,
                                         gamma);
    } else if(cvp->sparseMtx->threshType == 4) {
      sparseOffDiagThreshCopyGamma_rcmin(cvp->sparseMtx->offDiagThreshold,
                                         cvp->sparseMtx->nSizeMix,
                                         cvp->sparseMtx->nNonZeroMix,
                                         cvp->sparseMtx->mtxDataMix,
                                         cvp->sparseMtx->mtxRowIdxMix,
                                         cvp->sparseMtx->mtxColSumMix,
                                         &(cvp->sparseMtx->reduceNNZMix),
                                         cvp->sparseMtx->reduceDataMix,
                                         cvp->sparseMtx->reduceRowIdxMix,
                                         cvp->sparseMtx->reduceColSumMix,
                                         gamma);
    } else if(cvp->sparseMtx->threshType == 5) {
      sparseOffDiagMagCopyGamma(cvp->sparseMtx->offDiagThreshold,
                                cvp->sparseMtx->nSizeMix,
                                cvp->sparseMtx->nNonZeroMix,
                                cvp->sparseMtx->mtxDataMix,
                                cvp->sparseMtx->mtxRowIdxMix,
                                cvp->sparseMtx->mtxColSumMix,
                                &(cvp->sparseMtx->reduceNNZMix),
                                cvp->sparseMtx->reduceDataMix,
                                cvp->sparseMtx->reduceRowIdxMix,
                                cvp->sparseMtx->reduceColSumMix,
                                gamma);
    } else {
      printf("Invalid threshType: %d.\n",cvp->sparseMtx->threshType);
      return -1;
    }
    cvp->sparseMtx->optionSLU.Fact=DOFACT;
    if(!cvp->sparseMtx->isFirstFactor)
    {
      // You must destroy the L and U before each new factorization, or
      // you will have a memory leak
      Destroy_SuperNode_Matrix(&(cvp->sparseMtx->LsluMix));
      Destroy_CompCol_Matrix(&(cvp->sparseMtx->UsluMix));
      //if(cvp->sparseMtx->ILU) cvp->sparseMtx->optionSLU.Fact=SamePattern;
      //if(misMatch < cvp->sparseMtx->nSize/3) cvp->sparseMtx->optionSLU.Fact=SamePattern; // 2ma opt: 4000
    }

    (*jcurPtr)=TRUE; //indicate that Jacobian data was recomputed
  }
  else //jok
  {
    (*jcurPtr)=FALSE; //indicate that Jacobian data was not recomputed
    if(cvp->sparseMtx->fakeUpdate) {return 0;}

    sparseUpdateGamma(cvp->sparseMtx->nSizeMix,
                      cvp->sparseMtx->nNonZeroMix,
                      cvp->sparseMtx->mtxDataMix,
                      cvp->sparseMtx->mtxRowIdxMix,
                      cvp->sparseMtx->mtxColSumMix,
                      cvp->sparseMtx->reduceNNZMix,
                      cvp->sparseMtx->reduceDataMix,
                      cvp->sparseMtx->reduceRowIdxMix,
                      cvp->sparseMtx->reduceColSumMix,
                      gamma);
    cvp->sparseMtx->optionSLU.Fact=SamePattern_SameRowPerm;
  }

  int lwork=0;
  void *work=NULL;
  double rpg,rcond;
  double ferr[1],berr[1]; // length is the number of RHS

  ((NCformat *)cvp->sparseMtx->MsluMix.Store)->nnz=cvp->sparseMtx->reduceNNZMix;

  cvp->sparseMtx->BsluMix.ncol=0; // in dlinsolx1.c example this is supposed to
                                  // indicate that a solution is not needed only
                                  // the factorization

  // Perform permutation
  startTime = zerork::getHighResolutionTime();
  if(cvp->sparseMtx->optionSLU.Fact == DOFACT)
  {
    // Need to generate column permutation
    if(cvp->sparseMtx->permutationType == 1)
    {
      get_perm_c(MMD_AT_PLUS_A, &(cvp->sparseMtx->MsluMix), cvp->sparseMtx->colPermutationMix);
    } else {
      // should have read permutation from file perm_c.txt in setup of cvp
    }
  }
  ++(cvp->nColPerm);
  cvp->colPermTime += zerork::getHighResolutionTime() - startTime;

  startTime = zerork::getHighResolutionTime();
  dgssvx(&(cvp->sparseMtx->optionSLU),&(cvp->sparseMtx->MsluMix),
         cvp->sparseMtx->colPermutationMix,cvp->sparseMtx->rowPermutationMix,
         cvp->sparseMtx->colElimTreeMix,cvp->sparseMtx->equedMix,cvp->sparseMtx->RvecMix,
         cvp->sparseMtx->CvecMix,&(cvp->sparseMtx->LsluMix),&(cvp->sparseMtx->UsluMix),
         work,lwork,&(cvp->sparseMtx->BsluMix),&(cvp->sparseMtx->XsluMix),&rpg,&rcond,
         ferr,berr,
         &cvp->sparseMtx->Glu,
         &(cvp->sparseMtx->mem_usage),&(cvp->sparseMtx->statSLU), &flag);
  ++(cvp->nJacFactor);

  cvp->sparseMtx->LUnnz = ((SCformat *)cvp->sparseMtx->LsluMix.Store)->nnz + ((NCformat *)cvp->sparseMtx->UsluMix.Store)->nnz;
  cvp->sparseMtx->fillFactor = cvp->sparseMtx->LUnnz/cvp->sparseMtx->reduceNNZMix;

  cvp->jacFactorTime += zerork::getHighResolutionTime() - startTime;


  if(cvp->sparseMtx->isFirstFactor==1)
  {
    cvp->sparseMtx->isFirstFactor=0;
  }

  if(flag != 0)
  {return flag;}
  // flag > 0, singular matrix, zero diagonal at row,col = flag
  // flag < 0, illegal input
  // positive return values

  return 0;
}


// Solve Pz = r where P in this case is the full numerical Jacobian
//   M=I-gamma*J
#if defined SUNDIALS2
int jac_full_prec_solveV3(realtype t, N_Vector y, N_Vector fy,
		      N_Vector r, N_Vector z, realtype gamma,
		      realtype delta, int lr, void *user_data,
		      N_Vector tmp)
#elif defined SUNDIALS3 || defined SUNDIALS4
int jac_full_prec_solveV3(realtype t, N_Vector y, N_Vector fy,
                          N_Vector r, N_Vector z, realtype gamma,
                          realtype delta, int lr, void *user_data)
#endif
{
  cv_param *cvp=(cv_param *)user_data;
  int flag;
  int nsize=(cvp->nSpc)+1;
  int cpysize=nsize*sizeof(double);
  double *zptr=NV_DATA_S(z);
  double *rptr=NV_DATA_S(r);
  double startTime;

  int lwork=0;
  void *work=NULL;
  double rpg,rcond;
  double ferr[1],berr[1]; // length is the number of RHS

  cvp->sparseMtx->optionSLU.Fact=FACTORED;
  cvp->sparseMtx->Bslu.ncol=1; // in dlinsolx1.c example this is reset to the number
                               // of right hand sides

  startTime = zerork::getHighResolutionTime();
  // copy the data to the sparse matrix RHS
  memcpy(cvp->sparseMtx->vecRHS,rptr,cpysize);

  //backsolve with expert driver function
  dgssvx(&(cvp->sparseMtx->optionSLU),&(cvp->sparseMtx->Mslu),
	 cvp->sparseMtx->colPermutation,cvp->sparseMtx->rowPermutation,
	 cvp->sparseMtx->colElimTree,cvp->sparseMtx->equed,cvp->sparseMtx->Rvec,
	 cvp->sparseMtx->Cvec,&(cvp->sparseMtx->Lslu),&(cvp->sparseMtx->Uslu),
	 work,lwork,&(cvp->sparseMtx->Bslu),&(cvp->sparseMtx->Xslu),&rpg,&rcond,
	 ferr,berr,
#if SUPERLU_MAJOR_VERSION > 4
         &cvp->sparseMtx->Glu,
#endif
         &(cvp->sparseMtx->mem_usage),&(cvp->sparseMtx->statSLU), &flag);

  //book keeping
  ++(cvp->nBackSolve);
  cvp->backsolveTime += zerork::getHighResolutionTime() - startTime;

  //copy solution to output
  memcpy(zptr,cvp->sparseMtx->vecSoln,cpysize);

  return flag;
}

#if defined SUNDIALS3 || defined SUNDIALS4
int jac_full_prec_solveV3_mix(realtype t, N_Vector y, N_Vector fy,
                              N_Vector r, N_Vector z, realtype gamma,
                              realtype delta, int lr, void *user_data)
#endif
{
  cv_param *cvp=(cv_param *)user_data;
  int flag;
  int nsize=(cvp->nSpc)+1;
  int cpysize=2*nsize*sizeof(double);
  double *zptr=NV_DATA_S(z);
  double *rptr=NV_DATA_S(r);
  double startTime;

  int lwork=0;
  void *work=NULL;
  double rpg,rcond;
  double ferr[1],berr[1]; // length is the number of RHS

  cvp->sparseMtx->optionSLU.Fact=FACTORED;
  cvp->sparseMtx->BsluMix.ncol=1;  // in dlinsolx1.c example this is reset to the number
                                   // of right hand sides

  startTime = zerork::getHighResolutionTime();
  // copy the data to the sparse matrix RHS
  memcpy(cvp->sparseMtx->vecRHS,rptr,cpysize);

  //backsolve with expert driver function
  dgssvx(&(cvp->sparseMtx->optionSLU),&(cvp->sparseMtx->MsluMix),
	 cvp->sparseMtx->colPermutationMix,cvp->sparseMtx->rowPermutationMix,
	 cvp->sparseMtx->colElimTreeMix,cvp->sparseMtx->equedMix,cvp->sparseMtx->RvecMix,
	 cvp->sparseMtx->CvecMix,&(cvp->sparseMtx->LsluMix),&(cvp->sparseMtx->UsluMix),
	 work,lwork,&(cvp->sparseMtx->BsluMix),&(cvp->sparseMtx->XsluMix),&rpg,&rcond,
	 ferr,berr,
         &cvp->sparseMtx->Glu,
         &(cvp->sparseMtx->mem_usage),&(cvp->sparseMtx->statSLU), &flag);

  //book keeping
  ++(cvp->nBackSolve);
  cvp->backsolveTime += zerork::getHighResolutionTime() - startTime;

  //copy solution to output
  memcpy(zptr,cvp->sparseMtx->vecSoln,cpysize);

  return flag;
}


void setupJacobianSparse(realtype t,N_Vector y,N_Vector fy,cv_param* cvp,N_Vector tmp1,N_Vector tmp2,N_Vector tmp3)
{
  double dTemp,RuTemp,multFact,startTime;
  int nSpc;
  int rowId,sparseId,sparseId2,nElem;
  int j,k,N;
  double *sMptr;
  double *tmp1_ptr = NV_DATA_S(tmp1);
  double *tmp2_ptr = NV_DATA_S(tmp2);
  double *tmp3_ptr = NV_DATA_S(tmp3);
  void* user_data = (void*)cvp;

  startTime = zerork::getHighResolutionTime();

  nSpc=cvp->mech->getNumSpecies();
  N=nSpc+1;
  RuTemp=(cvp->mech->getGasConstant())*NV_Ith_S(y,nSpc);

  // set tmp1 to the strictly positive mass fraction array
  // set tmp2 to 1/C where C is the concentration for the strictly positive
  // mass fraction array
  for(j=0; j<nSpc; j++)
  {
    NV_Ith_S(tmp1,j)=NV_Ith_S(y,j);
    if(NV_Ith_S(tmp1,j) < cvp->minMassFrac)
      NV_Ith_S(tmp1,j)=cvp->minMassFrac;
    NV_Ith_S(tmp2,j)=cvp->molWt[j]/NV_Ith_S(tmp1,j)*cvp->invDens;
  }

  NV_Ith_S(tmp1,nSpc)=NV_Ith_S(y,nSpc);

  // calculate the reaction info for the strictly positive case
  const_vol_wsr(t,tmp1,tmp3,user_data);

  // SPARSE:
  // the entire sparse matrix array will be set to zero upon entry
  calcReaction_Jsparse(cvp->sparseMtx,NV_DATA_S(tmp2),cvp->fwdROP);

  // At this point sMptr stores d(wdot[k])/dC[j] ignoring the contribution
  // of perturbations in the third body species
  sMptr=cvp->sparseMtx->mtxData;

  // ---------------------------------------------------------------------
  // compute d(Tdot)/dy[j]

  // SPARSE:
  // step 1: compute matrix vector product (d(wdot[k])/dC[j])*E[k]
  for(j=0; j<nSpc; j++) // column number
  {
    sparseId=cvp->sparseMtx->lastRowIdx[j];
    sMptr[sparseId]=0.0;
    nElem=cvp->sparseMtx->mtxColSum[j+1]-cvp->sparseMtx->mtxColSum[j];
    nElem--; // decrement the element by one since the last row is
    // marked
    sparseId2=cvp->sparseMtx->mtxColSum[j];
    for(k=0; k<nElem; k++)
    {
      rowId=cvp->sparseMtx->mtxRowIdx[sparseId2];
      sMptr[sparseId]+=sMptr[sparseId2]*(cvp->Energy[rowId]);
      sparseId2++;
    }
    sMptr[sparseId]*=RuTemp;
  }

  // step 2: add the Tdot*Cv[j] term to d(Tdot)/dy[j]
  for(j=0; j<nSpc; j++)
  {
    sparseId=cvp->sparseMtx->lastRowIdx[j];
    sMptr[sparseId]+=(cvp->dTemp_dt)*(cvp->CpMass[j]*cvp->molWt[j]);
  }

  // step 3: divide by -1/(molWt[j]*meanCpMass)
  multFact=-1.0/cvp->meanCpMass;
  for(j=0; j<nSpc; j++)
  {
    sparseId=cvp->sparseMtx->lastRowIdx[j];
    sMptr[sparseId]*=cvp->invMolWt[j]*multFact;
  }

  // At this point Mmtx stores d(Tdot[k])/dy[j] ignoring the contribution
  // of perturbations in the third body species
  sparseId=0;
  for(j=0; j<nSpc; j++) // jth column
  {
    nElem=cvp->sparseMtx->mtxColSum[j+1]-cvp->sparseMtx->mtxColSum[j];
    for(k=0; k<nElem; k++)
    {
      rowId=cvp->sparseMtx->mtxRowIdx[sparseId];
      if(rowId<nSpc) // rowId==nSpc is the last row
      {
        cvp->sparseMtx->mtxData[sparseId]*=
          (cvp->molWt[rowId])*(cvp->invMolWt[j]);
      }
      sparseId++;
    }
  }

  // variable volume correction for constant pressure
  // if const press only
  double molWtMix = cvp->mech->getMolWtMixFromY(tmp1_ptr);
  for(j=0; j<nSpc; j++) { // column number
    sparseId=cvp->sparseMtx->mtxColSum[j];
    nElem=cvp->sparseMtx->mtxColSum[j+1]-cvp->sparseMtx->mtxColSum[j];
    for(k=0; k<nElem; k++) {
      rowId=cvp->sparseMtx->mtxRowIdx[sparseId];
      sMptr[sparseId] += molWtMix*cvp->invMolWt[j]*tmp3_ptr[rowId];
      sparseId++;
    }
  }

  // ---------------------------------------------------------------------
  // calculate d(ydot[k])/dT

  // step 1: perturb the temperature
  dTemp=NV_Ith_S(y,nSpc)*(cvp->sqrtUnitRnd);
  NV_Ith_S(tmp1,nSpc)+=dTemp;
  dTemp=NV_Ith_S(tmp1,nSpc)-NV_Ith_S(y,nSpc);
  multFact=1.0/dTemp;

  for(j=0; j<nSpc; j++)
  {NV_Ith_S(tmp1,j)=NV_Ith_S(y,j);}

  // step 2: calculate ydot at Temp+dTemp
  const_vol_wsr(t,tmp1,tmp3,user_data);

  sMptr=cvp->sparseMtx->mtxData;
  // step 3: approximate d(ydot[k])/dT with finite difference
  sparseId=cvp->sparseMtx->mtxColSum[nSpc];
  for(k=0; k<N; k++)
  {
    sMptr[sparseId]=(NV_Ith_S(tmp3,k)-NV_Ith_S(fy,k))*multFact;
    sparseId++;
  }

  cvp->jacSetupTime += zerork::getHighResolutionTime() - startTime;
  ++(cvp->nJacSetup);
}



void sparseOffDiagThreshCopy(const double tol, const int nsize, const int nnzA,
			     const double A[], const int ArowId[],
			     const int AcolSum[], int *nnzB, double B[],
			     int BrowId[], int BcolSum[])
{
  int j,k,nelem;
  int AsparseId=0;
  int BsparseId=0;

  AsparseId=BsparseId=0;
  BcolSum[0]=0;
  for(j=0; j<nsize; j++) //
    {
      nelem=AcolSum[j+1]-AcolSum[j];
      BcolSum[j+1]=BcolSum[j];
      for(k=0; k<nelem; k++)
	{
	  if(ArowId[AsparseId]==j || fabs(A[AsparseId]) > tol)
	    { // directly copy diagonal and any off-diagonal terms larger than
	      // tol
	      B[BsparseId]=A[AsparseId];
	      BrowId[BsparseId]=ArowId[AsparseId];
	      BcolSum[j+1]++; // increment the column sum
	      BsparseId++;    // increment B sparse index
	    }
	  AsparseId++; // increment the A sparse index
	}
    }
  (*nnzB)=BcolSum[nsize];
}

void sparseOffDiagThreshCopyGamma(const double tol, const int nsize, const int nnzA,
			     const double A[], const int ArowId[],
			     const int AcolSum[], int *nnzB, double B[],
			     int BrowId[], int BcolSum[], const double gamma)
{
  int j,k,nelem;
  int AsparseId=0;
  int BsparseId=0;

  AsparseId=BsparseId=0;
  BcolSum[0]=0;
  for(j=0; j<nsize; j++) //
    {
      nelem=AcolSum[j+1]-AcolSum[j];
      BcolSum[j+1]=BcolSum[j];
      for(k=0; k<nelem; k++)
	{
          bool diag = ArowId[AsparseId] == j;
          double currVal = -gamma*A[AsparseId];
          if (diag) currVal += 1.0;
	  if(diag || fabs(currVal) > tol)
	    { // directly copy diagonal and any off-diagonal terms larger than
	      // tol
	      B[BsparseId]=currVal;
	      BrowId[BsparseId]=ArowId[AsparseId];
	      BcolSum[j+1]++; // increment the column sum
	      BsparseId++;    // increment B sparse index
	    }
	  AsparseId++; // increment the A sparse index
	}
    }
  (*nnzB)=BcolSum[nsize];
}

void sparseOffDiagThreshCopy_cnorm(const double tol, const int nsize, const int nnzA,
			     const double A[], const int ArowId[],
			     const int AcolSum[], int *nnzB, double B[],
			     int BrowId[], int BcolSum[])
{
  int j,k,nelem;
  int AsparseId=0;
  int BsparseId=0;
  double *Adiag;

  Adiag=(double *)malloc(sizeof(double)*nsize);
  AsparseId=0;

  for(j=0; j<nsize; j++) // column number
    {
      nelem=AcolSum[j+1]-AcolSum[j];
      k=0;
      AsparseId=AcolSum[j];
      while(k<nelem)
	{
	  if(ArowId[AsparseId]==j)
	    {Adiag[j]=tol*fabs(A[AsparseId]); k=nelem;}
	  ++k;
	  ++AsparseId;
	}
    }

  AsparseId=BsparseId=0;
  BcolSum[0]=0;
  for(j=0; j<nsize; j++) // column number
    {
      nelem=AcolSum[j+1]-AcolSum[j];
      BcolSum[j+1]=BcolSum[j];
      for(k=0; k<nelem; k++)
	{
	  if(ArowId[AsparseId]==j || fabs(A[AsparseId]) > Adiag[j])
	    { // directly copy diagonal and any off-diagonal terms larger than
	      // tol
	      B[BsparseId]=A[AsparseId];
	      BrowId[BsparseId]=ArowId[AsparseId];
	      BcolSum[j+1]++; // increment the column sum
	      BsparseId++;    // increment B sparse index
	    }
	  AsparseId++; // increment the A sparse index
	}
    }
  (*nnzB)=BcolSum[nsize];
  free(Adiag);
}

void sparseOffDiagThreshCopyGamma_cnorm(const double tol, const int nsize, const int nnzA,
			     const double A[], const int ArowId[],
			     const int AcolSum[], int *nnzB, double B[],
			     int BrowId[], int BcolSum[], const double gamma)
{
  int j,k,nelem;
  int AsparseId=0;
  int BsparseId=0;
  double *Adiag;

  Adiag=(double *)malloc(sizeof(double)*nsize);
  AsparseId=0;

  for(j=0; j<nsize; j++) // column number
    {
      nelem=AcolSum[j+1]-AcolSum[j];
      k=0;
      AsparseId=AcolSum[j];
      while(k<nelem)
	{
	  if(ArowId[AsparseId]==j)
	    {Adiag[j]=tol*fabs(1.0-gamma*A[AsparseId]); k=nelem;}
	  ++k;
	  ++AsparseId;
	}
    }

  AsparseId=BsparseId=0;
  BcolSum[0]=0;
  for(j=0; j<nsize; j++) // column number
    {
      nelem=AcolSum[j+1]-AcolSum[j];
      BcolSum[j+1]=BcolSum[j];
      for(k=0; k<nelem; k++)
	{
          bool diag = ArowId[AsparseId] == j;
          double currVal = -gamma*A[AsparseId];
          if (diag) currVal += 1.0;
	  if(diag || fabs(currVal) > Adiag[j])
	    { // directly copy diagonal and any off-diagonal terms larger than
	      // tol
	      B[BsparseId]=currVal;
	      BrowId[BsparseId]=ArowId[AsparseId];
	      BcolSum[j+1]++; // increment the column sum
	      BsparseId++;    // increment B sparse index
	    }
	  AsparseId++; // increment the A sparse index
	}
    }
  (*nnzB)=BcolSum[nsize];
  free(Adiag);
}


void sparseOffDiagThreshCopy_rnorm(const double tol, const int nsize, const int nnzA,
			     const double A[], const int ArowId[],
			     const int AcolSum[], int *nnzB, double B[],
			     int BrowId[], int BcolSum[])
{
  int j,k,nelem;
  int AsparseId=0;
  int BsparseId=0;
  double *Adiag;

  Adiag=(double *)malloc(sizeof(double)*nsize);
  AsparseId=0;

  for(j=0; j<nsize; j++) // column number
    {
      nelem=AcolSum[j+1]-AcolSum[j];
      k=0;
      AsparseId=AcolSum[j];
      while(k<nelem)
	{
	  if(ArowId[AsparseId]==j)
	    {Adiag[j]=tol*fabs(A[AsparseId]); k=nelem;}
	  ++k;
	  ++AsparseId;
	}
    }

  AsparseId=BsparseId=0;
  BcolSum[0]=0;
  for(j=0; j<nsize; j++) // column number
    {
      nelem=AcolSum[j+1]-AcolSum[j];
      BcolSum[j+1]=BcolSum[j];
      for(k=0; k<nelem; k++)
	{
	  if(ArowId[AsparseId]==j || fabs(A[AsparseId]) > Adiag[ArowId[AsparseId]])
	    { // directly copy diagonal and any off-diagonal terms larger than
	      // tol
	      B[BsparseId]=A[AsparseId];
	      BrowId[BsparseId]=ArowId[AsparseId];
	      BcolSum[j+1]++; // increment the column sum
	      BsparseId++;    // increment B sparse index
	    }
	  AsparseId++; // increment the A sparse index
	}
    }
  (*nnzB)=BcolSum[nsize];
  free(Adiag);
}


void sparseOffDiagThreshCopyGamma_rnorm(const double tol, const int nsize, const int nnzA,
			     const double A[], const int ArowId[],
			     const int AcolSum[], int *nnzB, double B[],
			     int BrowId[], int BcolSum[], const double gamma)
{
  int j,k,nelem;
  int AsparseId=0;
  int BsparseId=0;
  double *Adiag;

  Adiag=(double *)malloc(sizeof(double)*nsize);
  AsparseId=0;

  for(j=0; j<nsize; j++) // column number
    {
      nelem=AcolSum[j+1]-AcolSum[j];
      k=0;
      AsparseId=AcolSum[j];
      while(k<nelem)
	{
	  if(ArowId[AsparseId]==j)
	    {Adiag[j]=tol*fabs(1.0-gamma*A[AsparseId]); k=nelem;}
	  ++k;
	  ++AsparseId;
	}
    }

  AsparseId=BsparseId=0;
  BcolSum[0]=0;
  for(j=0; j<nsize; j++) // column number
    {
      nelem=AcolSum[j+1]-AcolSum[j];
      BcolSum[j+1]=BcolSum[j];
      for(k=0; k<nelem; k++)
	{
          bool diag = ArowId[AsparseId] == j;
          double currVal = -gamma*A[AsparseId];
          if (diag) currVal += 1.0;
	  if(diag || fabs(currVal) > Adiag[ArowId[AsparseId]])
	    { // directly copy diagonal and any off-diagonal terms larger than
	      // tol
	      B[BsparseId]=currVal;
	      BrowId[BsparseId]=ArowId[AsparseId];
	      BcolSum[j+1]++; // increment the column sum
	      BsparseId++;    // increment B sparse index
	    }
	  AsparseId++; // increment the A sparse index
	}
    }
  (*nnzB)=BcolSum[nsize];
  free(Adiag);
}

void sparseOffDiagThreshCopy_rcmin(const double tol, const int nsize, const int nnzA,
			     const double A[], const int ArowId[],
			     const int AcolSum[], int *nnzB, double B[],
			     int BrowId[], int BcolSum[])
{
  int j,k,nelem;
  int AsparseId=0;
  int BsparseId=0;
  double *Adiag;

  Adiag=(double *)malloc(sizeof(double)*nsize);
  AsparseId=0;

  for(j=0; j<nsize; j++) // column number
    {
      nelem=AcolSum[j+1]-AcolSum[j];
      k=0;
      AsparseId=AcolSum[j];
      while(k<nelem)
	{
	  if(ArowId[AsparseId]==j)
	    {Adiag[j]=tol*fabs(A[AsparseId]); k=nelem;}
	  ++k;
	  ++AsparseId;
	}
    }

  AsparseId=BsparseId=0;
  BcolSum[0]=0;
  for(j=0; j<nsize; j++) // column number
    {
      nelem=AcolSum[j+1]-AcolSum[j];
      BcolSum[j+1]=BcolSum[j];
      for(k=0; k<nelem; k++)
	{
	  if(ArowId[AsparseId]==j ||
	     fabs(A[AsparseId]) > Adiag[ArowId[AsparseId]] ||
	     fabs(A[AsparseId]) > Adiag[j])
	    { // directly copy diagonal and any off-diagonal terms larger than
	      // tol
	      B[BsparseId]=A[AsparseId];
	      BrowId[BsparseId]=ArowId[AsparseId];
	      BcolSum[j+1]++; // increment the column sum
	      BsparseId++;    // increment B sparse index
	    }
	  AsparseId++; // increment the A sparse index
	}
    }
  (*nnzB)=BcolSum[nsize];
  free(Adiag);
}

void sparseOffDiagThreshCopyGamma_rcmin(const double tol, const int nsize, const int nnzA,
			     const double A[], const int ArowId[],
			     const int AcolSum[], int *nnzB, double B[],
			     int BrowId[], int BcolSum[], const double gamma)
{
  int j,k,nelem;
  int AsparseId=0;
  int BsparseId=0;
  double *Adiag;

  Adiag=(double *)malloc(sizeof(double)*nsize);
  AsparseId=0;

  for(j=0; j<nsize; j++) // column number
    {
      nelem=AcolSum[j+1]-AcolSum[j];
      k=0;
      AsparseId=AcolSum[j];
      while(k<nelem)
	{
	  if(ArowId[AsparseId]==j)
	    {Adiag[j]=tol*fabs(1.0-gamma*A[AsparseId]); k=nelem;}
	  ++k;
	  ++AsparseId;
	}
    }

  AsparseId=BsparseId=0;
  BcolSum[0]=0;
  for(j=0; j<nsize; j++) // column number
    {
      nelem=AcolSum[j+1]-AcolSum[j];
      BcolSum[j+1]=BcolSum[j];
      for(k=0; k<nelem; k++)
	{
          bool diag = ArowId[AsparseId] == j;
          double currVal = -gamma*A[AsparseId];
          if (diag) currVal += 1.0;
	  if(diag ||
	     fabs(currVal) > Adiag[ArowId[AsparseId]] ||
	     fabs(currVal) > Adiag[j])
	    { // directly copy diagonal and any off-diagonal terms larger than
	      // tol
	      B[BsparseId]=currVal;
	      BrowId[BsparseId]=ArowId[AsparseId];
	      BcolSum[j+1]++; // increment the column sum
	      BsparseId++;    // increment B sparse index
	    }
	  AsparseId++; // increment the A sparse index
	}
    }
  (*nnzB)=BcolSum[nsize];
  free(Adiag);
}


int compare_double(const void *a, const void *b)
{
  double aval = *((double *)a);
  double bval = *((double *)b);
  return ((aval > bval) ? -1 : 1);
}

void sparseOffDiagMagCopy(const double frac, const int nsize, const int nnzA,
			  const double A[], const int ArowId[],
			  const int AcolSum[], int *nnzB, double B[],
			  int BrowId[], int BcolSum[])
{
  int j,k,nelem,nkeep;
  int AsparseId=0;
  int BsparseId=0;
  double thresh;
  double *Asort;


  Asort=(double *)malloc(sizeof(double)*nnzA);

  for(j=0; j<nnzA; j++)
    {Asort[j]=fabs(A[j]);}

  qsort(Asort,nnzA,sizeof(double),compare_double);
//  sort(Asort,Asort+nnzA);
  nkeep=(int)(frac*(double)nnzA+0.5);
  thresh=Asort[nkeep-1];

  //printf("# threshold value = %.18g\n", thresh);
  //exit(-1);

  AsparseId=BsparseId=0;
  BcolSum[0]=0;
  for(j=0; j<nsize; j++) // column number
    {
      nelem=AcolSum[j+1]-AcolSum[j];
      BcolSum[j+1]=BcolSum[j];
      for(k=0; k<nelem; k++)
	{
	  if(ArowId[AsparseId]==j || fabs(A[AsparseId]) > thresh)
	    { // directly copy diagonal and any off-diagonal terms larger than
	      // tol
	      B[BsparseId]=A[AsparseId];
	      BrowId[BsparseId]=ArowId[AsparseId];
	      BcolSum[j+1]++; // increment the column sum
	      BsparseId++;    // increment B sparse index
	    }
	  AsparseId++; // increment the A sparse index
	}
    }
  (*nnzB)=BcolSum[nsize];
  free(Asort);
}


void sparseOffDiagMagCopyGamma(const double frac, const int nsize, const int nnzA,
			  const double A[], const int ArowId[],
			  const int AcolSum[], int *nnzB, double B[],
			  int BrowId[], int BcolSum[], const double gamma)
{
  int j,k,nelem,nkeep;
  int AsparseId=0;
  int BsparseId=0;
  double thresh;
  double *Asort;

  Asort=(double *)malloc(sizeof(double)*nnzA);

  for(j=0; j<nsize; j++) // column number
    {
      nelem=AcolSum[j+1]-AcolSum[j];
      AsparseId=AcolSum[j];
      for(k=0; k<nelem; ++k)
	{
           bool diag = ArowId[AsparseId] == j;
           double currVal = -gamma*A[AsparseId];
           if (diag) currVal += 1.0;
	   Asort[AsparseId]=fabs(currVal);
	   ++AsparseId;
	}
    }

  qsort(Asort,nnzA,sizeof(double),compare_double);
//  sort(Asort,Asort+nnzA);
  nkeep=(int)(frac*(double)nnzA+0.5);
  thresh=Asort[nkeep-1];

  AsparseId=BsparseId=0;
  BcolSum[0]=0;
  for(j=0; j<nsize; j++) // column number
    {
      nelem=AcolSum[j+1]-AcolSum[j];
      BcolSum[j+1]=BcolSum[j];
      for(k=0; k<nelem; k++)
	{
          bool diag = ArowId[AsparseId] == j;
          double currVal = -gamma*A[AsparseId];
          if (diag) currVal += 1.0;
	  if(diag || fabs(currVal) > thresh)
	    { // directly copy diagonal and any off-diagonal terms larger than
	      // tol
	      B[BsparseId]=currVal;
	      BrowId[BsparseId]=ArowId[AsparseId];
	      BcolSum[j+1]++; // increment the column sum
	      BsparseId++;    // increment B sparse index
	    }
	  AsparseId++; // increment the A sparse index
	}
    }
  (*nnzB)=BcolSum[nsize];
  free(Asort);
}


void sparseUpdateGamma(const int nsize, const int nnzA,
			     const double A[], const int ArowId[],
			     const int AcolSum[], int nnzB, double B[],
			     int BrowId[], int BcolSum[], const double gamma)
{
  int k,brow,bcol,arow,BsparseId,AsparseId,nElemAcol,lastAmatch;

  bcol = 0;
  lastAmatch = 0;
  nElemAcol = AcolSum[bcol+1] - AcolSum[bcol];
  for(BsparseId=0; BsparseId < nnzB; ++BsparseId)
    {
       brow = BrowId[BsparseId];
       if(BsparseId >= BcolSum[bcol+1])
         {
           ++bcol;
           lastAmatch = 0;
           nElemAcol = AcolSum[bcol+1] - AcolSum[bcol];
         }
       for(k=lastAmatch; k<nElemAcol; ++k)
        {
          AsparseId = AcolSum[bcol] + k;
          arow = ArowId[AsparseId];
          if(brow == arow)
            {
              bool diag = brow == bcol;
	      B[BsparseId] = -gamma*A[AsparseId];
              if (diag) B[BsparseId] += 1.0;
              lastAmatch = k;
              break;
            }
         }
     }
}

bool reThreshCheck(const double tol, const int nsize, const int nnzA,
		   const double A[], const int ArowId[],
 	           const int AcolSum[], const int nnzB, const double B[],
		   const int BrowId[], const int BcolSum[], const double gamma)
{
  int k,brow,bcol,arow,BsparseId,AsparseId,nElemAcol,lastAmatch;
  double maxDroppedVal = 0.0;
  int droppedVals = 0;
  int retainedVals = 0;

  bcol = 0;
  lastAmatch = 0;
  nElemAcol = AcolSum[bcol+1] - AcolSum[bcol];
  for(BsparseId=0; BsparseId < nnzB; ++BsparseId)
    {
       brow = BrowId[BsparseId];
       if(BsparseId >= BcolSum[bcol+1])
         {
           ++bcol;
           lastAmatch = 0;
           nElemAcol = AcolSum[bcol+1] - AcolSum[bcol];
         }
       for(k=lastAmatch; k<nElemAcol; ++k)
        {
          AsparseId = AcolSum[bcol] + k;
          arow = ArowId[AsparseId];
          double currVal = fabs(gamma*A[AsparseId]);
          if(brow != arow)
            {
	      maxDroppedVal = max(currVal,maxDroppedVal);
              if(currVal >= tol) ++droppedVals;
            } else {
              if(currVal <= tol) ++retainedVals;
              lastAmatch = k;
              if(BsparseId < BcolSum[bcol+1]) break; //not last val in B[:,j]
            }
         }
     }

  printf("droppedVals: %d, maxDroppedVal: %8.4e, retainedVals: %d, gamma: %8.4e\n",
          droppedVals,     maxDroppedVal,        retainedVals,     gamma);
//  if(droppedVals > nnzA/3 || maxDroppedVal > 25.0*tol || retainedVals > 500)
//  if(droppedVals > nnzA/3)
    {
      return true;
    }
  return false;
}


int calcMismatch(const double tol, const int nsize, const int nnzA,
		   const double A[], const int ArowId[],
 	           const int AcolSum[], const int nnzB, const double B[],
		   const int BrowId[], const int BcolSum[], const double gamma)
{
  int j,k,nelem,nelemOld,mismatch;
  int AsparseId=0;
  int BsparseId=0;
//  int newB[nnzA];
  int newBcolSum[nsize+1];
//  int newBrowId[nnzA];

  AsparseId=BsparseId=0;
  newBcolSum[0]=0;
  for(j=0; j<nsize; j++) //
    {
      nelem=AcolSum[j+1]-AcolSum[j];
      newBcolSum[j+1]=newBcolSum[j];
      for(k=0; k<nelem; k++)
	{
          bool diag = ArowId[AsparseId] == j;
          double currVal = -gamma*A[AsparseId];
          if (diag) currVal += 1.0;
	  if(diag || fabs(currVal) > tol)
	    { // directly copy diagonal and any off-diagonal terms larger than
	      // tol
//	      newB[BsparseId]=currVal;
//	      newBrowId[BsparseId]=ArowId[AsparseId];
	      newBcolSum[j+1]++; // increment the column sum
	      BsparseId++;    // increment B sparse index
	    }
	  AsparseId++; // increment the A sparse index
	}
    }

  mismatch = 0;
  //generated new structure for reduced matrix on this step
  for(j = 0; j<nsize; ++j)
  {
     nelemOld = BcolSum[j+1] - BcolSum[j];
     nelem = newBcolSum[j+1] - newBcolSum[j];
     mismatch += abs(nelem-nelemOld);
  }
  return mismatch;
}

//Copied from get_perm_c.c in SuperLU4.2  because it isn't exposed in the api
void
at_plus_a(
	  const int n,      /* number of columns in matrix A. */
	  const int nz,     /* number of nonzeros in matrix A */
	  int *colptr,      /* column pointer of size n+1 for matrix A. */
	  int *rowind,      /* row indices of size nz for matrix A. */
	  int *bnz,         /* out - on exit, returns the actual number of
                               nonzeros in matrix A'*A. */
	  int **b_colptr,   /* out - size n+1 */
	  int **b_rowind    /* out - size *bnz */
	  )
{
    register int i, j, k, col, num_nz;
    int *t_colptr, *t_rowind; /* a column oriented form of T = A' */
    int *marker;

    if ( !(marker = (int*) SUPERLU_MALLOC( n * sizeof(int)) ) )
	ABORT("SUPERLU_MALLOC fails for marker[]");
    if ( !(t_colptr = (int*) SUPERLU_MALLOC( (n+1) * sizeof(int)) ) )
	ABORT("SUPERLU_MALLOC fails for t_colptr[]");
    if ( !(t_rowind = (int*) SUPERLU_MALLOC( nz * sizeof(int)) ) )
	ABORT("SUPERLU_MALLOC fails t_rowind[]");


    /* Get counts of each column of T, and set up column pointers */
    for (i = 0; i < n; ++i) marker[i] = 0;
    for (j = 0; j < n; ++j) {
	for (i = colptr[j]; i < colptr[j+1]; ++i)
	    ++marker[rowind[i]];
    }
    t_colptr[0] = 0;
    for (i = 0; i < n; ++i) {
	t_colptr[i+1] = t_colptr[i] + marker[i];
	marker[i] = t_colptr[i];
    }

    /* Transpose the matrix from A to T */
    for (j = 0; j < n; ++j)
	for (i = colptr[j]; i < colptr[j+1]; ++i) {
	    col = rowind[i];
	    t_rowind[marker[col]] = j;
	    ++marker[col];
	}


    /* ----------------------------------------------------------------
       compute B = A + T, where column j of B is:

       Struct (B_*j) = Struct (A_*k) UNION Struct (T_*k)

       do not include the diagonal entry
       ---------------------------------------------------------------- */

    /* Zero the diagonal flag */
    for (i = 0; i < n; ++i) marker[i] = -1;

    /* First pass determines number of nonzeros in B */
    num_nz = 0;
    for (j = 0; j < n; ++j) {
	/* Flag the diagonal so it's not included in the B matrix */
	marker[j] = j;

	/* Add pattern of column A_*k to B_*j */
	for (i = colptr[j]; i < colptr[j+1]; ++i) {
	    k = rowind[i];
	    if ( marker[k] != j ) {
		marker[k] = j;
		++num_nz;
	    }
	}

	/* Add pattern of column T_*k to B_*j */
	for (i = t_colptr[j]; i < t_colptr[j+1]; ++i) {
	    k = t_rowind[i];
	    if ( marker[k] != j ) {
		marker[k] = j;
		++num_nz;
	    }
	}
    }
    *bnz = num_nz;

    /* Allocate storage for A+A' */
    if ( !(*b_colptr = (int*) SUPERLU_MALLOC( (n+1) * sizeof(int)) ) )
	ABORT("SUPERLU_MALLOC fails for b_colptr[]");
    if ( *bnz) {
      if ( !(*b_rowind = (int*) SUPERLU_MALLOC( *bnz * sizeof(int)) ) )
	ABORT("SUPERLU_MALLOC fails for b_rowind[]");
    }

    /* Zero the diagonal flag */
    for (i = 0; i < n; ++i) marker[i] = -1;

    /* Compute each column of B, one at a time */
    num_nz = 0;
    for (j = 0; j < n; ++j) {
	(*b_colptr)[j] = num_nz;

	/* Flag the diagonal so it's not included in the B matrix */
	marker[j] = j;

	/* Add pattern of column A_*k to B_*j */
	for (i = colptr[j]; i < colptr[j+1]; ++i) {
	    k = rowind[i];
	    if ( marker[k] != j ) {
		marker[k] = j;
		(*b_rowind)[num_nz++] = k;
	    }
	}

	/* Add pattern of column T_*k to B_*j */
	for (i = t_colptr[j]; i < t_colptr[j+1]; ++i) {
	    k = t_rowind[i];
	    if ( marker[k] != j ) {
		marker[k] = j;
		(*b_rowind)[num_nz++] = k;
	    }
	}
    }
    (*b_colptr)[n] = num_nz;

    SUPERLU_FREE(marker);
    SUPERLU_FREE(t_colptr);
    SUPERLU_FREE(t_rowind);
}


int get_fillfactor(int n, int* colSum, int* rowIdx, double* data, int * perm_c,
#if SUPERLU_MAJOR_VERSION > 4
  GlobalLU_t Glu,
#endif
                   superlu_options_t options)
{
    int m,nnz,info;
    m = n; //Assume square matrix
    nnz = colSum[n];

    SuperLUStat_t stat;
    SuperMatrix A;
    int      *perm_r; /* row permutations from partial pivoting */
    SuperMatrix L;      /* factor L */
    SuperMatrix U;      /* factor U */
//    SCformat *Lstore;
//    NCformat *Ustore;

    dCreate_CompCol_Matrix(&A, m, n, nnz, data, rowIdx, colSum, SLU_NC, SLU_D, SLU_GE);

    if ( !(perm_r = intMalloc(m)) ) ABORT("Malloc fails for perm_r[].");

    /* Initialize the statistics variables. */
    StatInit(&stat);

    int * work = NULL;
    int lwork = 0;
    int relax;
    int panel_size;
    int * etree;
    etree = intMalloc(A.ncol);
    SuperMatrix AC;

    panel_size = sp_ienv(1);
    relax = sp_ienv(2);

    if(options.ColPerm != MY_PERMC)  get_perm_c(options.ColPerm, &A, perm_c);

    sp_preorder(&options, &A, perm_c, etree, &AC);
    dgstrf(&options, &AC, relax, panel_size, etree, work, lwork, perm_c, perm_r,  &L, &U,
#if SUPERLU_MAJOR_VERSION > 4
    &Glu,
#endif
           &stat, &info);

    int lu_nnz = ((SCformat *)L.Store)->nnz + ((NCformat *)U.Store)->nnz;

    Destroy_SuperMatrix_Store(&A);
    Destroy_CompCol_Permuted(&AC);
    Destroy_SuperNode_Matrix(&L);
    Destroy_CompCol_Matrix(&U);

    StatFree(&stat);
    free(perm_r);
    free(etree);

    return lu_nnz;
}
