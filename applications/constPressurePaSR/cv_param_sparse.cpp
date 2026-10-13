#include "cv_param_sparse.h"

//n is length of perm_c which is allocated by calling function
void read_perm_c_from_file(int n, int* perm_c)
{
    int ncheck = 0;
    char line[1024];
    FILE *infile;
    infile=fopen("perm_c.txt", "r");

    //assumes perm_c has correct number of lines
    while (fgets(line, 1023, infile) != NULL && ncheck <= n) {
        perm_c[ncheck++] = atoi(line);
    }
    if(ncheck != n)
      {
        printf("ERROR: perm_c.txt not consistent with mechanism.");
        exit(-1);
      }
    return;
}



JsparseTermList * alloc_JsparseTermList(const int inpFD, const int inpFC)
{
  JsparseTermList *w;

  w=(JsparseTermList *)malloc(sizeof(JsparseTermList));
  if(w==NULL)
    {
      printf("ERROR: allocation 1 failed in alloc_JsparseTermList(...)\n");
      return NULL;
    }
  w->nFwdDestroy=inpFD;
  w->nFwdCreate =inpFC;

  w->fwdDestroy=(JsparseIdx *)malloc(sizeof(JsparseIdx)*inpFD);
  if(w->fwdDestroy==NULL)
    {
      free(w);
      printf("ERROR: allocation 2 failed in alloc_JsparseTermList(...)\n");
      return NULL;
    }

  w->fwdCreate=(JsparseIdx *)malloc(sizeof(JsparseIdx)*inpFC);
  if(w->fwdCreate==NULL)
    {
      free(w->fwdDestroy);
      free(w);
      printf("ERROR: allocation 3 failed in alloc_JsparseTermList(...)\n");
      return NULL;
    }

  return w;
}

void free_JsparseTermList(JsparseTermList *w)
{
  if(w != NULL)
    {
      free(w->fwdCreate);
      free(w->fwdDestroy);

      free(w);
    }
}

Jsparse * alloc_Jsparse(zerork::mechanism &mechInp, double tol,
                        bool doILU, bool fakeUpdate, int threshType,
                        double DiagPivotThresh, int permutationType)
{
  int nSpc=mechInp.getNumSpecies();
  int nStep=mechInp.getNumSteps();
  int nReac,nProd;
  int j,k,m,rowIdx,colIdx,nzAddr;

  int nFD,nFC;

  int *isNonZero; // local dense matrix indicating the non-zero terms

  Jsparse *w;
  w=(Jsparse *)malloc(sizeof(Jsparse));
  if(w==NULL)
    {
      printf("ERROR: allocation 1 failed in alloc_Jsparse(...)\n");
      fflush(stdout);
      return NULL;
    }

  countJacobianTerms(mechInp,&nFD,&nFC);
  w->termList=alloc_JsparseTermList(nFD,nFC);
  if(w->termList==NULL)
    {
      free(w);
      printf("ERROR: allocation 2 failed in alloc_Jsparse(...)\n");
      fflush(stdout);
      return NULL;
    }
  w->nSize=nSpc+1; // matrix size
  w->nSizeMix=2*w->nSize;

  // allocate temporary arrays
  isNonZero = (int *)malloc(sizeof(int)*(w->nSize)*(w->nSize));

  // allocate memory that is related to the matrix dimension
  w->mtxColSum  = (int *)malloc(sizeof(int)*(w->nSize+1));
  w->mtxColSumMix  = (int *)malloc(sizeof(int)*((2*w->nSize)+1));
  w->diagIdx    = (int *)malloc(sizeof(int)*(w->nSize));
  w->lastRowIdx = (int *)malloc(sizeof(int)*(w->nSize));

  // initialize the dense nonzero flags
  for(j=0; j<nSpc; j++) // process the first nSpc columns
    {
      for(k=0; k<nSpc; k++) // process the first nSpc rows
	{isNonZero[j*(w->nSize)+k]=0;}

      isNonZero[j*(w->nSize)+j]=1;    // mark the diagonal
      isNonZero[j*(w->nSize)+nSpc]=1; // mark the last row
    }
  for(k=0; k<(w->nSize); k++) // mark nSize rows in the last column
    {isNonZero[nSpc*(w->nSize)+k]=1;}

  // re-parse the system filling in the Jacobian term data
  // Jacobian = d ydot(k)/ dy(j)
  nFD=nFC=0;

  for(j=0; j<nStep; j++)
    {
      nReac=mechInp.getOrderOfStep(j);
      nProd=mechInp.getNumProductsOfStep(j);
      for(k=0; k<nReac; k++)
	{
	  colIdx=mechInp.getSpecIdxOfStepReactant(j,k); // species being perturbed

	  // forward destruction
	  for(m=0; m<nReac; m++)
	    {
	      rowIdx=mechInp.getSpecIdxOfStepReactant(j,m); // species being destroyed
	      isNonZero[colIdx*(w->nSize)+rowIdx]=1; // mark location in dense

	      w->termList->fwdDestroy[nFD].concIdx=colIdx;
	      w->termList->fwdDestroy[nFD].rxnIdx=j;
	      w->termList->fwdDestroy[nFD].sparseIdx=rowIdx;
	      ++nFD;
	    }
	  // forward creation
	  for(m=0; m<nProd; m++)
	    {
	      rowIdx=mechInp.getSpecIdxOfStepProduct(j,m); // species being created
	      isNonZero[colIdx*(w->nSize)+rowIdx]=1; // mark location in dense

	      w->termList->fwdCreate[nFC].concIdx=colIdx;
	      w->termList->fwdCreate[nFC].rxnIdx=j;
	      w->termList->fwdCreate[nFC].sparseIdx=rowIdx;
	      ++nFC;
	    }
	}
    }

  // check the jacobian term count
  if(nFD != w->termList->nFwdDestroy)
    {
      printf("ERROR: in alloc_Jterm_param()\n");
      printf("       nFD = %d != %d = w->nFwdDestroy\n",nFD,
	     w->termList->nFwdDestroy);
      exit(-1);
    }
  if(nFC != w->termList->nFwdCreate)
    {
      printf("ERROR: in alloc_Jterm_param()\n");
      printf("       nFC = %d != %d = w->nFwdCreate\n",nFC,
	     w->termList->nFwdCreate);
      exit(-1);
    }

   // count the number of nonzero terms in the dense matrix
   w->nNonZero=1; // start the count at one so it can still serve as a flag
   w->mtxColSum[0]=0;
   for(j=0; j<(w->nSize); j++) // process column j
     {
       for(k=0; k<(w->nSize); k++)
	 {
	   if(isNonZero[j*(w->nSize)+k]==1)
	     {
	       isNonZero[j*(w->nSize)+k]=w->nNonZero;
	       w->nNonZero++;
	     }
	 }
       // after counting column j store the running total in column j+1
       // of the column sum
       w->mtxColSum[j+1]=w->nNonZero-1;
     }
   // now at each nonzero term, isNonZero is storing the (address+1) in the
   // actual compressed column storage data array

   w->nNonZero--; // decrement the count
   //printf("# number of nonzero terms = %d\n",w->nNonZero);
   fflush(stdout);

   // allocate matrix data
   w->mtxData=(double *)malloc(sizeof(double)*w->nNonZero);
   w->mtxRowIdx=(int *)malloc(sizeof(int)*w->nNonZero);
   w->nNonZeroMix = w->nNonZero*2+2*w->nSize;
   w->mtxDataMix=(double *)malloc(sizeof(double)*w->nNonZeroMix);
   w->mtxRowIdxMix=(int *)malloc(sizeof(int)*w->nNonZeroMix);
   for(j=0; j<w->nNonZero; j++)
     {w->mtxData[j]=0.0;} // initialize to zero for the print function

   // scan the the isNonZero array to determine the row indexes
   // and the special data addresses
   for(j=0; j<(w->nSize); j++) // process column j
     {
       for(k=0; k<(w->nSize); k++)  // row k
	 {
	   nzAddr=isNonZero[j*(w->nSize)+k];
	   if(nzAddr>=1)
	     {w->mtxRowIdx[nzAddr-1]=k;}
	 }
       // record the diagonal address
       w->diagIdx[j]    =isNonZero[j*(w->nSize)+j]-1;
       w->lastRowIdx[j]=isNonZero[(j+1)*(w->nSize)-1]-1;
     }

   // use the isNonZero array as a lookup to store the proper compressed
   // column data storage
   for(j=0; j<nFD; j++)
     {
       rowIdx=w->termList->fwdDestroy[j].sparseIdx;
       colIdx=w->termList->fwdDestroy[j].concIdx;
       nzAddr=isNonZero[colIdx*(w->nSize)+rowIdx];
       if(nzAddr==0)
	 {
	   printf("ERROR: %d term in fwdDestroy points to a zero J term\n",
		  j);
	   printf("       at J(%d,%d)\n",rowIdx,colIdx);
	   exit(-1);
	 }
       w->termList->fwdDestroy[j].sparseIdx=nzAddr-1; // reset to sparse addr
     }
   for(j=0; j<nFC; j++)
     {
       rowIdx=w->termList->fwdCreate[j].sparseIdx;
       colIdx=w->termList->fwdCreate[j].concIdx;
       nzAddr=isNonZero[colIdx*(w->nSize)+rowIdx];
       if(nzAddr==0)
	 {
	   printf("ERROR: %d term in fwdCreate points to a zero J term\n",
		  j);
	   printf("       at J(%d,%d)\n",rowIdx,colIdx);
	   exit(-1);
	 }
       w->termList->fwdCreate[j].sparseIdx=nzAddr-1; // reset to sparse addr
     }

   w->mtxColSumMix[0] = 0;
   for(j=0; j<w->nSize; j++) {
      int kStart = w->mtxColSum[j];
      int kEnd = w->mtxColSum[j+1];
      w->mtxColSumMix[j+1] = w->mtxColSumMix[j] + (kEnd-kStart) + 1;
      for(k=kStart; k<kEnd; ++k)
      {
	w->mtxRowIdxMix[k+j] = w->mtxRowIdx[k];
      }
      w->mtxRowIdxMix[kEnd+j] = w->nSize+j;
   }

   int kOffset = w->nNonZero+w->nSize;
   for(j=0; j<w->nSize; j++) {
      int kStart = w->mtxColSum[j];
      int kEnd = w->mtxColSum[j+1];
      int jMix = j+w->nSize;
      w->mtxColSumMix[jMix+1] = w->mtxColSumMix[jMix] + (kEnd-kStart) + 1;
      //w->mtxRowIdxMix[w->mtxColSumMix[jMix]] = j;
      w->mtxRowIdxMix[kStart+kOffset+j] = j;
      for(k=kStart; k<kEnd; ++k)
      {
	w->mtxRowIdxMix[k+kOffset+j+1] = w->mtxRowIdx[k]+w->nSize;
      }
   }

   // Now the JsparseTermList should contain a consistent set of indexes
   // for processing the Jacobian term by term.  Note that each element
   // in the list is independent, so the lists of JsparseIdx can be sorted
   // to try to improve cache locality

   // sparse solver specific allocations
   // SuperLU
   w->isFirstFactor=1; // indicate to do the full factorization
   /* Set the default input options:
	options.Fact = DOFACT;
        options.Equil = YES;
    	options.ColPerm = COLAMD;
    	options.Trans = NOTRANS;
    	options.IterRefine = NOREFINE;
	options.DiagPivotThresh = 1.0;
    	options.SymmetricMode = NO;
    	options.PivotGrowth = NO;
    	options.ConditionNumber = NO;
    	options.PrintStat = YES;
   */
   set_default_options(&(w->optionSLU));

   /* Set the default options for ILU
        options.Fact = DOFACT;
        options.Equil = YES;
        options.ColPerm = COLAMD;
        options.Trans = NOTRANS;
        options.IterRefine = NOREFINE;
        options.DiagPivotThresh = 0.1; //different from complete LU
        options.SymmetricMode = NO;
        options.PivotGrowth = NO;
        options.ConditionNumber = NO;
        options.PrintStat = YES;
        options.RowPerm = LargeDiag;
        options.ILU_DropTol = 1e-4;
        options.ILU_FillTol = 1e-2;
        options.ILU_FillFactor = 10.0;
        options.ILU_DropRule = DROP_BASIC | DROP_AREA;
        options.ILU_Norm = INF_NORM;
        options.ILU_MILU = SILU;
   */
   ilu_set_default_options(&(w->optionSLU));

   w->optionSLU.ILU_DropTol = 5e-4;
   w->optionSLU.ILU_FillTol = 1e-4;
   w->optionSLU.ILU_FillFactor = 2.0;
   w->optionSLU.DiagPivotThresh = DiagPivotThresh;
   w->optionSLU.ILU_DropRule = DROP_BASIC | DROP_AREA;
   w->optionSLU.ILU_MILU = SILU; //SMILU_{1,2,3};
//   w->optionSLU.ColPerm = MMD_AT_PLUS_A; // best for iso-octane
//   w->optionSLU.ColPerm = COLAMD;
//   w->optionSLU.ColPerm = MMD_ATA;
   w->optionSLU.ColPerm = MY_PERMC;
   w->optionSLU.RowPerm = LargeDiag_MC64;
   w->optionSLU.Equil=YES;
//   w->optionSLU.SymmetricMode = YES; //Small DiagPivotThresh and MMD_AT_PLUS_A

   w->vecRHS=(double *)malloc(sizeof(double)*(w->nSize)*2);
   w->vecSoln=(double *)malloc(sizeof(double)*(w->nSize)*2);
   w->rowPermutation=(int *)malloc(sizeof(int)*(w->nSize));
   w->colPermutation=(int *)malloc(sizeof(int)*(w->nSize));
   w->colElimTree=(int *)malloc(sizeof(int)*(w->nSize));

   w->rowPermutationMix=(int *)malloc(sizeof(int)*(w->nSize*2));
   w->colPermutationMix=(int *)malloc(sizeof(int)*(w->nSize*2));
   w->colElimTreeMix=(int *)malloc(sizeof(int)*(w->nSize*2));

   w->reduceData=(double *)malloc(sizeof(double)*w->nNonZero);
   w->reduceRowIdx=(int *)malloc(sizeof(int)*w->nNonZero);
   w->reduceColSum=(int *)malloc(sizeof(int)*(w->nSize+1));

   w->reduceDataMix=(double *)malloc(sizeof(double)*w->nNonZeroMix);
   w->reduceRowIdxMix=(int *)malloc(sizeof(int)*w->nNonZeroMix);
   w->reduceColSumMix=(int *)malloc(sizeof(int)*(2*w->nSize+1));

   // create a compressed column matrix
   // SLU_NC ==
   // SLU_D  == data type double precision
   // SLU_GE == matrix structure general
   //dCreate_CompCol_Matrix(&(w->Mslu),w->nSize,w->nSize,w->nNonZero,
   //			  w->mtxData,w->mtxRowIdx,w->mtxColSum,
   //			  SLU_NC,SLU_D,SLU_GE);
    dCreate_CompCol_Matrix(&(w->Mslu),
			     w->nSize,
			     w->nSize,
			     w->nNonZero,
			     w->reduceData,
			     w->reduceRowIdx,
			     w->reduceColSum,
			     SLU_NC,SLU_D,SLU_GE);
   w->Rvec=(double *)malloc(sizeof(double)*(w->nSize));
   w->Cvec=(double *)malloc(sizeof(double)*(w->nSize));

    dCreate_CompCol_Matrix(&(w->MsluMix),
			     2*w->nSize,
			     2*w->nSize,
			     w->nNonZeroMix,
			     w->reduceDataMix,
			     w->reduceRowIdxMix,
			     w->reduceColSumMix,
			     SLU_NC,SLU_D,SLU_GE);
   w->RvecMix=(double *)malloc(sizeof(double)*(2*w->nSize));
   w->CvecMix=(double *)malloc(sizeof(double)*(2*w->nSize));

   dCreate_Dense_Matrix(&(w->Bslu),w->nSize,1,w->vecRHS,w->nSize,
			SLU_DN,SLU_D,SLU_GE);
   dCreate_Dense_Matrix(&(w->Xslu),w->nSize,1,w->vecSoln,w->nSize,
			SLU_DN,SLU_D,SLU_GE);

   dCreate_Dense_Matrix(&(w->BsluMix),2*w->nSize,1,w->vecRHS,2*w->nSize,
			SLU_DN,SLU_D,SLU_GE);
   dCreate_Dense_Matrix(&(w->XsluMix),2*w->nSize,1,w->vecSoln,2*w->nSize,
			SLU_DN,SLU_D,SLU_GE);

   StatInit(&(w->statSLU)); // initialize SuperLU statistics

   w->offDiagThreshold=tol;

   w->ILU = doILU;
   w->fakeUpdate = fakeUpdate;
   w->threshType = threshType;

   //Metis column permutation
   w->permutationType = permutationType;

#ifdef CVIDT_USE_METIS
   METIS_SetDefaultOptions(w->metis_options);

   //Set to -1 for default
   w->metis_options[METIS_OPTION_NUMBERING] = 0; // C-numbering
   w->metis_options[METIS_OPTION_PFACTOR] = 0; // Default: 0 ["Good values often in the range of 60 to 200"]
   w->metis_options[METIS_OPTION_CCORDER] = 1;  // Default: 0
   w->metis_options[METIS_OPTION_COMPRESS] = 1; // Default: 1
   w->metis_options[METIS_OPTION_NITER] = 10; // Default: 10
   w->metis_options[METIS_OPTION_NSEPS] = 1; // Default: 1
   w->metis_options[METIS_OPTION_RTYPE] = METIS_RTYPE_SEP1SIDED; // Default: ??
//METIS_RTYPE_FM FM-based cut refinnement. (SegFault)
//METIS_RTYPE_GREEDY Greedy-based cut and volume refinement. (SegFault)
//METIS_RTYPE_SEP2SIDED Two-sided node FM refinement.
//METIS_RTYPE_SEP1SIDED One-sided node FM refinement.
   w->metis_options[METIS_OPTION_CTYPE] = METIS_CTYPE_RM; // Default: SHEM
//METIS_CTYPE_RM Random matching.
//METIS_CTYPE_SHEM Sorted heavy-edge matching.
   w->metis_options[METIS_OPTION_UFACTOR] = 1; // Default: 1 or 30 (ptype=rb or ptype=kway)
   w->metis_options[METIS_OPTION_SEED] = 0; // Default: ??
#endif

   if(w->permutationType < 1 || w->permutationType > 4)
     {
        printf("ERROR: Invalid permutationType: %d.",w->permutationType);
        exit(-1);
     }
   if(w->permutationType == 4)
     {
       read_perm_c_from_file(w->nSize,w->colPermutation);
     }

   free(isNonZero);
   return w;
}

void free_Jsparse(Jsparse *w)
{
  if(w!=NULL)
    {
      StatFree(&(w->statSLU));
      Destroy_SuperMatrix_Store(&(w->Bslu));
      Destroy_SuperMatrix_Store(&(w->Xslu));
      Destroy_SuperMatrix_Store(&(w->BsluMix));
      Destroy_SuperMatrix_Store(&(w->XsluMix));
      if(!w->isFirstFactor) {
     //TODO: Leaking L and U because we don't know which have been allocated based on "fully_coupled"
        //Destroy_SuperNode_Matrix(&(w->Lslu));
        //Destroy_CompCol_Matrix(&(w->Uslu));
        //Destroy_SuperNode_Matrix(&(w->LsluMix));
        //Destroy_CompCol_Matrix(&(w->UsluMix));
      }

      free(w->mtxData);
      free(w->mtxRowIdx);
      free(w->mtxColSum);
      free(w->mtxRowIdxMix);
      free(w->mtxColSumMix);
      free(w->diagIdx);
      free(w->lastRowIdx);

      free(w->mtxDataMix);

      free(w->reduceData);
      free(w->reduceRowIdx);
      free(w->reduceColSum);
      Destroy_SuperMatrix_Store(&(w->Mslu));

      free(w->reduceDataMix);
      free(w->reduceRowIdxMix);
      free(w->reduceColSumMix);
      Destroy_SuperMatrix_Store(&(w->MsluMix));

      // SuperLU data
      free(w->vecSoln);
      free(w->vecRHS);
      free(w->Rvec);
      free(w->Cvec);
      free(w->rowPermutation);
      free(w->colPermutation);
      free(w->colElimTree);

      free(w->RvecMix);
      free(w->CvecMix);
      free(w->rowPermutationMix);
      free(w->colPermutationMix);
      free(w->colElimTreeMix);

      free_JsparseTermList(w->termList);
      free(w);
    }
}

// void print_Jsparse(Jsparse *w)
// {
//   int j,k,nElems;
//   int rowIdx;
//   double val;
//   printf("# Number of nonzero elements: %d\n",w->nNonZero);

//   for(j=0; j<w->nSize; j++)
//     {
//       nElems=w->mtxColSum[j+1]-w->mtxColSum[j];
//       printf("   Col %d has %d nonzero elements\n",j,nElems);
//       for(k=0; k<nElems; k++)
// 	{
// 	  val=w->mtxData[w->mtxColSum[j]+k];
// 	  rowIdx=w->mtxRowIdx[w->mtxColSum[j]+k];
// 	  printf("     J(%d,%d) = %14.6e\n",rowIdx,j,val);
// 	}
//       printf("\n");
//       //exit(-1);
//     }

//   printf("# Number of forward destruction terms: %d\n",
// 	 w->termList->nFwdDestroy);
//   printf("# Number of forward creation terms: %d\n",
// 	 w->termList->nFwdCreate);
//   printf("# Number of reverse destruction terms: %d\n",
// 	 w->termList->nRevDestroy);
//   printf("# Number of reverse creation terms: %d\n",
// 	 w->termList->nRevCreate);
//   printf("# Forward Destruction Terms:\n");
//   printf("# i.e. both species are reactants in the same reaction\n");
//   printf("#\n#                  numer   denom\n");
//   printf("# J term   rxn ID  spc ID  spc ID  sparse mtx ID\n");
//   for(j=0; j<(w->termList->nFwdDestroy); j++)
//     {
//       printf("%6d  %6d  %6d  %6d  %6d\n",j,
// 	     w->termList->fwdDestroy[j].rxnIdx,
// 	     w->mtxRowIdx[w->termList->fwdDestroy[j].sparseIdx],
// 	     w->termList->fwdDestroy[j].concIdx,
// 	     w->termList->fwdDestroy[j].sparseIdx);
//     }
//   printf("\n\n");
//   printf("# Forward Creation Terms:\n");
//   printf("# i.e. numerator species is a product and the denominator\n");
//   printf("#      is a reactant in the same reaction\n");
//   printf("#\n#                  numer   denom\n");
//   printf("# J term   rxn ID  spc ID  spc ID\n");
//   for(j=0; j<(w->termList->nFwdCreate); j++)
//     {
//       printf("%6d  %6d  %6d  %6d  %6d\n",j,
// 	     w->termList->fwdCreate[j].rxnIdx,
// 	     w->mtxRowIdx[w->termList->fwdCreate[j].sparseIdx],
// 	     w->termList->fwdCreate[j].concIdx,
// 	     w->termList->fwdCreate[j].sparseIdx);
//     }
//   printf("\n\n");
//   printf("# Reverse Destruction Terms:\n");
//   printf("# i.e. both species are products in the same reversible reaction\n");
//   printf("#\n#                  numer   denom\n");
//   printf("# J term   rxn ID  spc ID  spc ID  sparse mtx ID\n");
//   for(j=0; j<(w->termList->nRevDestroy); j++)
//     {
//       printf("%6d  %6d  %6d  %6d  %6d\n",j,
// 	     w->termList->revDestroy[j].rxnIdx,
// 	     w->mtxRowIdx[w->termList->revDestroy[j].sparseIdx],
// 	     w->termList->revDestroy[j].concIdx,
// 	     w->termList->revDestroy[j].sparseIdx);
//     }
//   printf("\n\n");
//   printf("# Reverse Creation Terms:\n");
//   printf("# i.e. numerator species is a reactant and the denominator\n");
//   printf("#      is a product in the same reversible reaction\n");
//   printf("#\n#                  numer   denom\n");
//   printf("# J term   rxn ID  spc ID  spc ID\n");
//   for(j=0; j<(w->termList->nRevCreate); j++)
//     {
//       printf("%6d  %6d  %6d  %6d  %6d\n",j,
// 	     w->termList->revCreate[j].rxnIdx,
// 	     w->mtxRowIdx[w->termList->revCreate[j].sparseIdx],
// 	     w->termList->revCreate[j].concIdx,
// 	     w->termList->revCreate[j].sparseIdx);
//     }
// }

void countJacobianTerms(zerork::mechanism &mechInp, int *nFwdDestroy, int *nFwdCreate)
{
  int j;
  int nStep=mechInp.getNumSteps();
  double nuSumReac,nuSumProd;

  (*nFwdDestroy)=0;
  (*nFwdCreate)=0;

  for(j=0; j<nStep; j++)
    {
      nuSumReac=mechInp.getOrderOfStep(j);
      nuSumProd=mechInp.getNumProductsOfStep(j);
      (*nFwdDestroy)+=nuSumReac*nuSumReac;
      (*nFwdCreate)+=nuSumReac*nuSumProd;
    }
}

double getElement_Jsparse(Jsparse *w,const int rowIdx, const int colIdx)
{
  int nElem=w->mtxColSum[colIdx+1]-w->mtxColSum[colIdx];
  int j;
  int currentRow;

  for(j=0; j<nElem; j++)
    {
      currentRow=w->mtxRowIdx[w->mtxColSum[colIdx]+j];
      if(rowIdx == currentRow)
	{return w->mtxData[w->mtxColSum[colIdx]+j];}
      else if(rowIdx < currentRow)
	{return 0.0;}
    }
  return 0.0;
}

void calcReaction_Jsparse(Jsparse *w, const double invPosConc[],
			  const double fwdROP[])
{
  int j;
  int rxnId,concId,sparseId;

  // set the full sparse array
  for(j=0; j<w->nNonZero; j++)
    {w->mtxData[j]=0.0;}

  // process the forward destruction terms
  for(j=0; j<w->termList->nFwdDestroy; j++)
    {
      concId   = w->termList->fwdDestroy[j].concIdx;
      rxnId    = w->termList->fwdDestroy[j].rxnIdx;
      sparseId = w->termList->fwdDestroy[j].sparseIdx;
      //printf(" j = %d, concId = %d, rxnId = %d, sparseId = %d\n",j,concId,rxnId,sparseId); fflush(stdout);
     w->mtxData[sparseId]-=fwdROP[rxnId]*invPosConc[concId];
    }
  // process the forward creation terms
  for(j=0; j<w->termList->nFwdCreate; j++)
    {
      concId   = w->termList->fwdCreate[j].concIdx;
      rxnId    = w->termList->fwdCreate[j].rxnIdx;
      sparseId = w->termList->fwdCreate[j].sparseIdx;
      w->mtxData[sparseId]+=fwdROP[rxnId]*invPosConc[concId];
    }
}
void change_JsparseThresh(Jsparse *w, double newThresh)
{w->offDiagThreshold=newThresh;}
