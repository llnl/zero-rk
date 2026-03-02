#include <algorithm> //std::max
#include "ode_funcs.h"
#include "cv_param_sparse.h"
#include "utility_funcs.h"

using zerork::getHighResolutionTime;

int const_vol_wsr(realtype t, N_Vector y, N_Vector ydot,
			 void *user_data)
{
  cv_param *cvp=(cv_param *)user_data;
  double *mfp=NV_DATA_S(y); // caution: assumes realtype == double
  double Esum,Temp;
  double pres, mixWt;
  double Ru = cvp->mech->getGasConstant();
  double mdot_in, mdot_out, K;
  double usum;
  int j;
  double startTime=getHighResolutionTime();

  // set temperature
  Temp = NV_Ith_S(y,cvp->nSpc)*cvp->Tref;

  // Get mass and mdot_in
  cvp->mass = NV_Ith_S(y,cvp->nSpc+1);
  mdot_in = std::max(cvp->mass/cvp->residenceTime, 0.0);

  // Get density
  cvp->Dens = cvp->mass/cvp->volume;
  cvp->invDens = 1.0/cvp->Dens;

  // Get current pressure
  mixWt = 0.0;
  for(j=0; j<cvp->nSpc; j++)
    mixWt += mfp[j]*cvp->invMolWt[j];
  mixWt = 1.0/mixWt;

  pres = cvp->Dens*Ru*Temp/mixWt;

  // set concentration via density and mass fraction
  cvp->mech->getCfromVY(cvp->invDens,mfp,cvp->conc);

  // compute the molar production rates at the current state (aka wdot)
  cvp->mech->getReactionRates(Temp,cvp->conc,cvp->netProd,cvp->createRate,
			      cvp->destroyRate,cvp->fwdROP);

  cvp->mech->getIntEnergy_RT(Temp,cvp->Energy);
  cvp->meanCvMass=cvp->mech->getMassCvFromTY(Temp,mfp,cvp->CvMass);

  // ydot = [kmol/m^3/s] * [kg/kmol] * [m^3/kg] = [(kg spec j)/(kg mix)/s]
  for(j=0; j<cvp->nSpc; j++) {
    NV_Ith_S(ydot,j)=(cvp->yInlet[j]-mfp[j])*mdot_in/cvp->mass +
      (cvp->netProd[j])*(cvp->molWt[j])*(cvp->invDens);
  }

  // Mass eq
  // dm/dt = m_in - m_out
  // m_out = K*(Pin-Pout) + m_in
  K = cvp->Kpressure*cvp->volume;
  mdot_out = std::max(-K*(cvp->pressure-pres) + mdot_in, 0.0);
  NV_Ith_S(ydot,cvp->nSpc+1)= mdot_in - mdot_out;

  // Energy Eq
  // m*cv*dT/dt = -Qdot + m_in*(h_in - sum_k(u_k*Y_k,in)) -p*V*m_out/m
  Esum=0.0;
  for(j=0; j<cvp->nSpc; j++)
    {Esum+=(cvp->Energy[j])*(cvp->netProd[j]);}

  Esum*=cvp->mech->getGasConstant()*NV_Ith_S(y,cvp->nSpc)*(cvp->invDens)
    /(cvp->meanCvMass);

  usum=0.0;
  for(j=0; j<cvp->nSpc; j++)
  {usum+=cvp->Energy[j]*cvp->yInlet[j]*cvp->invMolWt[j];}

  usum*=cvp->mech->getGasConstant()*NV_Ith_S(y,cvp->nSpc);

  cvp->dTemp_dt=-Esum +
    mdot_in*(cvp->enthalpyInlet - usum)/(cvp->mass*cvp->meanCvMass) -
    cvp->pressure*cvp->invDens*mdot_out/cvp->Tref/(cvp->mass*cvp->meanCvMass);

  if(cvp->energyEnabled) {
    NV_Ith_S(ydot,cvp->nSpc)=cvp->dTemp_dt;
  } else {
    NV_Ith_S(ydot,cvp->nSpc)=0.0;
  }

  (cvp->nFunc)++;
  cvp->funcTime += getHighResolutionTime() - startTime;
  return 0;
}

int tempRootFunc(realtype t, N_Vector y, realtype *rootFunc,
		 void *user_data)
{
  cv_param *cvp=(cv_param *)user_data;

  rootFunc[0]=NV_Ith_S(y,cvp->nSpc)*cvp->Tref - cvp->tempRoot;

  return 0;
}
