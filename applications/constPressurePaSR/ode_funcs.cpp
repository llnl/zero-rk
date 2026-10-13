#include "ode_funcs.h"
#include "cv_param_sparse.h"

int rhs_int(double* y, double* ydot, cv_param* cvp) {
  int nSpc = cvp->nSpc;
  double Esum,Temp;
  double pres, density, relative_volume, mix_mol_mass;
  double Ru = cvp->mech->getGasConstant();
  int j;
  double startTime=zerork::getHighResolutionTime();

  // set temperature
  Temp = y[cvp->nSpc]*cvp->Tref;

  // get pressure
  pres = cvp->pressure;

  // get density
  mix_mol_mass = 0.0;
  for(j=0; j<nSpc; j++)
    mix_mol_mass += y[j]*cvp->invMolWt[j];
  mix_mol_mass = 1.0/mix_mol_mass;

  density = pres*mix_mol_mass/(Ru*Temp);
  relative_volume = 1.0/density;
  cvp->Dens = density;
  cvp->invDens = relative_volume;

  // compute concentrations
  for(j=0; j<nSpc; j++)
    cvp->conc[j] = density*y[j]*cvp->invMolWt[j];

  // compute the molar production rates at the current state (aka wdot)
  cvp->mech->getReactionRatesLimiter(Temp,
                                     &cvp->conc[0],
                                     cvp->step_limiter,
                                     &cvp->netProd[0],
                                     &cvp->createRate[0],
                                     &cvp->destroyRate[0],
                                     &cvp->fwdROP[0]);

  for(j=0; j<cvp->nSpc; j++)
    ydot[j]=(cvp->netProd[j])*(cvp->molWt[j])*relative_volume;
  
  if(cvp->ymix != NULL) {
    for(j=0; j<cvp->nSpc; j++) {
       ydot[j] += (cvp->ymix[j] - y[j])/cvp->tau_mix;
    }
  }

  cvp->mech->getEnthalpy_RT(Temp,&cvp->Energy[0]);
  cvp->meanCpMass = cvp->mech->getMassCpFromTY(Temp,y,&cvp->CpMass[0]);

  Esum=0.0;
  for(j=0; j<cvp->nSpc; j++)
    //Esum+=(cvp->Energy[j])*(cvp->netProd[j]);
    Esum+=(cvp->Energy[j])*(ydot[j]*cvp->invMolWt[j]*density);

  Esum*=Ru*y[nSpc]*relative_volume/(cvp->meanCpMass);

  cvp->dTemp_dt=-Esum;

  //Heat transfer model
  cvp->htxModel->Update(Temp,pres,density,cvp->meanCpMass,y);
  double H = cvp->htxModel->H();
  double Qwall = -H*(Temp - cvp->htxModel->wall_temperature())/cvp->nParticles;
  double particle_mass = density*cvp->htxModel->reactor_volume()/cvp->nParticles;
  cvp->dTemp_dt += Qwall/(cvp->meanCpMass*particle_mass*cvp->Tref); // W/[J/(kg K)*kg] = K/s

  if(cvp->ymix != NULL) {
     cvp->dTemp_dt += (cvp->ymix[nSpc] - y[nSpc])/cvp->tau_mix;
  }

  ydot[nSpc]=cvp->dTemp_dt;

  (cvp->nFunc)++;

  cvp->funcTime += zerork::getHighResolutionTime() - startTime;
  return 0;
}

int const_vol_wsr(realtype t, N_Vector y, N_Vector ydot,
			 void *user_data)
{
  cv_param *cvp=(cv_param *)user_data;
  double *yptr=NV_DATA_S(y); // caution: assumes realtype == double
  double *ydotptr=NV_DATA_S(ydot); // caution: assumes realtype == double
  return rhs_int(yptr, ydotptr, cvp);
}

int const_vol_wsr_mix(realtype t, N_Vector y, N_Vector ydot,
			 void *user_data)
{
  cv_param *cvp=(cv_param *)user_data;
  int nState = cvp->nSpc + 1;
  double *yptr=NV_DATA_S(y); // caution: assumes realtype == double
  double *ydotptr=NV_DATA_S(ydot); // caution: assumes realtype == double
  double *yptr2 = &yptr[nState];
  double *ydotptr2 = &ydotptr[nState];

  cvp->ymix = yptr2;
  rhs_int(yptr, ydotptr, cvp);

  cvp->ymix = yptr;
  rhs_int(yptr2, ydotptr2, cvp);

  return 0;
}

