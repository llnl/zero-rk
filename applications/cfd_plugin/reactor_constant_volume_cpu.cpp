#include <algorithm> //std::max,min
#include <cassert>

#include "reactor_constant_volume_cpu.h"

#include "nvector/nvector_serial.h"

ReactorConstantVolumeCPU::ReactorConstantVolumeCPU(std::shared_ptr<zerork::mechanism> mech_ptr)
 :
    ReactorNVectorSerial(mech_ptr)
{}

ReactorConstantVolumeCPU::~ReactorConstantVolumeCPU()
{}

void ReactorConstantVolumeCPU::InitializeState(
    const double reactor_time,
    const int n_reactors,
    const double *T,
    const double *P,
    const double *mf,
    const double *dpdt,
    const double *e_src,
    const double *y_src)
{
  assert(n_reactors == 1);
  ReactorNVectorSerial::Reset(); //re-initialize to get deterministic behavior across solves
  initial_time_ = reactor_time;
  pressure_ = *P;
  initial_temperature_ = *T;
  inverse_density_ = 1.0/mech_ptr_->getDensityFromTPY(initial_temperature_,*P,mf);
  initial_energy_ = mech_ptr_->getMassIntEnergyFromTY(initial_temperature_,mf);
  double *y_ptr = NV_DATA_S(state_);
  for(int k = 0; k < num_species_; ++k) {
    y_ptr[k] = mf[k];
  }
  if(solve_temperature_) {
    y_ptr[num_species_] = initial_temperature_/double_options_["reference_temperature"];
  }
  dpdt_ = *dpdt;
  e_src_ = *e_src;
  y_src_ = y_src;
}

void ReactorConstantVolumeCPU::GetState(
    const double reactor_time,
    double *T,
    double *P,
    double *mf)
{
  double *y_ptr = NV_DATA_S(state_);
  if(solve_temperature_) {
    *T = NV_Ith_S(state_,num_species_)*double_options_["reference_temperature"];
  } else {
    double energy = initial_energy_ + e_src_*(reactor_time - initial_time_);
    *T = mech_ptr_->getTemperatureFromEY(energy, y_ptr, initial_temperature_);
  }
  for(int k = 0; k < num_species_; ++k) {
    mf[k] = y_ptr[k];
  }
  *P = mech_ptr_->getPressureFromTVY(*T, inverse_density_, mf) + dpdt_*(reactor_time-initial_time_);
}


int ReactorConstantVolumeCPU::GetTimeDerivative(const double reactor_time,
                                                N_Vector state,
                                                N_Vector derivative)
{
  //double startTime=getHighResolutionTime();
  double * y_ptr = NV_DATA_S(state);
  double * ydot_ptr = NV_DATA_S(derivative);
  double * net_production_rates_ptr = &(net_production_rates_[0]);
  double * creation_rates_ptr = &(creation_rates_[0]);
  double * destruction_rates_ptr = &(destruction_rates_[0]);
  double * forward_rates_of_production_ptr = &(forward_rates_of_production_[0]);
  double * energy_ptr = &(energy_[0]);
  double * cv_mass_ptr = &(cx_mass_[0]);
  const int num_spec = num_species_;
  const double reference_temperature = double_options_["reference_temperature"];
  double temperature = initial_temperature_;
  if(solve_temperature_) {
    temperature = y_ptr[num_spec]*reference_temperature;
  }
  if(temperature <= 0.0) return 1;
#define TLIMIT 1.0e4
  temperature = std::min(temperature,TLIMIT);

  if(e_src_ != 0) {
    double energy = initial_energy_ + e_src_*(reactor_time-initial_time_);
    temperature = mech_ptr_->getTemperatureFromEY(energy, y_ptr, temperature);
  }
  mech_ptr_->getIntEnergy_RT(temperature,energy_ptr);
  mean_cx_mass_ = mech_ptr_->getMassCvFromTY(temperature,y_ptr,cv_mass_ptr);

  double current_inverse_density = inverse_density_;
  if(dpdt_ != 0) {
    pressure_ = mech_ptr_->getPressureFromTVY(temperature, inverse_density_, y_ptr) + dpdt_*(reactor_time-initial_time_);
    current_inverse_density = 1.0/mech_ptr_->getDensityFromTPY(temperature,pressure_,y_ptr);
  }
  // set concentration via density and mass fraction
  mech_ptr_->getCfromVY(current_inverse_density,y_ptr,&concentrations_[0]);

  // compute the molar production rates at the current state_ (aka wdot)
  mech_ptr_->getReactionRatesLimiter(temperature, &concentrations_[0], &step_limiter_[0],
                                     net_production_rates_ptr, creation_rates_ptr, destruction_rates_ptr,
                                     forward_rates_of_production_ptr);

  double energy_sum=0.0;
  // ydot = [kmol/m^3/s] * [kg/kmol] * [m^3/kg] = [(kg spec j)/(kg mix)/s]
  if(y_src_ != nullptr) {
    for(int j=0; j<num_spec; ++j) {
      ydot_ptr[j]=(net_production_rates_[j]*mol_wt_[j]*current_inverse_density + y_src_[j]);
      energy_sum += energy_ptr[j]*(net_production_rates_[j]*current_inverse_density+y_src_[j]*inv_mol_wt_[j]);
    }
  } else {
    for(int j=0; j<num_spec; ++j) {
      ydot_ptr[j]=(net_production_rates_[j]*mol_wt_[j]*current_inverse_density);
      energy_sum += energy_ptr[j]*net_production_rates_[j]*current_inverse_density;
    }
  }

  if(solve_temperature_) {
    double dT_dt = -energy_sum * mech_ptr_->getGasConstant() *
                   y_ptr[num_spec] / mean_cx_mass_;
    ydot_ptr[num_spec]= dT_dt + (e_src_ + dpdt_*current_inverse_density) / (mean_cx_mass_*reference_temperature);
  }

  return 0;
}


