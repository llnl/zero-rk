
#include "reactor_constant_pressure_gpu.h"

#include "zerork_cuda_defs.h"
#include "nvector/nvector_cuda.h"
#include "utility_funcs.h"

#include <thrust/functional.h>

#include "cuda_transpose.h"

ReactorConstantPressureGPU::ReactorConstantPressureGPU(std::shared_ptr<zerork::mechanism_cuda> mech_ptr)
 :
    ReactorNVectorSerialCuda(mech_ptr)
{
}

ReactorConstantPressureGPU::~ReactorConstantPressureGPU()
{
}

template<typename T>
struct invert_functor
{
    __host__ __device__
        T operator()(const T& x) const { 
            return T(1.0)/x;
        }
};


template<typename T>
struct saxpy_functor
{
    const T a;

    saxpy_functor(T _a) : a(_a) {}

    __host__ __device__
        T operator()(const T& x, const T& y) const { 
            return a * x + y;
        }
};

void ReactorConstantPressureGPU::InitializeState(
    const double reactor_time,
    const int n_reactors,
    const double *T,
    const double *P,
    const double *mf,
    const double *dpdt,
    const double *e_src,
    const double *y_src)
{
  assert(n_reactors <= max_num_reactors_);
  num_reactors_ = n_reactors;
  initial_time_ = reactor_time;

  N_VDestroy(state_);
  N_VDestroy(tmp1_);
  N_VDestroy(tmp2_);
  N_VDestroy(tmp3_);
  state_ = N_VMake_Cuda(num_variables_*num_reactors_,&state_data_[0], thrust::raw_pointer_cast(&state_data_dev_[0]));
  tmp1_ = N_VMake_Cuda(num_variables_*num_reactors_,&tmp1_data_[0], thrust::raw_pointer_cast(&tmp1_data_dev_[0]));
  tmp2_ = N_VMake_Cuda(num_variables_*num_reactors_,&tmp2_data_[0], thrust::raw_pointer_cast(&tmp2_data_dev_[0]));
  tmp3_ = N_VMake_Cuda(num_variables_*num_reactors_,&tmp3_data_[0], thrust::raw_pointer_cast(&tmp3_data_dev_[0]));

  double *y_ptr_dev = N_VGetDeviceArrayPointer_Cuda(state_);
  cudaMemcpy(thrust::raw_pointer_cast(tmp1_data_dev_.data()),
             mf,sizeof(double)*num_reactors_*num_species_,cudaMemcpyHostToDevice);
  cuda_transpose(thrust::raw_pointer_cast(state_data_dev_.data()),
		 thrust::raw_pointer_cast(tmp1_data_dev_.data()), num_species_, num_reactors_);

  initial_temperatures_dev_ = zerork::device_vector<double>(T, T+n_reactors);
  if(solve_temperature_) {
    cudaMemcpy(thrust::raw_pointer_cast(tmp3_data_dev_.data()),
               thrust::raw_pointer_cast(initial_temperatures_dev_.data()),sizeof(double)*num_reactors_,cudaMemcpyDeviceToDevice);
    thrust::device_ptr<double> scaled_temps(&y_ptr_dev[num_species_*num_reactors_]);
    const double inv_reference_temperature = 1/double_options_["reference_temperature"];
    thrust::transform(tmp3_data_dev_.begin(), tmp3_data_dev_.begin() + num_reactors_,
                      scaled_temps, thrust::placeholders::_1*inv_reference_temperature);
  }

  pressures_dev_ = zerork::device_vector<double>(P, P+n_reactors);

  if(y_src != nullptr) {
    cudaMemcpy(thrust::raw_pointer_cast(tmp2_data_dev_.data()),
               y_src,sizeof(double)*num_reactors_*num_species_,cudaMemcpyHostToDevice);
    y_src_dev_.resize(num_reactors_*num_species_);
    cuda_transpose(thrust::raw_pointer_cast(y_src_dev_.data()),
		   thrust::raw_pointer_cast(tmp2_data_dev_.data()), num_species_, num_reactors_);
  } else {
     y_src_dev_.clear();
  }

  if(dpdt != nullptr) {
    dpdts_dev_ = zerork::device_vector<double>(dpdt, dpdt+n_reactors);
  } else {
    dpdts_dev_.clear();
  }

  if(e_src != nullptr) {
    e_src_dev_ = zerork::device_vector<double>(e_src, e_src+n_reactors);
  } else {
    e_src_dev_.clear();
  }

  inverse_densities_dev_.resize(num_reactors_);
  mech_ptr_->getDensityFromTPY_mr_dev(num_reactors_, thrust::raw_pointer_cast(&initial_temperatures_dev_[0]),
                                 thrust::raw_pointer_cast(&pressures_dev_[0]), y_ptr_dev,
                                 thrust::raw_pointer_cast(&inverse_densities_dev_[0]));

  initial_energies_dev_.resize(num_reactors_);
  mech_ptr_->getMassEnthalpyFromTY_mr_dev(num_reactors_,
                                          thrust::raw_pointer_cast(&initial_temperatures_dev_[0]), y_ptr_dev,
                                          thrust::raw_pointer_cast(&energy_dev_[0]),
                                          thrust::raw_pointer_cast(&initial_energies_dev_[0]));

  //Need to invert density.
  thrust::transform(inverse_densities_dev_.begin(), inverse_densities_dev_.end(), inverse_densities_dev_.begin(), invert_functor<double>());
  thrust::copy(inverse_densities_dev_.begin(), inverse_densities_dev_.end(), inverse_densities_.begin());
}

void ReactorConstantPressureGPU::GetState(
    const double reactor_time,
    double *T,
    double *P,
    double *mf)
{
  double *y_ptr_dev = N_VGetDeviceArrayPointer_Cuda(state_);

  cuda_transpose(thrust::raw_pointer_cast(tmp1_data_dev_.data()),
		 thrust::raw_pointer_cast(state_data_dev_.data()), num_reactors_, num_species_);
  //TODO: Async
  cudaMemcpy(mf,thrust::raw_pointer_cast(tmp1_data_dev_.data()),sizeof(double)*num_reactors_*num_species_,cudaMemcpyDeviceToHost);
  if(solve_temperature_) {
    thrust::device_ptr<double> scaled_temps(&y_ptr_dev[num_species_*num_reactors_]);
    thrust::transform(scaled_temps, scaled_temps + num_reactors_,
                      temperatures_dev_.begin(),
                      thrust::placeholders::_1*double_options_["reference_temperature"]);
  } else {
    temperatures_dev_ = initial_temperatures_dev_;
    zerork::device_vector<double> energies_dev(initial_energies_dev_); //might be worth saving this temp vector
    if(e_src_dev_.size() > 0) {
      const double delta_t = reactor_time-initial_time_;
      thrust::transform(e_src_dev_.begin(), e_src_dev_.end(), initial_energies_dev_.begin(), energies_dev.begin(), saxpy_functor<double>(delta_t));
    }
    mech_ptr_->getTemperatureFromHY_mr_dev(num_reactors_, thrust::raw_pointer_cast(&energies_dev[0]),
                                           y_ptr_dev, thrust::raw_pointer_cast(&temperatures_dev_[0]));
  }
  //TODO: Async
  cudaMemcpy(T,thrust::raw_pointer_cast(&temperatures_dev_[0]),sizeof(double)*num_reactors_,cudaMemcpyDeviceToHost);
  zerork::device_vector<double> current_pressures = pressures_dev_;
  if(dpdts_dev_.size() != 0) {
    const double delta_t = reactor_time-initial_time_;
    thrust::transform(dpdts_dev_.begin(), dpdts_dev_.end(),
                      pressures_dev_.begin(), current_pressures.begin(),
                      saxpy_functor<double>(delta_t));
  }
  if(y_src_dev_.size() != 0) {
    mech_ptr_->getPressureFromTVY_mr_dev(num_reactors_,thrust::raw_pointer_cast(&temperatures_dev_[0]),
                                         thrust::raw_pointer_cast(&inverse_densities_dev_[0]),
                                         y_ptr_dev, thrust::raw_pointer_cast(&current_pressures[0]));
  }
  cudaMemcpy(P,thrust::raw_pointer_cast(&current_pressures[0]),sizeof(double)*num_reactors_,cudaMemcpyDeviceToHost);
}

int ReactorConstantPressureGPU::GetTimeDerivative(const double reactor_time,
                                                N_Vector state,
                                                N_Vector derivative)
{
  double* y_ptr_dev = N_VGetDeviceArrayPointer_Cuda(state);
  double* ydot_ptr_dev = N_VGetDeviceArrayPointer_Cuda(derivative);

#ifdef ZERORK_NEG_CONC_CHECK
  int neg_fractions_flag = this->CheckMassFractionsDevice(y_ptr_dev);
  if(neg_mass_fracs == 1) {
    return 1;
  }
#endif

  if(solve_temperature_) { 
    this->SetTemperatures(&(y_ptr_dev[num_species_*num_reactors_]), thrust::raw_pointer_cast(&temperatures_dev_[0]));
  } else {
    temperatures_dev_ = initial_temperatures_dev_;
  }

  //Update density (only if no y_src)
  if(y_src_dev_.size() == 0) {
    zerork::device_vector<double> current_pressures = pressures_dev_;
    if(dpdts_dev_.size() != 0) {
      const double delta_t = reactor_time-initial_time_;
      thrust::transform(dpdts_dev_.begin(), dpdts_dev_.end(), pressures_dev_.begin(), current_pressures.begin(), saxpy_functor<double>(delta_t));
    }

    mech_ptr_->getDensityFromTPY_mr_dev(num_reactors_,thrust::raw_pointer_cast(&temperatures_dev_[0]),
                                        thrust::raw_pointer_cast(&current_pressures[0]),y_ptr_dev,
                                        thrust::raw_pointer_cast(&inverse_densities_dev_[0]));
    //Need to invert density.
    thrust::transform(inverse_densities_dev_.begin(), inverse_densities_dev_.end(), inverse_densities_dev_.begin(), invert_functor<double>());
  }

  if(e_src_dev_.size() != 0) {
    const double delta_t = reactor_time-initial_time_;
    zerork::device_vector<double> energies_dev(num_reactors_); //might be worth saving this temp vector
    thrust::transform(e_src_dev_.begin(), e_src_dev_.end(), initial_energies_dev_.begin(), energies_dev.begin(), saxpy_functor<double>(delta_t));
    mech_ptr_->getTemperatureFromHY_mr_dev(num_reactors_, thrust::raw_pointer_cast(&energies_dev[0]), y_ptr_dev, thrust::raw_pointer_cast(&temperatures_dev_[0]));
  }

  // set concentration via density and mass fraction
  mech_ptr_->getCfromVY_mr_dev(num_reactors_,thrust::raw_pointer_cast(&inverse_densities_dev_[0]),y_ptr_dev,
                               thrust::raw_pointer_cast(&concentrations_dev_[0]));

  // compute the molar production rates at the current state (aka wdot)
  mech_ptr_->getReactionRatesLimiter_CUDA_mr_dev(num_reactors_,thrust::raw_pointer_cast(&temperatures_dev_[0]),
                                      thrust::raw_pointer_cast(&concentrations_dev_[0]), thrust::raw_pointer_cast(&step_limiter_[0]),
                                      thrust::raw_pointer_cast(&net_production_rates_dev_[0]),
                                      thrust::raw_pointer_cast(&creation_rates_dev_[0]),thrust::raw_pointer_cast(&destruction_rates_dev_[0]),
                                      thrust::raw_pointer_cast(&forward_rates_of_production_dev_[0]));

  this->ConcentrationDerivative(thrust::raw_pointer_cast(&inverse_densities_dev_[0]), ydot_ptr_dev);

  if(solve_temperature_) {
    mech_ptr_->getEnthalpy_RT_mr_dev(num_reactors_,thrust::raw_pointer_cast(&temperatures_dev_[0]),thrust::raw_pointer_cast(&energy_dev_[0]));
    mech_ptr_->getMassCpFromTY_mr_dev(num_reactors_,thrust::raw_pointer_cast(&temperatures_dev_[0]),y_ptr_dev,
                                      thrust::raw_pointer_cast(&cx_mass_dev_[0]),thrust::raw_pointer_cast(&mean_cx_mass_dev_[0]));

    this->TemperatureDerivative(thrust::raw_pointer_cast(&inverse_densities_dev_[0]), y_ptr_dev,ydot_ptr_dev);
  }
  return 0;
}


int ReactorConstantPressureGPU::RootFunction(double t, N_Vector y, double *root_function)
{
//  double ignition_temperature = initial_temperature_ + double_options_["delta_temperature_ignition"];
//  double current_temperature = NV_Ith_S(y,num_species_)*double_options_["reference_temperature"];
//  root_function[0] = ignition_temperature - current_temperature;
  return 0;
}


int ReactorConstantPressureGPU::GetNumRootFunctions()
{
  return 0;
}


void ReactorConstantPressureGPU::GetAbsoluteToleranceCorrection(N_Vector correction) {
  std::vector<double> atol_vector_cpu(num_variables_*num_reactors_,1.0);
  if(int_options_["abstol_dens"]) {
    for(int k = 0; k < num_reactors_; ++k)
    {
      double reactor_density = 1.0/inverse_densities_[k];
      for(int j=0; j < num_species_; ++j) {
        double molar_density = reactor_density*inv_mol_wt_[j]*1.0e-3; //mks->cgs
        atol_vector_cpu[j*num_reactors_+k] = 1.0/molar_density;
      }
    }
  }
  cudaMemcpy(N_VGetDeviceArrayPointer_Cuda(correction),&(atol_vector_cpu[0]),
             sizeof(double)*num_variables_*num_reactors_,cudaMemcpyHostToDevice);
}

