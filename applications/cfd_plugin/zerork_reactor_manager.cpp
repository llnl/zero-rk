
#ifdef USE_MPI
#include "mpi.h"
#endif

#include <cassert>
#include <iomanip>
#include <stdexcept>

#include "zerork_reactor_manager.h"
#include "ZeroRKCFDPluginIFP.h"

#include "solver_cvode.h"
#include "solver_seulex.h"
#include "solver_sodex.h"
#include "solver_radau.h"
#include "reactor_constant_volume_cpu.h"
#include "reactor_constant_pressure_cpu.h"

#ifdef ZERORK_GPU
#include "reactor_constant_volume_gpu.h"
#include "reactor_constant_pressure_gpu.h"
#endif

#include "utility_funcs.h"

#include "file_utilities.h" //zerork::utilities

using zerork::getHighResolutionTime;

ZeroRKReactorManager::ZeroRKReactorManager()
  :
      ZeroRKReactorManagerBase(),
      mech_ptr_(nullptr)
{
  n_calls_ = 0;
  n_cycle_ = 0;
  rank_ = 0;
  sorted_rank_ = 0;
  nranks_ = 1;
  root_rank_ = 0;
  tried_init_ = false;
  avg_reactor_time_ = 1.0;

#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD,&(rank_));
  MPI_Comm_size(MPI_COMM_WORLD,&(nranks_));
#endif

  //Default options
  string_options_["mech_filename"] = std::string("mech.dat");
  string_options_["therm_filename"] = std::string("therm.dat");

  //Manager Options
  int_options_["verbosity"] = 4;
  int_options_["sort_reactors"]= 1;
  int_options_["load_balance"] = 1;
  if(nranks_ == 1) {
    int_options_["load_balance"] = 0;
  }
  int_options_["load_balance_noise"] = 0;
  int_options_["load_balance_mem"] = 1;
  int_options_["reactor_weight_mult"] = 1;
  int_options_["dump_reactors"] = 0;
  int_options_["dump_failed_reactors"] = 0;
  int_options_["output_performance_log"] = 1;

  //Solver options
  int_options_["max_steps"] = 5000;
  int_options_["dense"] = 0;
  int_options_["analytic"] = 1;
  int_options_["iterative"] = 1;
  int_options_["integrator"] = 0;
  int_options_["abstol_dens"] = 0;
  int_options_["cvode_num_retries"] = 5;
  double_options_["rel_tol"] = 1.0e-8;
  double_options_["abs_tol"] = 1.0e-20;
  double_options_["eps_lin"] = 1.0e-3;
  double_options_["nonlinear_convergence_coeff"] = 0.05;
  double_options_["preconditioner_threshold"] = 1.0e-3;
  double_options_["max_dt"] = 0.05;
  double_options_["cvode_retry_absolute_tolerance_adjustment"] = 0.1;
  double_options_["cvode_retry_relative_tolerance_adjustment"] = 1.0;

  //Reactor Options
  int_options_["constant_volume"] = 1;
  double_options_["reference_temperature"] = 1.0;
  double_options_["delta_temperature_ignition"] = 0.0;
  double_options_["min_mass_fraction"] = 1.0e-30;
  int_options_["always_solve_temperature"] = 1;
  double_options_["solve_temperature_threshold"] = 2.0;
  double_options_["step_limiter"] = 1.0e22;

  //GPU Options
  int_options_["gpu"] = 0;
  int_options_["initial_gpu_multiplier"] = 8;
  int_options_["n_reactors_min"] = 128;
  int_options_["n_reactors_max"] = 1024;

  //File-output Options
  string_options_["reactor_timing_log_filename"] = std::string(zerork::utilities::null_filename);
  string_options_["mechanism_parsing_log_filename"] = std::string(zerork::utilities::null_filename);

  T_other_.clear();
  P_other_.clear();
  dpdt_other_.clear();
  e_src_other_.clear();
  y_src_other_.clear();
  mf_other_.clear();
  rc_other_.clear();
  rg_other_.clear();
  reactor_ids_other_.clear();
  root_times_other_.clear();
  temp_delta_other_.clear();

  rc_owned_ = true;
  rg_owned_ = true;
  root_times_owned_ = true;
  temp_delta_owned_ = true;
  dpdt_defined_ = false;
  e_src_defined_ = false;
  y_src_defined_ = false;
  dpdt_self_ = nullptr;
  e_src_self_ = nullptr;
  y_src_self_ = nullptr;
  rc_self_ = nullptr;
  rg_self_ = nullptr;
  temp_delta_self_ = nullptr;
  reactor_ids_defined_ = false;
  root_times_self_ = nullptr;

  cb_fn_ = nullptr;
  cb_fn_data_ = nullptr;
}

zerork_status_t ZeroRKReactorManager::ReadOptionsFile(const std::string& options_filename) {

  std::unique_ptr<ZeroRKCFDPluginIFP> inputFileDBptr;
  try {
    inputFileDBptr = std::make_unique<ZeroRKCFDPluginIFP>(options_filename);
  } catch (const std::runtime_error& e) {
    return ZERORK_STATUS_FAILED_OPTIONS_PARSE;
  }
  const ZeroRKCFDPluginIFP& inputFileDB(*inputFileDBptr);

  int_options_["always_solve_temperature"] = inputFileDB.always_solve_temperature();
  double_options_["solve_temperature_threshold"] = inputFileDB.solve_temperature_threshold();

  //TODO: Don't over-ride if options file used default value?
  int_options_["verbosity"] = inputFileDB.verbosity();
  int_options_["sort_reactors"] = inputFileDB.sort_reactors();
  int_options_["load_balance_mem"] = inputFileDB.load_balance_mem();

  int_options_["max_steps"] = inputFileDB.max_steps();
  int_options_["dense"] = inputFileDB.dense();
  int_options_["analytic"] = inputFileDB.analytic();
  int_options_["iterative"] = inputFileDB.iterative();
  int_options_["integrator"] = inputFileDB.integrator();
  int_options_["abstol_dens"] = inputFileDB.abstol_dens();
  int_options_["cvode_num_retries"] = inputFileDB.cvode_num_retries();
  double_options_["abs_tol"] = inputFileDB.absolute_tolerance();
  double_options_["rel_tol"] = inputFileDB.relative_tolerance();
  double_options_["eps_lin"] = inputFileDB.eps_lin();
  double_options_["nonlinear_convergence_coeff"] = inputFileDB.nonlinear_convergence_coeff();
  double_options_["preconditioner_threshold"] = inputFileDB.preconditioner_threshold();
  double_options_["max_dt"] = inputFileDB.max_dt();
  double_options_["cvode_retry_absolute_tolerance_adjustment"] = inputFileDB.cvode_retry_absolute_tolerance_adjustment();
  double_options_["cvode_retry_relative_tolerance_adjustment"] = inputFileDB.cvode_retry_relative_tolerance_adjustment();

  int_options_["constant_volume"] = inputFileDB.constant_volume();
  double_options_["reference_temperature"] = inputFileDB.reference_temperature();
  double_options_["delta_temperature_ignition"] = inputFileDB.delta_temperature_ignition();
  double_options_["min_mass_fraction"] = inputFileDB.min_mass_fraction();
  double_options_["step_limiter"] = inputFileDB.step_limiter();

  int_options_["gpu"] = inputFileDB.gpu();
  int_options_["initial_gpu_multiplier"] = inputFileDB.initial_gpu_multiplier();
  int_options_["n_reactors_max"] = inputFileDB.n_reactors_max();
  int_options_["n_reactors_min"] = inputFileDB.n_reactors_min();
#ifdef USE_MPI
  int_options_["load_balance"] = inputFileDB.load_balance();
  int_options_["load_balance_noise"] = inputFileDB.load_balance_noise();
  int_options_["reactor_weight_mult"] = inputFileDB.reactor_weight_mult();
#endif
  int_options_["dump_reactors"] = inputFileDB.dump_reactors();
  int_options_["dump_failed_reactors"] = inputFileDB.dump_failed_reactors();
  int_options_["output_performance_log"] = inputFileDB.output_performance_log();

  string_options_["reactor_timing_log_filename"] = inputFileDB.reactor_timing_log();
  string_options_["mechanism_parsing_log_filename"] = inputFileDB.mechanism_parsing_log();
  return ZERORK_STATUS_SUCCESS;
}

zerork_status_t ZeroRKReactorManager::LoadMechanism() {
  std::string cklog_filename(string_options_["mechanism_parsing_log_filename"]);
  if(rank_ != root_rank_) {
     cklog_filename = std::string(zerork::utilities::null_filename);
  }

  try {
    mech_ptr_ = std::make_shared<zerork::mechanism>(string_options_["mech_filename"].c_str(),
                    string_options_["therm_filename"].c_str(),
                    cklog_filename.c_str());
  } catch (const std::runtime_error& e) {
    mech_ptr_ = nullptr;
    return ZERORK_STATUS_FAILED_MECHANISM_PARSE;
  }
#ifdef ZERORK_GPU
  rank_has_gpu_.assign(nranks_,0);
  gpu_id_ = -1;
  if(int_options_["gpu"] != 0) {
    gpu_id_ = -2;
    AssignGpuId();
  }
  int have_gpu = gpu_id_ > -1 ? 1 : 0;
  rank_has_gpu_[rank_] = have_gpu;
#ifdef USE_MPI
  MPI_Allgather(&have_gpu,1,MPI_INT,&rank_has_gpu_[0],1,MPI_INT,MPI_COMM_WORLD);
#endif
  n_cpu_ranks_ = nranks_;
  n_gpu_ranks_ = 0;
  for(int i = 0; i < nranks_; ++i) {
    if(rank_has_gpu_[i] == 1) {
      n_gpu_ranks_ += 1;
      n_cpu_ranks_ -= 1;
    }
  }

  if(rank_has_gpu_[rank_]) {
    try {
      int verbosity = 0;
      mech_cuda_ptr_ = std::make_shared<zerork::mechanism_cuda>(string_options_["mech_filename"].c_str(),
                          string_options_["therm_filename"].c_str(),
                          cklog_filename.c_str(), verbosity, int_options_["n_reactors_max"]);
    } catch (const std::runtime_error& e) {
      mech_cuda_ptr_ = nullptr;
      return ZERORK_STATUS_FAILED_MECHANISM_PARSE;
    }
  }
#endif
  return ZERORK_STATUS_SUCCESS;
}

#ifdef ZERORK_GPU
//Auto-assigns based on node rank
//Use CUDA_VISIBLE_DEVICES to choose/re-order
void ZeroRKReactorManager::AssignGpuId() {
  int n_devices;
  //This is to work-around issues with job schedulers that
  //don't show all GPUs to all ranks on the node
  if(getenv("ZERORK_GPUS_PER_NODE") != NULL) {
    n_devices = atoi(getenv("ZERORK_GPUS_PER_NODE"));
  } else {
    cudaGetDeviceCount(&n_devices);
  }
  if(rank_ == 0) printf("n_devices = %d\n",n_devices);
  if(n_devices == 0) {
    gpu_id_ = -1;
  }
  if(gpu_id_ == -2) {
#ifdef USE_MPI
    gpu_id_ = -1;
    int rank, nprocs, namelen;
    char host_name[MPI_MAX_PROCESSOR_NAME] = "";
    MPI_Comm communicator = MPI_COMM_WORLD;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &nprocs);
    MPI_Get_processor_name(host_name,&namelen);

    std::vector<std::string> host_names(nprocs);
    host_names[rank] = host_name;
    for(int n=0; n<nprocs; n++) {
      strncpy(host_name,host_names[n].c_str(),host_names[n].length());
      MPI_Bcast(host_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, n, communicator);
      host_names[n] = host_name;
    }
    strncpy(host_name,host_names[rank].c_str(),host_names[rank].length());
    std::sort(host_names.begin(), host_names.end());

    int color = 0;
    for (int n=0; n<nprocs; n++) {
      if( n>0 && host_names[n-1] != host_names[n]) color++;
      if(std::string(host_name) == host_names[n]) break;
    }

    int node_rank;
    MPI_Comm nodeComm;
    MPI_Comm_split(communicator, color, 0, &nodeComm);
    MPI_Comm_rank(nodeComm, &node_rank);
    MPI_Comm_free(&nodeComm);
#else
   int node_rank = 0;
#endif

    int ranks_per_gpu = 1;
    if(getenv("ZERORK_GPU_MPS_RANKS") != NULL) {
      ranks_per_gpu = atoi(getenv("ZERORK_GPU_MPS_RANKS"));
    }
    /* Assign device to MPI process*/
    if(node_rank / n_devices < ranks_per_gpu) {
      gpu_id_ = node_rank % n_devices;
    }
    if(gpu_id_ >= 0) {
#ifdef USE_MPI
      printf("Assigning device %d to process on node %s rank %d \n", gpu_id_, host_name, rank);
#else
      printf("Assigning device %d to process\n", gpu_id_);
#endif
      cudaSetDevice(gpu_id_);
      cudaDeviceSynchronize();
      if(cudaGetLastError() != cudaSuccess) {
        //Failed to set device.  Fall back to cpu.
#ifdef USE_MPI
        printf("Failed to assign device %d to process on node %s rank %d \n", gpu_id_, host_name, rank);
#else
        printf("Failed to assign device %d to process\n", gpu_id_);
#endif
        gpu_id_ = -1;
      }
    }
  }
}
#endif

#ifdef USE_MPI
int ZeroRKReactorManager::RecvReactors(size_t send_rank) {
  MPI_Status status;
  int n_recv_reactors;
  MPI_Recv(&n_recv_reactors,1, MPI_INT, send_rank, EXCHANGE_SEND_TAG_, MPI_COMM_WORLD,
           &status);
  if(n_recv_reactors == 0) return n_recv_reactors;

  size_t recv_idx = n_reactors_other_;
  size_t extra_tx_count = 0;
  n_reactors_other_ += n_recv_reactors;

  T_other_.resize(n_reactors_other_);
  P_other_.resize(n_reactors_other_);
  if(dpdt_defined_) {
    dpdt_other_.resize(n_reactors_other_);
    extra_tx_count += 1;
  }
  if(e_src_defined_) {
    e_src_other_.resize(n_reactors_other_);
    extra_tx_count += 1;
  }
  if(y_src_defined_) {
    y_src_other_.resize(n_reactors_other_*num_species_);
    extra_tx_count += num_species_;
  }
  rg_other_.resize(n_reactors_other_);
  rc_other_.resize(n_reactors_other_);
  root_times_other_.assign(n_reactors_other_, -1.0); //N.B. at start negative one by definition so we don't communicate it
  temp_delta_other_.resize(n_reactors_other_);
  mf_other_.resize(n_reactors_other_*num_species_);

  const int load_balance_mem = int_options_["load_balance_mem"];
  std::vector<double> recv_buf(tx_count_per_reactor_+extra_tx_count);
  if(load_balance_mem != 0) {
    recv_buf.resize((tx_count_per_reactor_+extra_tx_count)*n_recv_reactors);
    MPI_Recv(&recv_buf[0],(tx_count_per_reactor_+extra_tx_count)*n_recv_reactors,
             MPI_DOUBLE, send_rank, EXCHANGE_SEND_TAG_,
             MPI_COMM_WORLD, &status);
  }
  size_t buf_idx = 0;
  for(size_t i = 0; i < n_recv_reactors; ++i, ++recv_idx) {
     //Bring 'em in
     if(load_balance_mem == 0) {
       MPI_Recv(&recv_buf[0],tx_count_per_reactor_+extra_tx_count,
                MPI_DOUBLE, send_rank, EXCHANGE_SEND_TAG_,
                MPI_COMM_WORLD, &status);
       buf_idx = 0;
     }
     //Unpack 'em
     memcpy(&mf_other_[recv_idx*num_species_], &recv_buf[buf_idx],
            sizeof(double)*num_species_);
     buf_idx += num_species_;
     memcpy(&T_other_[recv_idx], &recv_buf[buf_idx], sizeof(double));
     buf_idx += 1;
     memcpy(&P_other_[recv_idx], &recv_buf[buf_idx], sizeof(double));
     buf_idx += 1;
     if(dpdt_defined_) {
       memcpy(&dpdt_other_[recv_idx], &recv_buf[buf_idx], sizeof(double));
       buf_idx += 1;
     }
     if(e_src_defined_) {
       memcpy(&e_src_other_[recv_idx], &recv_buf[buf_idx], sizeof(double));
       buf_idx += 1;
     }
     if(y_src_defined_) {
       memcpy(&y_src_other_[recv_idx*num_species_], &recv_buf[buf_idx],
              sizeof(double)*num_species_);
       buf_idx += num_species_;
     }
     memcpy(&rg_other_[recv_idx], &recv_buf[buf_idx], sizeof(double));
     buf_idx += 1;
     memcpy(&rc_other_[recv_idx], &recv_buf[buf_idx], sizeof(double));
     buf_idx += 1;
     memcpy(&temp_delta_other_[recv_idx], &recv_buf[buf_idx], sizeof(double));
     buf_idx += 1;
  }
  if(reactor_ids_defined_) {
     reactor_ids_other_.resize(n_reactors_other_);
     MPI_Recv(&reactor_ids_other_[0],n_reactors_other_,
              MPI_INT, send_rank, EXCHANGE_SEND_TAG_,
              MPI_COMM_WORLD, &status);
  }
  return n_recv_reactors;
}
#endif


#ifdef USE_MPI
void ZeroRKReactorManager::SendReactors(std::vector<size_t> send_reactor_idxs, size_t recv_rank) {
  int send_nreactors = (int) send_reactor_idxs.size();
  MPI_Send(&send_nreactors,1,
           MPI_INT, (int) recv_rank, EXCHANGE_SEND_TAG_, MPI_COMM_WORLD);

  if(send_nreactors == 0) {
    return;
  }

  size_t extra_tx_count = 0;
  if(dpdt_defined_) extra_tx_count+=1;
  if(e_src_defined_) extra_tx_count+=1;
  if(y_src_defined_) extra_tx_count+=num_species_;
  std::vector<double> send_buf;
  const int load_balance_mem = int_options_["load_balance_mem"];
  if(load_balance_mem == 0) {
    send_buf.resize(tx_count_per_reactor_+extra_tx_count);
  } else {
    send_buf.resize((tx_count_per_reactor_+extra_tx_count)*send_nreactors);
  }
  int buf_idx = 0;
  for(int i = 0; i < send_nreactors; ++i) {
    if(load_balance_mem == 0) {
      buf_idx = 0;
    }
    int send_idx = send_reactor_idxs[i];
    memcpy(&send_buf[buf_idx], &mf_self_[send_idx*num_species_stride_],
           sizeof(double)*num_species_);
    buf_idx += num_species_;
    memcpy(&send_buf[buf_idx], &T_self_[send_idx], sizeof(double));
    buf_idx += 1;
    memcpy(&send_buf[buf_idx], &P_self_[send_idx], sizeof(double));
    buf_idx += 1;
    if(dpdt_defined_) {
      memcpy(&send_buf[buf_idx], &dpdt_self_[send_idx], sizeof(double));
      buf_idx += 1;
    }
    if(e_src_defined_) {
      memcpy(&send_buf[buf_idx], &e_src_self_[send_idx], sizeof(double));
      buf_idx += 1;
    }
    if(y_src_defined_) {
      memcpy(&send_buf[buf_idx], &y_src_self_[send_idx*num_species_stride_],
             sizeof(double)*num_species_);
      buf_idx += num_species_;
    }
    memcpy(&send_buf[buf_idx], &rg_self_[send_idx], sizeof(double));
    buf_idx += 1;
    memcpy(&send_buf[buf_idx], &rc_self_[send_idx], sizeof(double));
    buf_idx += 1;
    memcpy(&send_buf[buf_idx], &temp_delta_self_[send_idx], sizeof(double));
    buf_idx += 1;

    if(load_balance_mem == 0) {
      MPI_Send(&send_buf[0], tx_count_per_reactor_+extra_tx_count,
               MPI_DOUBLE, (int) recv_rank, EXCHANGE_SEND_TAG_, MPI_COMM_WORLD);
    }
  }
  if(load_balance_mem != 0) {
    MPI_Send(&send_buf[0], send_nreactors*(tx_count_per_reactor_+extra_tx_count),
             MPI_DOUBLE, (int) recv_rank, EXCHANGE_SEND_TAG_, MPI_COMM_WORLD);
  }
  if(reactor_ids_defined_) {
     std::vector<int> send_ids(send_nreactors);
     for(int i = 0; i < send_nreactors; ++i) {
         int send_idx = send_reactor_idxs[i];
         send_ids[i] = reactor_ids_self_[send_idx];
     }
     MPI_Send(&send_ids[0], send_nreactors, MPI_INT,
              (int) recv_rank, EXCHANGE_SEND_TAG_, MPI_COMM_WORLD);
  }
}
#endif

zerork_status_t ZeroRKReactorManager::SetInputVariables(int n_cycle,
                                             double time,
                                             double dt,
                                             int n_reactors,
                                             double* T,
                                             double* P,
                                             double* mf)

{
  n_cycle_ = n_cycle;
  dt_calc_ = dt;
  n_reactors_self_ = n_reactors;
  n_reactors_other_ = 0;
  T_self_ = T;
  P_self_ = P;
  mf_self_ = mf;

  if(n_reactors_self_ > 0) {
    if(rg_owned_) {
      rg_default_.resize(n_reactors, 0.0);
      rg_self_ = &rg_default_[0];
    }
    if(rc_owned_) {
      rc_default_.resize(n_reactors, 1.0);
      rc_self_ = &rc_default_[0];
    }
    if(root_times_owned_) {
      root_times_default_.assign(n_reactors, -1.0);
      root_times_self_ = &root_times_default_[0];
    }
    if(temp_delta_owned_) {
      temp_delta_default_.resize(n_reactors, 0.0);
      temp_delta_self_ = &temp_delta_default_[0];
    }

    sorted_reactor_idxs_.assign(n_reactors_self_,0);
    std::iota(sorted_reactor_idxs_.begin(), sorted_reactor_idxs_.end(), 0);
    if(int_options_["sort_reactors"]) {
        sorted_reactor_idxs_ = sort_indexes_pointer(n_reactors_self_, rc_self_);
    }
  }
  return ZERORK_STATUS_SUCCESS;
}

zerork_status_t ZeroRKReactorManager::SetAuxFieldPointer(zerork_field_t ft, double* field_pointer) {
  if(ft == ZERORK_FIELD_DPDT) {
      dpdt_self_ = field_pointer;
      dpdt_defined_ = true;
  } else if (ft == ZERORK_FIELD_COST) {
      rc_self_ = field_pointer;
      rc_default_.clear();
      rc_owned_ = false;
  } else if (ft == ZERORK_FIELD_GPU) {
      rg_self_ = field_pointer;
      rg_default_.clear();
      rg_owned_ = false;
  } else if (ft == ZERORK_FIELD_Y_SRC) {
      y_src_self_ = field_pointer;
      y_src_defined_ = true;
  } else if (ft == ZERORK_FIELD_E_SRC) {
      e_src_self_ = field_pointer;
      e_src_defined_ = true;
  } else if (ft == ZERORK_FIELD_IGNITION_TIME) {
      root_times_self_ = field_pointer;
      root_times_default_.clear();
      root_times_owned_ = false;
  } else if (ft == ZERORK_FIELD_TEMPERATURE_DELTA) {
      temp_delta_self_ = field_pointer;
      temp_delta_default_.clear();
      temp_delta_owned_ = false;
  } else {
      return ZERORK_STATUS_INVALID_FIELD_NAME;
  }
  return ZERORK_STATUS_SUCCESS;
}

zerork_status_t ZeroRKReactorManager::SetReactorIDs(int* reactor_ids) {
  reactor_ids_self_ = reactor_ids;
  reactor_ids_defined_ = true;
  return ZERORK_STATUS_SUCCESS;
}

zerork_status_t ZeroRKReactorManager::SetCallbackFunction(zerork_callback_fn fn, void* cb_fn_data) {
  cb_fn_ = fn;
  cb_fn_data_ = cb_fn_data;
  return ZERORK_STATUS_SUCCESS;
}

zerork_status_t ZeroRKReactorManager::FinishInit() {
  if(!tried_init_) {
    tried_init_ = true;

    if(mech_ptr_ == nullptr) {
      zerork_status_t flag = this->LoadMechanism();
      if(flag) return flag;
    }

    num_species_ = mech_ptr_->getNumSpecies();
    num_species_stride_ = num_species_;

    tx_count_per_reactor_ = 5 + num_species_;

#ifdef USE_MPI
    rank_weights_.assign(nranks_,1);
#ifdef ZERORK_GPU
    gpu_multiplier_ = int_options_["initial_gpu_multiplier"];
    for(int i = 0; i < nranks_; ++i) {
      if(rank_has_gpu_[i] == 1) {
        rank_weights_[i] = gpu_multiplier_;
      }
    }
#endif
    double sum_weights = 0;
    for(int i = 0 ; i < nranks_; ++i) {
      sum_weights += rank_weights_[i];
    }
    double weight_factor = nranks_/sum_weights;
    for(int i = 0 ; i < nranks_; ++i) {
      rank_weights_[i] *= weight_factor;
    }
#endif

    all_time_ranks_.assign(nranks_,0);
    n_reactors_solved_ranks_.assign(nranks_,0);
    n_weight_updates_ = 0;

    if(rank_ == root_rank_) {
      if(int_options_["verbosity"] > 1) {
        printf("* %-25s%-31s *\n", "Zero-RK Lib Build Date: ",__DATE__);
      }
      if(int_options_["output_performance_log"]!=0) {
        reactor_log_file_.open(string_options_["reactor_timing_log_filename"]);

        // Timing log file
        reactor_log_file_ << "#";
        reactor_log_file_ << std::setw(11) << "solve_number";
        reactor_log_file_ << std::setw(17) << "reactors_solved";
        reactor_log_file_ << std::setw(17) << "n_cpu";
        reactor_log_file_ << std::setw(17) << "n_cpu_notemp";
        reactor_log_file_ << std::setw(17) << "n_gpu";
        reactor_log_file_ << std::setw(17) << "n_gpu_notemp";
        reactor_log_file_ << std::setw(17) << "n_gpu_groups";
        reactor_log_file_ << std::setw(17) << "n_steps_avg";
        reactor_log_file_ << std::setw(17) << "n_steps_avg_cpu";
        reactor_log_file_ << std::setw(17) << "n_steps_avg_gpu";
        reactor_log_file_ << std::setw(17) << "max_time_cpu";
        reactor_log_file_ << std::setw(17) << "max_time_gpu";
        reactor_log_file_ << std::setw(17) << "step_time_cpu";
        reactor_log_file_ << std::setw(17) << "step_time_gpu";
        reactor_log_file_ << std::setw(17) << "avg_time_total";
        reactor_log_file_ << std::setw(17) << "max_time_total";
        reactor_log_file_ << std::endl;
        reactor_log_file_.flush();
      }
    }

    num_species_stride_ = num_species_;
    if(int_options_.find("num_species_stride") != int_options_.end()) {
      num_species_stride_ = int_options_["num_species_stride"];
    }
  } else { //tried_init_
    if(mech_ptr_ == nullptr) {
      return ZERORK_STATUS_FAILED_MECHANISM_PARSE;
    }
  }
  return ZERORK_STATUS_SUCCESS;
}

#ifndef USE_MPI
zerork_status_t ZeroRKReactorManager::LoadBalance()
{
  return ZERORK_STATUS_SUCCESS;
}
#else
zerork_status_t ZeroRKReactorManager::LoadBalance()
{
  const int load_balance = int_options_["load_balance"];
  if(nranks_ == 1 || !load_balance) {
    return ZERORK_STATUS_SUCCESS;
  }

  const int load_balance_noise = int_options_["load_balance_noise"];
  const int reactor_weight_mult = int_options_["reactor_weight_mult"];
  int n_weighted_reactors = 0;
  std::vector<int> weighted_reactors_on_rank(nranks_,0);
  if(load_balance == 1) {
    for(int k = 0; k < n_reactors_self_; ++k) {
      if(rc_self_[k] <= 0.0) rc_self_[k] = 1.0;
      n_weighted_reactors += std::max((int)rc_self_[k],1)+load_balance_noise;
    }
  } else if (load_balance == 2) {
    for(int k = 0; k < n_reactors_self_; ++k) {
        if(rg_self_[k] <= 0.0) rg_self_[k] = avg_reactor_time_;
        int weighted = (int)(reactor_weight_mult*rg_self_[k]/avg_reactor_time_);
        n_weighted_reactors += std::max(weighted,1);
    }
  }

  MPI_Allgather(&n_weighted_reactors,1,MPI_INT,
                &weighted_reactors_on_rank[0],1,MPI_INT,MPI_COMM_WORLD);

  bool sending = false;
  bool recving = false;

  //Counts of how many reactors we want to add to ours to be balanced overall
  std::vector<int> reactors_wanted(nranks_,0);
  std::vector<int> reactor_deficit(nranks_,0);

  //Sparse matrix of reactors transmitted to/from ranks.
  //  (N.B. Column indicies are in reverse order from normal CSR)
  comm_mtx_row_sum_.assign(nranks_+1,0);
  comm_mtx_col_idx_.clear();
  comm_mtx_reactors_.clear();

  int total_weighted_reactors = 0;
  sorted_rank_ = -1;

  double sum_weights = 0.0;
  for(int i = 0; i < nranks_; ++i) {
    total_weighted_reactors += weighted_reactors_on_rank[i];
    sum_weights += rank_weights_[i];
  }

  int total_wanted = 0;
  double weight_factor = ((double)total_weighted_reactors)/sum_weights;
  for(int i = 0; i < nranks_; ++i) {
    reactors_wanted[i] = rank_weights_[i]*weight_factor;
    total_wanted += reactors_wanted[i];
  }

  int verbosity = int_options_["verbosity"];
  for(int i = 0; i < nranks_; ++i) {
    reactor_deficit[i] = reactors_wanted[i] - weighted_reactors_on_rank[i];
    if(rank_==root_rank_ && verbosity > 0) {
      printf("Rank[%d] reactors_wanted, weighted_reactors, reactor_deficit,"
             " rank_weight: %d, %d, %d, %f\n",
             i, reactors_wanted[i], weighted_reactors_on_rank[i],
             reactor_deficit[i], rank_weights_[i]);
    }
  }

  sorted_deficit_idxs_ = sort_indexes(reactor_deficit);
  for(int i = 0; i < nranks_; ++i) {
    //note where we are in the list
    if(sorted_deficit_idxs_[i] == rank_) {
      sorted_rank_ = i;
      break;
    }
  }
  assert(sorted_rank_ >= 0);

  const int MIN_CHUNK = 1;
  int give_rank = nranks_-1; // smallest deficit (i.e. negative deficits)
                            // gets stolen from first
  for(int i = 0; i < nranks_; ++i) {
    comm_mtx_row_sum_[i+1] = comm_mtx_row_sum_[i]; //keep the running sum going
    double current_deficit = reactor_deficit[sorted_deficit_idxs_[i]];
    while(current_deficit >= MIN_CHUNK &&       //while we still want more
          give_rank > i) {                      //and there's more to be taken
      int extra_reactors_wanted    = current_deficit;
      int extra_reactors_available = -reactor_deficit[sorted_deficit_idxs_[give_rank]];
      if(extra_reactors_available >= MIN_CHUNK) {
        int reactors_taken = std::min(extra_reactors_available, extra_reactors_wanted);
        reactor_deficit[sorted_deficit_idxs_[i]] -= reactors_taken;
        reactor_deficit[sorted_deficit_idxs_[give_rank]] += reactors_taken;
        comm_mtx_row_sum_[i+1] += 1;
        comm_mtx_col_idx_.push_back(give_rank);
        comm_mtx_reactors_.push_back(reactors_taken);
      }
      if(-reactor_deficit[sorted_deficit_idxs_[give_rank]] < MIN_CHUNK) { //no more to give on this rank_
        give_rank -= 1;
      }
      if(reactor_deficit[sorted_deficit_idxs_[give_rank]] > 0) { //no more to give at all
        break;
      }
      current_deficit = reactor_deficit[sorted_deficit_idxs_[i]];
    }
  }
  for(int i = 0; i < comm_mtx_col_idx_.size(); ++i) {
    if(comm_mtx_col_idx_[i] == sorted_rank_) {
      sending = true;
      break;
    }
  }
  if((comm_mtx_row_sum_[sorted_rank_+1]-comm_mtx_row_sum_[sorted_rank_]) > 0) recving = true;

  //Exchange excess reactors
  if(recving) {
    for(int i = comm_mtx_row_sum_[sorted_rank_]; i < comm_mtx_row_sum_[sorted_rank_+1]; ++i) {
      int sorted_send_rank = comm_mtx_col_idx_[i];
      int send_rank = sorted_deficit_idxs_[sorted_send_rank]; //reactor_deficit[sorted_send_rank].first;
      int recv_nreactors = RecvReactors(send_rank);
      comm_mtx_reactors_[i] = recv_nreactors; //over-write weighted value with actual value
    }
  }

  if(sending) {
    int send_idx = n_reactors_self_;
    for(int ircv = 0; ircv < nranks_; ++ircv) {
      int recv_rank = sorted_deficit_idxs_[ircv]; //reactor_deficit[ircv].first;
      if(recv_rank == rank_) continue;
      for(int j = comm_mtx_row_sum_[ircv]; j < comm_mtx_row_sum_[ircv+1]; ++j) {
        int sorted_send_rank = comm_mtx_col_idx_[j];
        int send_rank = sorted_deficit_idxs_[sorted_send_rank]; //reactor_deficit[sorted_send_rank].first;
        if(send_rank == rank_) {
          int send_weighted_nreactors = comm_mtx_reactors_[j];
          int send_count = 0;
          std::vector<size_t> send_reactor_idxs;
          while(send_count < send_weighted_nreactors && send_idx>0) {
            send_idx -= 1;
            size_t sorted_reactor_idx = sorted_reactor_idxs_[send_idx];
            if(load_balance == 1) {
              send_count += std::max((int)rc_self_[sorted_reactor_idx],1)+load_balance_noise;
            } else if(load_balance == 2) {
              int weighted = (int) (reactor_weight_mult*rg_self_[sorted_reactor_idx]/avg_reactor_time_);
              send_count += std::max(weighted,1);
            }
            send_reactor_idxs.push_back(sorted_reactor_idx);
          }
          size_t send_nreactors = send_reactor_idxs.size();
          comm_mtx_reactors_[j] = send_nreactors; //over-write weighted value
          n_reactors_other_ -= send_nreactors;
          SendReactors(send_reactor_idxs, recv_rank);
        }
      }
    }
  }

  if(n_reactors_other_ > 0) {
    if(int_options_["sort_reactors"]) {
      //To maintain sorting we need to sort again
      // Here we sort by temperature delta if 
      // the constant temperature reactors are enabled
      // so that the GPU is more likely to have a full
      // batch of constant-temperature reactors
      if(int_options_["always_solve_temperature"] != 0) {
        std::vector<double> rc_combined(rc_self_, rc_self_ + n_reactors_self_);
        rc_combined.insert(rc_combined.end(),rc_other_.begin(), rc_other_.end());
        sorted_reactor_idxs_ = sort_indexes(rc_combined);
      } else {
        std::vector<double> td_combined(temp_delta_self_, temp_delta_self_ + n_reactors_self_);
        td_combined.insert(td_combined.end(),temp_delta_other_.begin(), temp_delta_other_.end());
        sorted_reactor_idxs_ = sort_indexes(td_combined);
      }
    } else {
      //Need to resize to full length
      sorted_reactor_idxs_.assign(n_reactors_self_+n_reactors_other_,0);
      std::iota(sorted_reactor_idxs_.begin(), sorted_reactor_idxs_.end(), 0);
    }
  }
  return ZERORK_STATUS_SUCCESS;
}
#endif //USE_MPI

zerork_status_t ZeroRKReactorManager::SolveReactors()
{
  n_calls_++;
  n_gpu_groups_ = 0;
  sum_cpu_reactor_time_ = 0.0;
  sum_gpu_reactor_time_ = 0.0;
  n_steps_cpu_ = 0;
  n_steps_gpu_ = 0;
  n_cpu_solve_ = 0;
  n_cpu_solve_no_temperature_ = 0;
  n_gpu_solve_ = 0;
  n_gpu_solve_no_temperature_ = 0;

  const int always_solve_temp = int_options_["always_solve_temperature"];
  const bool dump_reactors = (int_options_["dump_reactors"]!=0);

  int n_reactors_self_calc = n_reactors_self_ + n_reactors_other_;
  std::vector<int> solved_gpu(n_reactors_self_calc, 0);

  std::vector<double *> T_ptrs(n_reactors_self_calc);
  std::vector<double *> P_ptrs(n_reactors_self_calc);
  std::vector<double *> dpdt_ptrs(n_reactors_self_calc);
  std::vector<double *> e_src_ptrs(n_reactors_self_calc);
  std::vector<double *> y_src_ptrs(n_reactors_self_calc);
  std::vector<double *> rc_ptrs(n_reactors_self_calc);
  std::vector<double *> rg_ptrs(n_reactors_self_calc);
  std::vector<double *> mf_ptrs(n_reactors_self_calc);
  std::vector<double *> root_times_ptrs(n_reactors_self_calc);
  std::vector<double *> temp_delta_ptrs(n_reactors_self_calc);
  std::vector<int *> reactor_id_ptrs(n_reactors_self_calc);
  for(int j = 0; j < n_reactors_self_calc; ++j) {
    int j_sort = sorted_reactor_idxs_[j];
    if(j_sort < n_reactors_self_) {
      T_ptrs[j] = &T_self_[j_sort];
      P_ptrs[j] = &P_self_[j_sort];
      dpdt_ptrs[j] = &dpdt_self_[j_sort];
      if(e_src_defined_) {
        e_src_ptrs[j] = &e_src_self_[j_sort];
      }
      if(y_src_defined_) {
        y_src_ptrs[j] = &y_src_self_[j_sort*num_species_stride_];
      }
      if(reactor_ids_defined_) {
        reactor_id_ptrs[j] = &reactor_ids_self_[j_sort];
      }
      rc_ptrs[j] = &rc_self_[j_sort];
      rg_ptrs[j] = &rg_self_[j_sort];
      root_times_ptrs[j] = &root_times_self_[j_sort];
      temp_delta_ptrs[j] = &temp_delta_self_[j_sort];
      mf_ptrs[j] = &mf_self_[j_sort*num_species_stride_];
    } else {
      j_sort -= n_reactors_self_;
      T_ptrs[j] = &T_other_[j_sort];
      P_ptrs[j] = &P_other_[j_sort];
      dpdt_ptrs[j] = &dpdt_other_[j_sort];
      if(e_src_defined_) {
        e_src_ptrs[j] = &e_src_other_[j_sort];
      }
      if(y_src_defined_) {
        y_src_ptrs[j] = &y_src_other_[j_sort*num_species_];
      }
      if(reactor_ids_defined_) {
        reactor_id_ptrs[j] = &reactor_ids_other_[j_sort];
      }
      rc_ptrs[j] = &rc_other_[j_sort];
      rg_ptrs[j] = &rg_other_[j_sort];
      root_times_ptrs[j] = &root_times_other_[j_sort];
      temp_delta_ptrs[j] = &temp_delta_other_[j_sort];
      mf_ptrs[j] = &mf_other_[j_sort*num_species_];
    }
    if(dump_reactors) {
      DumpReactor("pre", j, *T_ptrs[j], *P_ptrs[j], *rc_ptrs[j], *rg_ptrs[j], mf_ptrs[j]);
    }
  }

  const int verbosity = int_options_["verbosity"];
  const int integrator = int_options_["integrator"];
  const bool constant_volume = (int_options_["constant_volume"] == 1);
  const double solve_temp_threshold = double_options_["solve_temperature_threshold"];
#ifdef ZERORK_GPU
  const int n_reactors_min = int_options_["n_reactors_min"];
  const int n_reactors_max = int_options_["n_reactors_max"];
  if(int_options_["gpu"] != 0 && rank_has_gpu_[rank_]) {
    //Instantiate reactors on first call, after options are set
    if(!reactor_gpu_ptr_) {
      if(constant_volume) {
        reactor_gpu_ptr_ = std::make_unique<ReactorConstantVolumeGPU>(mech_cuda_ptr_);
      } else {
        reactor_gpu_ptr_ = std::make_unique<ReactorConstantPressureGPU>(mech_cuda_ptr_);
      }
    }
    reactor_gpu_ptr_->SetIntOptions(int_options_);
    reactor_gpu_ptr_->SetDoubleOptions(double_options_);

    std::vector<double> T_gpu(n_reactors_max);
    std::vector<double> T_gpu_init(n_reactors_max);
    std::vector<double> P_gpu(n_reactors_max);
    std::vector<double> dpdt_gpu;
    if(dpdt_defined_) dpdt_gpu.resize(n_reactors_max);
    std::vector<double> e_src_gpu;
    if(e_src_defined_) e_src_gpu.resize(n_reactors_max);
    std::vector<double> y_src_gpu;
    if(y_src_defined_) y_src_gpu.resize(num_species_*n_reactors_max);
    std::vector<double> mf_gpu(num_species_*n_reactors_max);
    int n_remaining = n_reactors_self_calc;
    while(n_remaining > 0) {
      int n_curr = 1;
      n_curr = std::min(n_remaining, n_reactors_max);
      if(n_curr != n_remaining && n_remaining < 2*n_reactors_max && n_curr == n_reactors_max) {
        n_curr = n_remaining/2;
      }

      if(n_curr >= n_reactors_min) {
        if(verbosity >= 2) {
            printf("RANK[%d]: Solving GPU group of size = %d.\n", rank_,n_curr);
        }
        n_gpu_groups_++;

        // N.B. n_calls_ logic is a work-around to some memory issue that happens in cuSolverRf
        // it seems like if the smaller matrix is used first, some internal buffer is over-run
        // when we later call the functions with a larger matrix even though we destroy and create
        // a new cuSolverRf handle.
        bool solve_temperature = n_calls_ == 1 ? true : false;
        double start_time = getHighResolutionTime();
        for(int k = 0; k < n_curr; ++k)
        {
          int k_reactor = n_remaining - k - 1;
          int k_reactor_curr = n_curr - k - 1;
          T_gpu[k_reactor_curr] = *T_ptrs[k_reactor];
          T_gpu_init[k_reactor_curr] = *T_ptrs[k_reactor];
          P_gpu[k_reactor_curr] = *P_ptrs[k_reactor];
          if(dpdt_defined_) {
            dpdt_gpu[k_reactor_curr] = *dpdt_ptrs[k_reactor];
          }
          if(e_src_defined_) {
            e_src_gpu[k_reactor_curr] = *e_src_ptrs[k_reactor];
          }
          for(int j = 0; j < num_species_; ++j) {
            //mass fractions will be transposed in reactor InitializeState
            mf_gpu[k_reactor_curr*num_species_+j] = mf_ptrs[k_reactor][j];
            if(y_src_defined_) {
              y_src_gpu[k_reactor_curr*num_species_+j] = y_src_ptrs[k_reactor][j];
            }
          }
          if(*temp_delta_ptrs[k_reactor] > 0.0 || always_solve_temp == 1) {
            solve_temperature = true;
          }
        }

        long int nstep_reactors;
        std::unique_ptr<SolverBase> solver;
        if(integrator == 0) {
          solver.reset(new CvodeSolver(*reactor_gpu_ptr_));
        } else if(integrator == 1) {
          solver.reset(new SeulexSolver(*reactor_gpu_ptr_));
        } else if(integrator == 2) {
          solver.reset(new SodexSolver(*reactor_gpu_ptr_));
        } else if(integrator == 3) {
          solver.reset(new RadauSolver(*reactor_gpu_ptr_));
        } else {
          throw(std::runtime_error("Invalid integrator specified for GPU"));
        }

        solver->SetIntOptions(int_options_);
        solver->SetDoubleOptions(double_options_);
        //if( cb_fn_ != nullptr && int_options_["load_balance"] == 0) {
        //  solver->SetCallbackFunction(cb_fn_, cb_fn_data_);
        //}

        reactor_gpu_ptr_->SetSolveTemperature(solve_temperature);
        reactor_gpu_ptr_->SetIntOption("iterative",solver->Iterative());
        reactor_gpu_ptr_->SetStepLimiter(double_options_["step_limiter"]);

        double* dpdt_ptr = nullptr;
        if(dpdt_defined_) {
          dpdt_ptr = &dpdt_gpu[0];
        }
        double* e_src_ptr = nullptr;
        if(e_src_defined_) {
          e_src_ptr = &e_src_gpu[0];
        }
        double* y_src_ptr = nullptr;
        if(y_src_defined_) {
          y_src_ptr = &y_src_gpu[0];
        }
        reactor_gpu_ptr_->InitializeState(0.0, n_curr, &T_gpu[0], &P_gpu[0],
                                          &mf_gpu[0], dpdt_ptr, e_src_ptr, y_src_ptr);
        nstep_reactors = solver->Integrate(dt_calc_);
        reactor_gpu_ptr_->GetState(dt_calc_, &T_gpu[0], &P_gpu[0], &mf_gpu[0]);

        double reactor_time = getHighResolutionTime() - start_time;
        sum_gpu_reactor_time_ += reactor_time;

        if(nstep_reactors >= 0) {
          n_steps_gpu_ += nstep_reactors*n_curr;
          n_gpu_solve_ += n_curr;
          if(!solve_temperature) n_gpu_solve_no_temperature_ += n_curr;
          for(int k = 0; k < n_curr; ++k) {
            int k_reactor = n_remaining - k - 1;
            int k_reactor_curr = n_curr - k - 1;
            *T_ptrs[k_reactor] = T_gpu[k_reactor_curr];
            *P_ptrs[k_reactor] = P_gpu[k_reactor_curr];
            for(int j = 0; j < num_species_; ++j) {
              mf_ptrs[k_reactor][j] = mf_gpu[k_reactor_curr*num_species_+j];
            }
            solved_gpu[k_reactor] = 1;
            *rc_ptrs[k_reactor] = nstep_reactors;
            *rg_ptrs[k_reactor] = reactor_time/n_curr*gpu_multiplier_;

            double temp_delta = T_gpu[k_reactor_curr] - T_gpu_init[k_reactor_curr];
            if(temp_delta < solve_temp_threshold) temp_delta = 0.0;
            *temp_delta_ptrs[k_reactor] = temp_delta;

            if(dump_reactors) {
              DumpReactor("postg", k_reactor, *T_ptrs[k_reactor], *P_ptrs[k_reactor],
                          *rc_ptrs[k_reactor], *rg_ptrs[k_reactor], mf_ptrs[k_reactor]);
            }
          }
        }
        n_remaining -= n_curr;
      } else {
        break;
      }
    }
  }
#endif //ZERORK_GPU

  //Instantiate reactors on first call, after options are set
  if(!reactor_ptr_) {
    if(constant_volume) {
      reactor_ptr_ = std::make_unique<ReactorConstantVolumeCPU>(mech_ptr_);
    } else {
      reactor_ptr_ = std::make_unique<ReactorConstantPressureCPU>(mech_ptr_);
    }
  }

  reactor_ptr_->SetIntOptions(int_options_);
  reactor_ptr_->SetDoubleOptions(double_options_);

  std::unique_ptr<SolverBase> solver;
  if(integrator == 0) {
    solver.reset(new CvodeSolver(*reactor_ptr_));
  } else if(integrator == 1) {
    solver.reset(new SeulexSolver(*reactor_ptr_));
  } else if(integrator == 2) {
    solver.reset(new SodexSolver(*reactor_ptr_));
  } else if(integrator == 3) {
    solver.reset(new RadauSolver(*reactor_ptr_));
  } else {
    throw(std::runtime_error("Invalid integrator specified"));
  }
  solver->SetIntOptions(int_options_);
  solver->SetDoubleOptions(double_options_);
  if(cb_fn_ != nullptr && int_options_["load_balance"] == 0 && n_reactors_self_calc == 1) {
    solver->SetCallbackFunction(cb_fn_, cb_fn_data_);
  }

  reactor_ptr_->SetIntOption("iterative",solver->Iterative());
  reactor_ptr_->SetStepLimiter(double_options_["step_limiter"]);

  const int dump_failed_reactors = int_options_["dump_failed_reactors"];
  zerork_status_t flag = ZERORK_STATUS_SUCCESS;
  for(int k = 0; k < n_reactors_self_calc; ++k)
  {
    if(solved_gpu[k] == 0) {
      double dpdt_reactor = 0.0;
      if(dpdt_defined_) {
        dpdt_reactor = *dpdt_ptrs[k];
      }
      double e_src_reactor = 0.0;
      if(e_src_defined_) {
        e_src_reactor = *e_src_ptrs[k];
      }
      double* y_src_reactor = nullptr;
      if(y_src_defined_) {
        y_src_reactor = y_src_ptrs[k];
      }
      int reactor_id = k;
      if(reactor_ids_defined_) {
        reactor_id = *reactor_id_ptrs[k];
      }
      bool solve_temperature = false;
      if(*temp_delta_ptrs[k] > 0.0 || always_solve_temp == 1) {
        solve_temperature = true;
      }
      reactor_ptr_->SetSolveTemperature(solve_temperature);
      double T_init = *T_ptrs[k];
      reactor_ptr_->SetID(reactor_id);
      double start_time = getHighResolutionTime();
      reactor_ptr_->InitializeState(0.0, 1, T_ptrs[k], P_ptrs[k],
                                    mf_ptrs[k], &dpdt_reactor,
                                    &e_src_reactor,
                                    y_src_reactor);
      int nsteps = solver->Integrate(dt_calc_);
      double reactor_time = getHighResolutionTime() - start_time;
      if(nsteps < 0) {
        flag = ZERORK_STATUS_FAILED_SOLVE;
        if(dump_failed_reactors!=0) {
          DumpReactor("failed_state", k, *T_ptrs[k], *P_ptrs[k],
                      *rc_ptrs[k], *rg_ptrs[k], mf_ptrs[k]);
        }
      } else {
        reactor_ptr_->GetState(dt_calc_, T_ptrs[k], P_ptrs[k], mf_ptrs[k]);
        *root_times_ptrs[k] = reactor_ptr_->GetRootTime();
        n_steps_cpu_ += nsteps;
        double temp_delta = *T_ptrs[k] - T_init;
        if(temp_delta < solve_temp_threshold) temp_delta = 0.0;
        *temp_delta_ptrs[k] = temp_delta;
      }
      *rc_ptrs[k] = nsteps;
      *rg_ptrs[k] = reactor_time;
      sum_cpu_reactor_time_ += reactor_time;
      ++n_cpu_solve_;
      if(!solve_temperature) ++n_cpu_solve_no_temperature_;
      if(dump_reactors) {
        DumpReactor("postc", k, *T_ptrs[k], *P_ptrs[k],
                    *rc_ptrs[k], *rg_ptrs[k], mf_ptrs[k]);
      }
    }
  }

#ifdef USE_MPI
  int local_flag = (int) flag;
  MPI_Allreduce(&local_flag, &flag, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
  return flag;
}

#ifndef USE_MPI
zerork_status_t ZeroRKReactorManager::RedistributeResults()
{
  return ZERORK_STATUS_SUCCESS;
}
#else
zerork_status_t ZeroRKReactorManager::RedistributeResults()
{
  if(nranks_ == 1 || !int_options_["load_balance"]) {
    return ZERORK_STATUS_SUCCESS;
  }
  //startTime = getHighResolutionTime();
  MPI_Barrier(MPI_COMM_WORLD);
  //double synchTime = getHighResolutionTime() - startTime;

  const int load_balance_mem = int_options_["load_balance_mem"];
  if(n_reactors_other_ > 0) {
    int send_idx = 0;
    for(int i = comm_mtx_row_sum_[sorted_rank_]; i < comm_mtx_row_sum_[sorted_rank_+1]; ++i)
    {
      int sorted_recv_rank = comm_mtx_col_idx_[i];
      int recv_rank = sorted_deficit_idxs_[sorted_recv_rank];
      int send_nreactors = comm_mtx_reactors_[i];
      if(send_nreactors > 0)
      {
        std::vector<double> send_buf(6+num_species_);
        if(load_balance_mem != 0) {
          send_buf.resize((6+num_species_)*send_nreactors);
        }
        int buf_idx = 0;
        for(int k = 0; k < send_nreactors; ++k) {
          if(load_balance_mem == 0) {
            buf_idx = 0;
          }
          send_buf[buf_idx] = rg_other_[send_idx];
          buf_idx += 1;
          send_buf[buf_idx] = rc_other_[send_idx];
          buf_idx += 1;
          send_buf[buf_idx] = T_other_[send_idx];
          buf_idx += 1;
          send_buf[buf_idx] = P_other_[send_idx];
          buf_idx += 1;
          send_buf[buf_idx] = root_times_other_[send_idx];
          buf_idx += 1;
          send_buf[buf_idx] = temp_delta_other_[send_idx];
          buf_idx += 1;
          for(int j = 0; j < num_species_; ++j) {
            send_buf[buf_idx] = mf_other_[send_idx*num_species_+j];
            buf_idx += 1;
          }

          if(load_balance_mem == 0) {
            MPI_Send(&send_buf[0], 6+num_species_, MPI_DOUBLE,
                     recv_rank, EXCHANGE_RETURN_TAG_,MPI_COMM_WORLD);
          }
          send_idx += 1;
        }
        if(load_balance_mem != 0) {
          MPI_Send(&send_buf[0], (6+num_species_)*send_nreactors, MPI_DOUBLE,
                   recv_rank, EXCHANGE_RETURN_TAG_,MPI_COMM_WORLD);
        }
      }
    }
  }

  if(n_reactors_other_ < 0) {
    MPI_Status status;
    int recv_idx = n_reactors_self_;
    for(int ircv = 0; ircv < nranks_; ++ircv) {
      int send_rank = sorted_deficit_idxs_[ircv];
      if(send_rank == rank_) continue;
      for(int j = comm_mtx_row_sum_[ircv]; j < comm_mtx_row_sum_[ircv+1]; ++j) {
        int sorted_recv_rank = comm_mtx_col_idx_[j];
        int recv_rank = sorted_deficit_idxs_[sorted_recv_rank];
        if(recv_rank == rank_) {
          int recv_nreactors = comm_mtx_reactors_[j];
          if(recv_nreactors > 0) {
            std::vector<double> recv_buf(6+num_species_);
            if(load_balance_mem != 0) {
              recv_buf.resize((6+num_species_)*recv_nreactors);
              MPI_Recv(&recv_buf[0], (6+num_species_)*recv_nreactors, MPI_DOUBLE,
                       send_rank, EXCHANGE_RETURN_TAG_, MPI_COMM_WORLD,
                       &status);
            }
            int buf_idx = 0;
            for(int k = 0; k < recv_nreactors; ++k) {
              recv_idx -= 1;
              size_t sorted_recv_idx = sorted_reactor_idxs_[recv_idx];
              if(load_balance_mem == 0) {
                MPI_Recv(&recv_buf[0], 6+num_species_, MPI_DOUBLE,
                         send_rank, EXCHANGE_RETURN_TAG_, MPI_COMM_WORLD,
                         &status);

                buf_idx = 0;
              }
              rg_self_[sorted_recv_idx] = recv_buf[buf_idx];
              buf_idx += 1;
              rc_self_[sorted_recv_idx] = recv_buf[buf_idx];
              buf_idx += 1;
              T_self_[sorted_recv_idx] = recv_buf[buf_idx];
              buf_idx += 1;
              P_self_[sorted_recv_idx] = recv_buf[buf_idx];
              buf_idx += 1;
              root_times_self_[sorted_recv_idx] = recv_buf[buf_idx];
              buf_idx += 1;
              temp_delta_self_[sorted_recv_idx] = recv_buf[buf_idx];
              buf_idx += 1;
              for(int kk = 0; kk < num_species_; ++kk) {
                mf_self_[sorted_recv_idx*num_species_stride_+kk] = recv_buf[buf_idx];
                buf_idx += 1;
              }
            }
          }
        }
      }
    }
  }
  //commTime += getHighResolutionTime() - startTime;
  return ZERORK_STATUS_SUCCESS;
}
#endif

zerork_status_t ZeroRKReactorManager::PostSolve() {
  if(int_options_["output_performance_log"]!=0) ProcessPerformance();
#ifdef ZERORK_GPU
  UpdateRankWeights();
#endif
  return ZERORK_STATUS_SUCCESS;
}

void ZeroRKReactorManager::ProcessPerformance()
{
  double all_time = sum_cpu_reactor_time_ + sum_gpu_reactor_time_;
  double max_cpu_reactor_time = sum_cpu_reactor_time_;
  double max_gpu_reactor_time = sum_gpu_reactor_time_;

  n_reactors_solved_ranks_[rank_] = n_cpu_solve_ + n_gpu_solve_;
  all_time_ranks_[rank_] = all_time;
#ifdef USE_MPI
  if(nranks_ > 1) {
    int n_total_solved = n_cpu_solve_ + n_gpu_solve_;
    MPI_Gather(&n_total_solved,1,MPI_INT,&n_reactors_solved_ranks_[0],1,MPI_INT,root_rank_,MPI_COMM_WORLD);
    MPI_Gather(&all_time,1,MPI_DOUBLE,&all_time_ranks_[0],1,MPI_DOUBLE,root_rank_,MPI_COMM_WORLD);
    // Collect timing/step count data
    double rr; //reduced real
    int ri; //reduced int

    MPI_Reduce(&n_cpu_solve_,&ri,1,MPI_INT,MPI_SUM,root_rank_,MPI_COMM_WORLD);
    if(rank_ == root_rank_) n_cpu_solve_ = ri;
    MPI_Reduce(&n_cpu_solve_no_temperature_,&ri,1,MPI_INT,MPI_SUM,root_rank_,MPI_COMM_WORLD);
    if(rank_ == root_rank_) n_cpu_solve_no_temperature_ = ri;
    MPI_Reduce(&n_gpu_solve_,&ri,1,MPI_INT,MPI_SUM,root_rank_,MPI_COMM_WORLD);
    if(rank_ == root_rank_) n_gpu_solve_ = ri;
    MPI_Reduce(&n_gpu_solve_no_temperature_,&ri,1,MPI_INT,MPI_SUM,root_rank_,MPI_COMM_WORLD);
    if(rank_ == root_rank_) n_gpu_solve_no_temperature_ = ri;
    MPI_Reduce(&n_gpu_groups_,&ri,1,MPI_INT,MPI_SUM,root_rank_,MPI_COMM_WORLD);
    if(rank_ == root_rank_) n_gpu_groups_ = ri;

    MPI_Reduce(&n_steps_cpu_,&ri,1,MPI_INT,MPI_SUM,root_rank_,MPI_COMM_WORLD);
    if(rank_ == root_rank_) n_steps_cpu_ = ri;
    MPI_Reduce(&n_steps_gpu_,&ri,1,MPI_INT,MPI_SUM,root_rank_,MPI_COMM_WORLD);
    if(rank_ == root_rank_) n_steps_gpu_ = ri;

    //Get max time for cpu and gpu
    MPI_Reduce(&sum_cpu_reactor_time_,&rr,1,MPI_DOUBLE,MPI_MAX,root_rank_,MPI_COMM_WORLD);
    if(rank_ == root_rank_) max_cpu_reactor_time = rr;
    MPI_Reduce(&sum_gpu_reactor_time_,&rr,1,MPI_DOUBLE,MPI_MAX,root_rank_,MPI_COMM_WORLD);
    if(rank_ == root_rank_) max_gpu_reactor_time = rr;
    //Calc per step times based on sum of times
    MPI_Reduce(&sum_cpu_reactor_time_,&rr,1,MPI_DOUBLE,MPI_SUM,root_rank_,MPI_COMM_WORLD);
    if(rank_ == root_rank_) sum_cpu_reactor_time_ = rr;
    MPI_Reduce(&sum_gpu_reactor_time_,&rr,1,MPI_DOUBLE,MPI_SUM,root_rank_,MPI_COMM_WORLD);
    if(rank_ == root_rank_) sum_gpu_reactor_time_ = rr;
  } else {
    max_cpu_reactor_time = sum_cpu_reactor_time_;
    max_gpu_reactor_time = sum_gpu_reactor_time_;
  }
#else
  max_cpu_reactor_time = sum_cpu_reactor_time_;
  max_gpu_reactor_time = sum_gpu_reactor_time_;
#endif
  if(rank_ == root_rank_) {
    int n_steps_total = n_steps_cpu_ + n_steps_gpu_;
    int n_total_solved = n_cpu_solve_ + n_gpu_solve_;
    double nstep_avg = n_total_solved > 0 ? n_steps_total/n_total_solved : 0;

    double nstep_avg_gpu = n_gpu_solve_ > 0 ? n_steps_gpu_/n_gpu_solve_ : 0;
    double nstep_avg_cpu = n_cpu_solve_ > 0 ? n_steps_cpu_/n_cpu_solve_ : 0;

    double cpu_per_step_time = n_cpu_solve_ > 0 ? sum_cpu_reactor_time_/n_steps_cpu_ : 0;
    double gpu_per_step_time = n_gpu_solve_ > 0 ? sum_gpu_reactor_time_/n_steps_gpu_ : 0;

    double avg_time = 0.0;
    double max_time = 0.0;
    double total_time = 0.0;
    for(int i = 0; i < nranks_; ++i) {
       max_time = std::max(max_time,all_time_ranks_[i]);
       total_time += all_time_ranks_[i];
    }
    avg_time = total_time/nranks_;
    avg_reactor_time_ = n_total_solved > 0 ? avg_time/n_total_solved : 1.0;

    //Print stats to file
    //reactor_log_file_ << std::setprecision(16);
    reactor_log_file_ << std::setw(13) <<  n_cycle_;
    reactor_log_file_ << std::setw(17) <<  n_total_solved;
    reactor_log_file_ << std::setw(17) <<  n_cpu_solve_;
    reactor_log_file_ << std::setw(17) <<  n_cpu_solve_no_temperature_;
    reactor_log_file_ << std::setw(17) <<  n_gpu_solve_;
    reactor_log_file_ << std::setw(17) <<  n_gpu_solve_no_temperature_;
    reactor_log_file_ << std::setw(17) <<  n_gpu_groups_;
    reactor_log_file_ << std::setw(17) <<  nstep_avg;
    reactor_log_file_ << std::setw(17) <<  nstep_avg_cpu;
    reactor_log_file_ << std::setw(17) <<  nstep_avg_gpu;
    reactor_log_file_ << std::setw(17) <<  max_cpu_reactor_time;
    reactor_log_file_ << std::setw(17) <<  max_gpu_reactor_time;
    reactor_log_file_ << std::setw(17) <<  cpu_per_step_time;
    reactor_log_file_ << std::setw(17) <<  gpu_per_step_time;
    reactor_log_file_ << std::setw(17) <<  avg_time;
    reactor_log_file_ << std::setw(17) <<  max_time;
    reactor_log_file_ << std::endl;
    reactor_log_file_.flush();

    if(int_options_["verbosity"] > 0) {
      if(int_options_["load_balance"]) {
        double wasted_time = max_time - avg_time;
        for(int i = 0; i < nranks_; ++i) {
          printf("Rank %d calculated %d reactors in %f seconds.\n",i,n_reactors_solved_ranks_[i],all_time_ranks_[i]);
        }
        printf("Max Time, Avg Time, Wasted Time = %f, %f, %f\n",max_time,avg_time,wasted_time);
      } else {
        printf("Rank %d calculated %d reactors in %f seconds.\n",rank_,n_cpu_solve_+n_gpu_solve_,all_time_ranks_[rank_]);
      }
    }
  }
#ifdef USE_MPI
  if(int_options_["load_balance"] == 2) {
    MPI_Bcast(&avg_reactor_time_,1,MPI_DOUBLE,root_rank_,MPI_COMM_WORLD);
  }
#endif
  //commTime += getHighResolutionTime() - startTime;
}


#ifdef ZERORK_GPU
void ZeroRKReactorManager::UpdateRankWeights() {
  if(nranks_ == 1 || int_options_["load_balance"] == 0) {
    return;
  }
#ifdef USE_MPI
  //Update rank_weights_
  if(rank_==root_rank_) {
    if(int_options_["gpu"] != 0 && n_weight_updates_ % 10 == 0 && n_gpu_solve_ == 0) {
      double sum_weights = 0.0;
      std::vector<double> test_weights(nranks_,1.0);
      const int initial_gpu_multiplier = int_options_["initial_gpu_multiplier"];
      for(int i = 0; i < nranks_; ++i) {
        if(rank_has_gpu_[i]) test_weights[i] *= initial_gpu_multiplier;
        sum_weights += test_weights[i];
      }
      double weight_factor = nranks_/sum_weights;
      sum_weights=0.0;
      for(int i = 0; i < nranks_; ++i) {
        test_weights[i] *= weight_factor;
        sum_weights += test_weights[i];
      }
      weight_factor = (n_cpu_solve_ + n_gpu_solve_)/sum_weights;
      bool override_weights = false;
      std::vector<int> reactors_wanted(nranks_);
      const int n_reactors_min = int_options_["n_reactors_min"];
      for(int i = 0; i < nranks_; ++i) {
        reactors_wanted[i] = test_weights[i]*weight_factor;
        if(rank_has_gpu_[i] &&
           reactors_wanted[i] > n_reactors_min &&
           test_weights[i] > rank_weights_[i]) {
          override_weights=true;
          break;
        }
      }
      if(override_weights) {
        for(int i = 0; i < nranks_; ++i) {
          rank_weights_[i] = test_weights[i];
        }
      }
    }
    if(n_gpu_solve_ > 0 && n_cpu_ranks_ > 0) {
      //double gpu_ratio = (sum_cpu_reactor_time_/n_cpu_ranks_) / (sum_gpu_reactor_time_/n_gpu_ranks_);
      double sum_cpu_rank_time = 0.0;
      double sum_gpu_rank_time = 0.0;
      for(int i = 0; i < nranks_; ++i) {
        if(rank_has_gpu_[i]) {
          sum_gpu_rank_time += all_time_ranks_[i];
        } else {
          sum_cpu_rank_time += all_time_ranks_[i];
        }
      }
      double gpu_ratio = (sum_cpu_rank_time/n_cpu_ranks_) / (sum_gpu_rank_time/n_gpu_ranks_);

      gpu_multiplier_ = 0.5*gpu_multiplier_ + 0.5*gpu_ratio*gpu_multiplier_;

      double sum_weights = 0.0;
      for(int i = 0; i < nranks_; ++i) {
        if(rank_has_gpu_[i]) {
          rank_weights_[i] = gpu_multiplier_;
        } else {
          rank_weights_[i] = 1.0;
        }
        sum_weights += rank_weights_[i];
      }

      double weight_factor = nranks_/sum_weights;
      for(int i = 0 ; i < nranks_; ++i) {
        rank_weights_[i] *= weight_factor;
      }
    }
  }

  MPI_Bcast(&gpu_multiplier_,1,MPI_DOUBLE,root_rank_,MPI_COMM_WORLD);
  MPI_Bcast(&rank_weights_[0],nranks_,MPI_DOUBLE,root_rank_,MPI_COMM_WORLD);
  n_weight_updates_ += 1;
#endif
}
#endif

void ZeroRKReactorManager::DumpReactor(std::string tag, int id, double T, double P,
                                       double rc, double rg, double* mf) {
      std::ofstream dump_file;
      std::ostringstream dump_file_name;
      dump_file_name << std::setfill('0') << "dumpfile_" << tag
                     << "_" << std::setw(6)
                     << n_calls_ << "_" << std::setw(6)
                     << rank_;// << "_" << std::setw(6) << id;
      dump_file.open(dump_file_name.str(), std::ofstream::out | std::ofstream::app);
      dump_file << id << "\t";
      dump_file << std::setprecision(20);
      dump_file << T << "\t";
      dump_file << P << "\t";
      dump_file << rc << "\t";
      dump_file << rg << "\t";
      for(int k = 0 ; k < num_species_; ++k) {
        dump_file << mf[k] << "\t";
      }
      dump_file << "\n";
      dump_file.close();
}


