#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <fstream>
#include <utilities/string_utilities.h>
#include <utilities/math_utilities.h>
#include <utilities/file_utilities.h>

#include "flame_params.h"

// Get scalar dissipation rate
static double GetDissipationRate(double mixture_fraction);

// Get scalar dissipation rate for modified flamelet equations
static double GetDissipationRateYSI(double mixture_fraction);

static double NormalizeComposition(const size_t num_elements,
                                   double composition[]);

FlameParams::FlameParams(const std::string &input_name, MPI_Comm &comm)
{

  comm_ = comm;
  MPI_Comm_size(comm_, &npes_);
  MPI_Comm_rank(comm_, &my_pe_);

  int error_code;

  parser_  = NULL;
  reactor_ = NULL;
  trans_   = NULL;
  logger_  = NULL;
  sparse_matrix_.clear();
  sparse_matrix_dist_ = NULL;
  valid_jacobian_structure_ = true;

  if(!zerork::utilities::FileIsReadable(input_name)) {
    printf("# ERROR: Input file %s is not readable\n",input_name.c_str());
    exit(-1);
  }

  // setup parser
  input_name_ = input_name;
  parser_ = new UnsteadyFlameIFP(input_name);
  if(parser_ == NULL) {
    printf("# ERROR: Parser for file %s is not created\n",input_name.c_str());
    exit(-1);
  }

  // setup constant pressure reactor
  reactor_ = new ConstPressureReactor(parser_->mech_file().c_str(),
                                      parser_->therm_file().c_str(),
                                      parser_->log_file().c_str(),
                                      COMPRESSED_COL_STORAGE,
                                      parser_->pressure());
  if(reactor_ == NULL) {
    printf("# ERROR: Could not create ConstPressureReactor for files:\n"
           "#            mechanism      file = %s\n"
           "#            thermodynamics file = %s\n"
           "#            log            file = %s\n",
           parser_->mech_file().c_str(),
           parser_->therm_file().c_str(),
           parser_->log_file().c_str());
    exit(-1);
  }
  reactor_->SetReferenceTemperature(parser_->ref_temperature());

  mechanism_ = reactor_->GetMechanism();

  // setup transport interface
  trans_ = transport::InterfaceFactory::CreateMassBased(parser_->transport_model());

  std::vector<std::string> transport_files;
  transport_files.push_back(parser_->trans_file());

  error_code = trans_->Initialize(mechanism_,
                                  transport_files,
                                  parser_->log_file());
  if(error_code != transport::NO_ERROR) {
    printf("# ERROR: Could not Initialize MassTransportInterface for files:\n"
           "#            mechanism      file = %s\n"
           "#            thermodynamics file = %s\n"
           "#            transport      file = %s\n"
           "#            log            file = %s\n"
           "#        Initialize returned error code = %d\n",
           parser_->mech_file().c_str(),
           parser_->therm_file().c_str(),
           parser_->trans_file().c_str(),
           parser_->log_file().c_str(),
           error_code);
    exit(-1);
  }
  // setup logger
  logger_ = new zerork::utilities::Logger(parser_->log_file());
  if(logger_ == NULL) {
    printf("# ERROR: Could not create new logger for file %s\n",
           parser_->log_file().c_str());
    exit(-1);
  }

  // Set inlet values
  SetInlet();
  // Set grid
  SetGrid();
  // Set properties for fixed temperature cases
  SetFixedTProperties();
  // Set arrays
  SetMemory();

  logger_->FFlush();

}

FlameParams::~FlameParams()
{
  if(parser_ != NULL) {
    delete parser_;
  }
  if(reactor_ != NULL) {
    delete reactor_;
  }
  if(trans_ != NULL) {
    delete trans_;
  }
  if(logger_ != NULL) {
    delete logger_;
  }
  if(transport_input_.mass_fraction_ != NULL) {
    delete [] transport_input_.mass_fraction_;
  }
  if(transport_input_.grad_temperature_ != NULL) {
    delete [] transport_input_.grad_temperature_;
  }
  if(transport_input_.grad_pressure_ != NULL) {
    delete [] transport_input_.grad_pressure_;
  }
  if(transport_input_.grad_mass_fraction_ != NULL) {
    delete [] transport_input_.grad_mass_fraction_;
  }
  if(integrator_type_ == 2) {
    if (sparse_matrix_dist_ != NULL) {
      delete sparse_matrix_dist_;
    }
  }

  if(integrator_type_ == 3) {
    for(size_t j=0; j<sparse_matrix_.size(); ++j) {
      if(sparse_matrix_[j] != NULL) {
        delete sparse_matrix_[j];
      }
    }
  }
}

// Uses parser_->inlet_fuel_comp(),
//      parser_->inlet_oxidizer_comp(),
void FlameParams::SetInlet()
{
  const int num_species = reactor_->GetNumSpecies();
  std::map<std::string, double>::const_iterator iter;
  std::vector<double> stoichiometric_mole_fractions;
  std::vector<double> exhaust_mole_fractions;
  std::vector<double> exhaust_mass_fractions;
  std::vector<double> fuel_mole_fractions;
  std::vector<double> oxidizer_mole_fractions;
  std::vector<double> molecular_mass;
  double fuel_fraction;
  double phi_term;
  double mole_fraction_sum;

  fuel_mole_fractions.assign(num_species, 0.0);
  oxidizer_mole_fractions.assign(num_species, 0.0);
  stoichiometric_mole_fractions.assign(num_species, 0.0);
  molecular_mass.assign(num_species, 0.0);

  fuel_mass_fractions_.assign(num_species, 0.0);
  oxidizer_mass_fractions_.assign(num_species, 0.0);
  stoichiometric_mass_fractions_.assign(num_species, 0.0);

  exhaust_mole_fractions.assign(num_species, 0.0);
  exhaust_mass_fractions.assign(num_species, 0.0);

  reactor_->GetSpeciesMolecularWeight(&molecular_mass[0]);

  // Get fuel composition
  if(parser_->inlet_fuel_comp().size() == 0) {
    printf("# ERROR: inlet_fuel_comp().size() is zero.\n");
    exit(-1);
  }

  logger_->PrintF("# From inlet_fuel_comp map in %s\n",
                  input_name_.c_str());
  logger_->PrintF("#   Species Name             Mole Fraction\n");

  mole_fraction_sum = 0.0;
  fuel_molecular_mass_ = 0.0;
  for(iter=parser_->inlet_fuel_comp().begin();
      iter != parser_->inlet_fuel_comp().end();
      ++iter) {

    std::string species_name=iter->first;
    std::string state_name = std::string("MassFraction_")+species_name;
    double mole_fraction = iter->second;
    int species_id = reactor_->GetIdOfState(state_name.c_str());
    logger_->PrintF("%16s  %24.18e\n",
                    species_name.c_str(),
                    mole_fraction);

    if(0 <= species_id  && species_id < num_species) {
      fuel_species_id_.push_back(species_id);
      fuel_mole_fractions[species_id] = mole_fraction;
      mole_fraction_sum += mole_fraction;
      fuel_molecular_mass_ += mole_fraction*molecular_mass[species_id];
    } else {
      printf("# ERROR: did not find species %s in the mechanism\n",
             species_name.c_str());
      exit(-1);
    }
  }
  // re-normalize fuel mole fractions
  NormalizeComposition(num_species,&fuel_mole_fractions[0]);

  // Compute fuel mass fractions
  for(int j=0; j<num_species; ++j) {
    fuel_mass_fractions_[j] =
      fuel_mole_fractions[j]*molecular_mass[j]/fuel_molecular_mass_;
  }
  // renormalize fuel mass fractions
  NormalizeComposition(num_species,&fuel_mass_fractions_[0]);

  // Get oxidizer composition
  if(parser_->inlet_oxidizer_comp().size() == 0) {
    printf("# ERROR: inlet_oxidizer_comp().size() is zero.\n");
    exit(-1);
  }

  logger_->PrintF("# From inlet_oxidizer_comp map in %s\n",
                  input_name_.c_str());
  logger_->PrintF("#   Species Name             Mole Fraction\n");

  mole_fraction_sum = 0.0;
  oxidizer_molecular_mass_ = 0.0;
  for(iter=parser_->inlet_oxidizer_comp().begin();
      iter != parser_->inlet_oxidizer_comp().end();
      ++iter) {

    std::string species_name=iter->first;
    std::string state_name = std::string("MassFraction_")+species_name;
    double mole_fraction = iter->second;
    int species_id = reactor_->GetIdOfState(state_name.c_str());
    logger_->PrintF("%16s  %24.18e\n",
                    species_name.c_str(),
                    mole_fraction);

    if(0 <= species_id  && species_id <num_species) {
      oxidizer_species_id_.push_back(species_id);
      oxidizer_mole_fractions[species_id] = mole_fraction;
      mole_fraction_sum += mole_fraction;
      oxidizer_molecular_mass_ += mole_fraction*molecular_mass[species_id];
    } else {
      printf("# ERROR: did not find species %s in the mechanism\n",
             species_name.c_str());
      exit(-1);
    }
  }
  // re-normalize oxidizer mole fractions
  NormalizeComposition(num_species,&oxidizer_mole_fractions[0]);

  // Compute oxidizer mass fractions
  for(int j=0; j<num_species; ++j) {
    oxidizer_mass_fractions_[j] =
      oxidizer_mole_fractions[j]*molecular_mass[j]/oxidizer_molecular_mass_;
  }
  // renormalize oxidizer mass fractions
  NormalizeComposition(num_species,&oxidizer_mass_fractions_[0]);

  // compute the excess atomic oxygen of the fuel and oxidizer stream
  double fuel_atomic_oxygen_sum, oxidizer_atomic_oxygen_sum;
  double excess_atomic_oxygen;
  std::vector<int> num_hydrogen, num_carbon, num_oxygen;
  num_hydrogen.assign(num_species, 0);
  num_carbon.assign(num_species, 0);
  num_oxygen.assign(num_species, 0);

  reactor_->GetSpeciesHydrogenCount(&num_hydrogen[0]);
  reactor_->GetSpeciesCarbonCount(&num_carbon[0]);
  reactor_->GetSpeciesOxygenCount(&num_oxygen[0]);

  fuel_atomic_oxygen_sum = oxidizer_atomic_oxygen_sum = 0.0;
  for(int j=0; j<num_species; ++j) {

    excess_atomic_oxygen =
      -2.0*num_carbon[j] - 0.5*num_hydrogen[j] + 1.0*num_oxygen[j];

    fuel_atomic_oxygen_sum     +=
      fuel_mole_fractions[j]*excess_atomic_oxygen;

    oxidizer_atomic_oxygen_sum +=
      oxidizer_mole_fractions[j]*excess_atomic_oxygen;
  }

  logger_->PrintF(
    "# Moles of excess atomic oxygen/mole of fuel    :  %24.18e\n",
    fuel_atomic_oxygen_sum);
  logger_->PrintF(
    "# Moles of excess atomic oxygen/mole of oxidizer:  %24.18e\n",
    oxidizer_atomic_oxygen_sum);

  if(fabs(fuel_atomic_oxygen_sum-oxidizer_atomic_oxygen_sum) < 1.0e-300) {

    printf("# ERROR: can not balance fuel and oxidizer compositions\n"
           "#        because the excess atomic oxygen is the same\n");
    exit(-1);
  }

  phi_term =
    -oxidizer_atomic_oxygen_sum/fuel_atomic_oxygen_sum; //*parser_->inlet_phi();

  fuel_fraction = phi_term/(1.0+phi_term);

  if(0.0 > fuel_fraction || 1.0 < fuel_fraction) {
    printf("# ERROR: can not balance fuel and oxidizer compositions\n"
           "#        because the fuel fraction is out-of-bounds:\n"
           "#            fuel     excess atomic oxygen = %24.18e\n"
           "#            oxidizer excess atomic oxygen = %24.18e\n"
           "#            fuel fraction = %24.18e\n",
           fuel_atomic_oxygen_sum,
           oxidizer_atomic_oxygen_sum,
           fuel_fraction);
    exit(-1);
  }

  stoichiometric_molecular_mass_ = 0.0;
  for(int j=0; j<num_species; ++j) {

    stoichiometric_mole_fractions[j] =
      fuel_fraction*fuel_mole_fractions[j] +
      (1.0-fuel_fraction)*oxidizer_mole_fractions[j];

    stoichiometric_molecular_mass_ += molecular_mass[j]*stoichiometric_mole_fractions[j];
  }
  // renormalize inlet mole fractions
  NormalizeComposition(num_species,&stoichiometric_mole_fractions[0]);

  // Compute stoichiometric mass fractions
  stoichiometric_mixture_fraction_ = 0.0;
  for(int j=0; j<num_species; ++j) {
    stoichiometric_mass_fractions_[j] =
      fuel_fraction*fuel_mole_fractions[j]*molecular_mass[j]/stoichiometric_molecular_mass_;
    stoichiometric_mixture_fraction_ += stoichiometric_mass_fractions_[j];
  }
  // Don't renormalize

  // Get scalar dissipation rate
  scalar_dissipation_rate_ = parser_->scalar_dissipation_rate();

  // Add EGR
  double egr = parser_->egr();
  if(egr > 0.0) {
    cerr << " EGR NOT IMPLEMENTED. \n";
  }

  ref_temperature_ =  parser_->ref_temperature();

  fuel_temperature_ = parser_->fuel_temperature()/ref_temperature_;
  oxidizer_temperature_ = parser_->oxidizer_temperature()/ref_temperature_;

  fuel_relative_volume_ =
    reactor_->GetGasConstant()*parser_->fuel_temperature()/
    (parser_->pressure()*fuel_molecular_mass_);

  oxidizer_relative_volume_ =
    reactor_->GetGasConstant()*parser_->oxidizer_temperature()/
    (parser_->pressure()*oxidizer_molecular_mass_);

} // void FlameParams::SetInlet()


void FlameParams::SetGrid()
{
  if(parser_->grid_file() == std::string(zerork::utilities::null_filename)) {
    // if no grid provided -> uniform grid from input file parameters
    if(parser_->num_points() < 2) {
      printf("# ERROR: number of grid points must be two or greater than two\n"
	     "#            num_points = %d\n", parser_->num_points());
      exit(-1);
    }

    const int num_points = parser_->num_points();
    const double delta_z = 1.0/(double)(num_points+2-1);
    z_.assign(num_points, 0.0);
    for(int j=0; j<num_points; ++j) {
      z_[j] = delta_z + (double)j*delta_z;
      //z_[j] = (double)j/(double)(num_points-1);
    }

  } else {
    // read grid from file
    // Right now all processors are doing the same read here
    std::ifstream grid_file;
    std::string line;
    std::string delimiters = std::string(zerork::utilities::WHITESPACE) + ",";
    std::vector<std::string> fields;

    grid_file.open(parser_->grid_file().c_str());

    if(grid_file) {
      while(zerork::utilities::GetAnyLine(grid_file,&line)) {
	zerork::utilities::SplitStringToVector(line,
                                                delimiters,
                                                &fields);
        if(fields.size() == 1) {
          if(zerork::utilities::StringIsDouble(fields[0])) {
            z_.push_back(atof(fields[0].c_str()));
          }
        }
      }

    } else {
      printf("# ERROR: could not open grid file = %s\n",
             parser_->grid_file().c_str());
      exit(-1);
    }//if file

    if(z_.size() == 0) {
      printf("# ERROR: no position data found in\n"
             "#        grid file = %s\n",
             parser_->grid_file().c_str());
      exit(-1);
    } // if z.size

    // Remove first and last points (Z=0 and Z=1)
    z_.erase(z_.begin());
    z_.pop_back();
  }

  const int num_points = z_.size();
  num_local_points_ = num_points/npes_;
  num_points_ = num_points;

  if(num_points % npes_ != 0 ) {
    printf("Number of grid points not divisible by number of processors \n");
    MPI_Finalize();
    exit(-1);
  }

  // Compute midpoints and spacings
  zm_.assign(num_points, 0.0);
  dz_.assign(num_points, 1.0);
  dzm_.assign(num_points, 1.0);

  nover_ = 2;
  const int nover = nover_;
  if(num_local_points_ < nover ) {
    printf("Need at least two grid points per processor for second order discretization \n");
    MPI_Finalize();
    exit(-1);
  }

  MPI_Status status;
  dz_local_.assign( num_local_points_+(2*nover), 0.0);
  dzm_local_.assign( num_local_points_+(2*nover), 0.0);
  inv_dz_local_.assign( num_local_points_+(2*nover), 0.0);
  inv_dzm_local_.assign( num_local_points_+(2*nover), 0.0);

  // Compute midpoints and grid spacings
  for(int j=1; j<num_points; ++j) {
    dz_[j] = z_[j]-z_[j-1];
    zm_[j] = 0.5*(z_[j]+z_[j-1]);
  }
  dz_[0] = z_[0];  //hack
  zm_[0] = z_[0]-0.5*dz_[0];
  for(int j=0; j<num_points-1; ++j) {
    dzm_[j] = zm_[j+1]-zm_[j];
  }
  dzm_[num_points-1] = dz_[num_points-1];

  for (int j=0; j<num_local_points_; ++j) {
    int jglobal = j + my_pe_*num_local_points_;
    dz_local_[nover+j] = dz_[jglobal];
    dzm_local_[nover+j] = dzm_[jglobal];
    inv_dz_local_[nover+j] = 1.0/dz_[jglobal];
    inv_dzm_local_[nover+j] = 1.0/dzm_[jglobal];
  }

  // Send left and right
  if (my_pe_ !=0) {
    MPI_Send(&dz_local_[nover], nover, MPI_DOUBLE, my_pe_-1, 0, comm_);
    MPI_Send(&dzm_local_[nover], nover, MPI_DOUBLE, my_pe_-1, 0, comm_);
    MPI_Send(&inv_dz_local_[nover], nover, MPI_DOUBLE, my_pe_-1, 0, comm_);
    MPI_Send(&inv_dzm_local_[nover], nover, MPI_DOUBLE, my_pe_-1, 0, comm_);
  }
  if (my_pe_ != npes_-1) {
    MPI_Send(&dz_local_[num_local_points_], nover, MPI_DOUBLE, my_pe_+1, 0, comm_);
    MPI_Send(&dzm_local_[num_local_points_], nover, MPI_DOUBLE, my_pe_+1, 0, comm_);
    MPI_Send(&inv_dz_local_[num_local_points_], nover, MPI_DOUBLE, my_pe_+1, 0, comm_);
    MPI_Send(&inv_dzm_local_[num_local_points_], nover, MPI_DOUBLE, my_pe_+1, 0, comm_);
  }

  if (my_pe_ !=0) {
    MPI_Recv(&dz_local_[0], nover, MPI_DOUBLE, my_pe_-1, 0, comm_, &status);
    MPI_Recv(&dzm_local_[0], nover, MPI_DOUBLE, my_pe_-1, 0, comm_, &status);
    MPI_Recv(&inv_dz_local_[0], nover, MPI_DOUBLE, my_pe_-1, 0, comm_, &status);
    MPI_Recv(&inv_dzm_local_[0], nover, MPI_DOUBLE, my_pe_-1, 0, comm_, &status);
  }
  if (my_pe_ != npes_-1) {
    MPI_Recv(&dz_local_[num_local_points_+nover], nover, MPI_DOUBLE, my_pe_+1, 0, comm_, &status);
    MPI_Recv(&dzm_local_[num_local_points_+nover], nover, MPI_DOUBLE, my_pe_+1, 0, comm_, &status);
    MPI_Recv(&inv_dz_local_[num_local_points_+nover], nover, MPI_DOUBLE, my_pe_+1, 0, comm_, &status);
    MPI_Recv(&inv_dzm_local_[num_local_points_+nover], nover, MPI_DOUBLE, my_pe_+1, 0, comm_, &status);
  }

  if(my_pe_ == 0) {
    for(int j=0; j<nover; ++j) {
      dz_local_[j] = dz_[0];
      dzm_local_[j] = dzm_[0];
      inv_dz_local_[j] = 1.0/dz_[0];
      inv_dzm_local_[j] = 1.0/dzm_[0];
    }
  }

  if(my_pe_ == npes_-1) {
    for(int j=num_local_points_+nover; j<num_local_points_+2*nover; ++j) {
      dz_local_[j] = dz_[num_points-1];
      dzm_local_[j] = dzm_[num_points-1];
      inv_dz_local_[j] = 1.0/dz_[num_points-1];
      inv_dzm_local_[j] = 1.0/dzm_[num_points-1];
    }
  }

}

void FlameParams::SetFixedTProperties()
{
  const int num_points = z_.size();
  std::vector<double> file_position, file_fixed_temperature;

  fixed_temperature_.clear();
  fixed_temperature_.assign(num_points,
                           parser_->fuel_temperature()/
                           parser_->ref_temperature());

  if(parser_->fixed_temperature_file() == std::string(zerork::utilities::null_filename)) {
    // if no temperature file is specified, the temperature equation
    // is used to compute the temperature
    fix_temperature_ = false;
  } else {
    fix_temperature_ = true;
    std::ifstream fixed_temperature_file;
    std::string line;
    std::string delimiters = std::string(zerork::utilities::WHITESPACE) + ",";
    std::vector<std::string> fields;
    fixed_temperature_file.open(parser_->fixed_temperature_file().c_str());

    if(fixed_temperature_file) {
      while(zerork::utilities::GetAnyLine(fixed_temperature_file,&line)) {
        // remove everything before the comment character
	zerork::utilities::SplitBeforeDelimiter(line,
                                        "!",
                                        true,
                                        &line);
	zerork::utilities::SplitStringToVector(line,
                                       delimiters,
                                       &fields);
        // only record valid data lines
        if(fields.size() == 2) {
          if(zerork::utilities::StringIsDouble(fields[0]) &&
             zerork::utilities::StringIsDouble(fields[1])) {

            file_position.push_back(atof(fields[0].c_str()));
            file_fixed_temperature.push_back(atof(fields[1].c_str()));
          }
        }
      } // while(zerork::utilities::GetAnyLine(fixed_temperature_file,&line))

    } else {
      printf("# ERROR: could not open fixed_temperature_file = %s\n",
             parser_->fixed_temperature_file().c_str());
      exit(-1);
    }

    if(file_position.size() == 0) {
      printf("# ERROR: no position and temperature data found in\n"
             "#        fixed_temperature_file = %s\n",
             parser_->fixed_temperature_file().c_str());
      exit(-1);
    }

    zerork::utilities::SortVectors(&file_position, &file_fixed_temperature);
    logger_->PrintF("# Fixed temperature profile from file = %s\n",
                    parser_->fixed_temperature_file().c_str());
    logger_->PrintF("# Number of file points = %d\n",
                    (int)file_position.size());
    logger_->PrintF("# Position [m]              Temperature [K]\n");
    for(size_t j=0; j<file_position.size(); ++j) {
      logger_->PrintF("%24.16e  %24.16e\n",
                      file_position[j],
                      file_fixed_temperature[j]);
    }


    logger_->PrintF("# Fixed temperature profile on the computational grid\n");
    logger_->PrintF("# Number of grid points = %d\n",(int)z_.size());
    logger_->PrintF("# Position [m]              Wall Temperature [K]\n");
    zerork::utilities::InterpolationTable table(file_position.size(),
                                        &file_position[0],
                                        &file_fixed_temperature[0],
                                        zerork::utilities::LINEAR,
                                        false); // use_interpolation
    const int num_points = z_.size();
    for(int j=0; j<num_points; ++j) {
      fixed_temperature_[j] = table.Interpolate(z_[j]);
      logger_->PrintF("%24.16e  %24.16e\n",
                      z_[j],
                      fixed_temperature_[j]);
      fixed_temperature_[j] /= parser_->ref_temperature();
    }

  } // if(parser_->fixed_temperature_file() == std::string(zerork::utilities::null_filename)) else

}

// requires SetGrid to be set first
void FlameParams::SetMemory()
{
  const int num_species  = reactor_->GetNumSpecies();
  const int num_states   = reactor_->GetNumStates();
  const int num_nonzeros = reactor_->GetJacobianSize();
  const int num_reactors = num_local_points_;
  const int num_points   = z_.size();
  const int num_local_points = num_local_points_;

  std::vector<int> column_id;
  std::vector<int> row_id;
  int last_dense_id, dense_id;

  std::vector<int> row_id_zerod, col_id_zerod;

  // Setup the MassTransportFactory data structure
  transport_input_.num_dimensions_        = 1;
  transport_input_.ld_grad_temperature_   = 1;
  transport_input_.ld_grad_pressure_      = 1;
  transport_input_.ld_grad_mass_fraction_ = num_species;
  transport_input_.mass_fraction_         = new double[num_species];
  transport_input_.grad_temperature_      = new double[1]; // num_dimensions
  transport_input_.grad_pressure_         = new double[1]; // num_dimensions
  transport_input_.grad_mass_fraction_    = new double[num_species];

  // Apply constant pressure approximation
  transport_input_.pressure_         = parser_->pressure();
  transport_input_.grad_pressure_[0] = 0.0;

  // create and set the inverse molecular mass array
  inv_molecular_mass_.assign(num_species, 0.0);
  reactor_->GetSpeciesMolecularWeight(&inv_molecular_mass_[0]);
  for(int j=0; j<num_species; ++j) {
    inv_molecular_mass_[j] = 1.0/inv_molecular_mass_[j];
  }

  // create extended (with ghost cells) state variable array
  // for parallel simulations. nover=2 for second order operators
  y_ext_.assign( (num_local_points_+(2*nover_))*num_states, 0.0);
  rhs_ext_.assign( (num_local_points_+(2*nover_))*num_states, 0.0);

  // create the workspace for the species specific heats
  species_specific_heats_.assign(num_species*(num_local_points+(2*nover_)), 0.0);

  // create the workspace for the species mass fluxes and Lewis numbers
  species_mass_flux_.assign(num_species*(num_local_points+(2*nover_)), 0.0); //larger size for derivatives
  species_lewis_numbers_.assign((num_local_points+(2*nover_))*num_species, 0.0);

  // create the workspace for the flux interface conductivities
  thermal_conductivity_.assign(num_local_points+(2*nover_), 0.0);//larger size for derivatives

  // create the workspace for the mixture specific heat at each grid point
  mixture_specific_heat_.assign(num_local_points+(2*nover_), 0.0);//larger size for derivatives

  // create the workspace for the dissipation rate at each point
  dissipation_rate_.assign(num_local_points+(2*nover_), 0.0); //larger size for derivaties

  // create the workspace for the enthalpy flux sum and convection velocity
  enthalpy_flux_sum_.assign(num_local_points+(2*nover_), 0.0); //larger size for derivaties
  convection_velocity_.assign(num_local_points+(2*nover_), 0.0);

  // Get convective scheme type
  convective_scheme_type_ = parser_->convective_scheme_type();

  // For tracking/output
  max_thermal_diffusivity_ = 0.0;

  // create the Jacobian data structures needed for the banded and SPGMR
  // integrators
  integrator_type_ = parser_->integrator_type();
  store_jacobian_  = parser_->store_jacobian();

  // Flag for implicit transport
  implicit_transport_ = parser_->implicit_transport();

  // Get step limiter
  step_limiter_.assign( reactor_->GetNumSteps(), parser_->step_limiter() );

  // Flag for unity Lewis number equations
  unity_Lewis_ = parser_->unity_Lewis();
  if(unity_Lewis_){printf("# WARNING: Solving unity Lewis number equations.\n");}

  // Flag to solve full flamelet equations
  full_equations_ = parser_->full_equations();
  if(!full_equations_){printf("# WARNING: Not solving full flamelet equations.\n");}

  // Flag to include soot source terms
  soot_ = parser_->soot();
  Y_sootmax = 0.0;
  if(soot_) {printf("# WARNING: Including PAH/density correction source terms.\n");}

  // Compute dissipation rate
  const int num_total_points = z_.size();
  for(int j=0; j<num_local_points+2*nover_; ++j) {
    int jlocal = j-nover_;
    int jglobal = jlocal + my_pe_*num_local_points;
    if(fix_temperature_) {
      if(jglobal<0 || jglobal>num_total_points-1) {
        dissipation_rate_[j] = 0.0;
      } else {
        dissipation_rate_[j] = GetDissipationRateYSI(z_[jglobal]);
      }
    } else {
      double stoichiometric_dissipation_rate;
      double local_dissipation_rate;
      stoichiometric_dissipation_rate =
	GetDissipationRate(stoichiometric_mixture_fraction_);

      if(jglobal<0 || jglobal>num_total_points-1) {
	local_dissipation_rate = 0.0;
      } else {
	local_dissipation_rate = GetDissipationRate(z_[jglobal]);
      }

    dissipation_rate_[j] = scalar_dissipation_rate_*
      local_dissipation_rate/stoichiometric_dissipation_rate;
    }
  }

  //if(integrator_type_ < 0 || integrator_type_ > 2) {
  //  // ****** Here ******
  //}

  if(integrator_type_ == 2) {
    row_id_zerod.assign(num_nonzeros, 0);
    col_id_zerod.assign(num_nonzeros, 0);

    reactor_->GetJacobianPattern(&row_id_zerod[0],
                                 &col_id_zerod[0]);

    dense_to_sparse_.assign(num_states*num_states, 1); // 1 to force dense!!
    dense_to_sparse_offdiag_.assign(num_states*num_states, 1); // 1 to force dense!!
    printf("# WARNING: USING DENSE BLOCK TRIDIAGONAL JACOBIAN\n");

    // Chemical jacobian pattern -- only for diagonal block
    for (int j=0; j<num_nonzeros; j++) {
      dense_id = num_states*col_id_zerod[j] + row_id_zerod[j];
      dense_to_sparse_[dense_id] = 1;
    }

    // Transport -- only diagonals
    for (int j=0; j<num_states; j++) {
      for (int i=0; i<num_states; i++) {
        dense_id = num_states*j + i;

        if(i==j) {
          dense_to_sparse_[dense_id] = 1;
          dense_to_sparse_offdiag_[dense_id] = 1;
        }
      }
    }

    // Count non-zeros
    num_off_diagonals_ = nover_*num_states;
    num_nonzeros_loc_ = 0;
    int width = 2*num_off_diagonals_ + 1;
    int i1, i2;
    for (int group=1; group<=width; group++) {
      for (int j=group-1; j<num_local_points*num_states; j+=width) {
        int jglobal = j + my_pe_*num_local_points*num_states;
        int jstate = jglobal % num_states;
        i1 = max(0, jglobal - jstate - num_states);
        i2 = min(jglobal + (num_states-1 - jstate) + num_states, num_points*num_states-1);
	for (int i=i1; i<=i2; i++) {
          int istate = i % num_states;
          dense_id = num_states*jstate + istate; //here j is column and i is row
          //Diagonal block
          if (i>= jglobal-jstate && i<=jglobal+num_states-1-jstate) {
            if (dense_to_sparse_[dense_id] == 1)
              num_nonzeros_loc_++;
          }
          //Off-diagonal block
          if (i<jglobal-jstate || i>jglobal+num_states-1-jstate) {
            if (dense_to_sparse_offdiag_[dense_id] == 1)
              num_nonzeros_loc_++;
          }
        }
      }
    }

    // SuperLU
    reactor_jacobian_dist_.assign(num_nonzeros_loc_, 0.0);
    col_id_.assign(num_nonzeros_loc_, 0);
    row_id.assign(num_nonzeros_loc_, 0);
    row_sum_.assign(num_local_points*num_states+1,0);

    // Get pattern "manually" for now
    int innz=0;
    // For compressed row storage j is the row and i is the column
    for (int j=0; j<num_local_points*num_states; j++) {
      int jglobal = j + my_pe_*num_local_points*num_states;
      int jstate = jglobal % num_states;
      i1 = max(0, jglobal - jstate - num_states);
      i2 = min(jglobal + (num_states-1 - jstate) + num_states, num_points*num_states-1);
      for (int i=i1; i<=i2; i++) {
        // Compressed column storage
        // row_id_[innz] = i;
        // column_id[innz] = j;
        // Compressed row storage
        int istate = i % num_states;
        dense_id = num_states*istate + jstate; //i is column and j is row

        //Diagonal block.
        if (i>= jglobal-jstate && i<=jglobal+num_states-1-jstate) {
          if (dense_to_sparse_[dense_id] == 1) {
            row_id[innz] = j;
            col_id_[innz] = i;
            innz++;
          }
        }
        //Off-diagonal block.
        if (i<jglobal-jstate || i>jglobal+num_states-1-jstate) {
          if (dense_to_sparse_offdiag_[dense_id] == 1) {
            row_id[innz] = j;
            col_id_[innz] = i;
            innz++;
          }
        }
      } // for i=i1 to i2
    } //for j=0 to < num_local_states

    for(int j=0; j<num_nonzeros_loc_; ++j) {
      ++row_sum_[row_id[j]+1];
    }

    for(int j=0; j<num_local_points*num_states; ++j) {
      row_sum_[j+1] += row_sum_[j];
    }

    sparse_matrix_dist_ = new SparseMatrix_dist(num_local_points*num_states, num_nonzeros_loc_, comm_);
    if(sparse_matrix_dist_ == NULL) {
      printf("fail sparse_matrix_dist_ \n");
      logger_->PrintF("# ERROR: failed to allocate new SparseMatrix object\n");
      exit(-1); // TODO: add recoverable failure
    }

  } // if integrator_type == 2

  if(integrator_type_ == 3) {

    row_id_.assign(num_nonzeros, 0);
    column_id.assign(num_nonzeros, 0);
    column_sum_.assign(num_states+1,0);

    reactor_jacobian_.assign(num_nonzeros, 0.0);

    // The Jacobian pattern is assumed to be in compressed column storage
    reactor_->GetJacobianPattern(&row_id_[0],
                                 &column_id[0]);

    last_dense_id = -1;
    for(int j=0; j<num_nonzeros; ++j) {

      dense_id = num_states*column_id[j] + row_id_[j];

      // check that the column index and row index are increasing consistent
      // with the compressed column storage pattern
      if(dense_id > last_dense_id) {

        last_dense_id = dense_id;
        ++column_sum_[column_id[j]+1];

      } else {

        logger_->PrintF(
          "# ERROR: reactor Jacobian element %d is not in compressed column"
          "#        storage order:\n"
          "#              J element %d (row, col) = (%d, %d)\n"
          "#              J element %d (row, col) = (%d, %d)\n"
          "#        Jacobian structure invalid.\n",
          j,
          j-1,
          row_id_[j-1],
          column_id[j-1],
          j,
          row_id_[j],
          column_id[j],
          j);

        valid_jacobian_structure_ = false;
        exit(-1); // TODO: add recoverable failure

      }  // if(dense_id > last_dense_id) else

    } // for(int j=0; j<num_nonzeros; ++j)

    // At this point the column sum contains just the count per column, need
    // to loop through to change the count to a sum
    for(int j=0; j<num_states; ++j) {
      column_sum_[j+1] += column_sum_[j];
    }

    diagonal_id_.assign(num_states, -1); // use negative one as flag to
                                         // check if there are any missing
                                         // diagonal terms that would prevent
                                         // its use for the sparse factorization
    for(int j=0; j<num_nonzeros; ++j) {
      if(row_id_[j] == column_id[j]) {
        diagonal_id_[row_id_[j]] = j;
      }
    }
    for(int j=0; j<num_states; ++j) {
      if(diagonal_id_[j] == -1) {
        logger_->PrintF(
	  "# ERROR: reactor Jacobian does not have diagonal in row %d\n",
          j);

        valid_jacobian_structure_ = false;
        exit(-1); // TODO: add recoverable failure
      }
    }

    // Create the vector of the sparse matrix elements
    sparse_matrix_.assign(num_reactors, NULL);
    for(int j=0; j<num_reactors; ++j) {
      sparse_matrix_[j] = new SparseMatrix(num_states, num_nonzeros);
      if(sparse_matrix_[j] == NULL) {
        logger_->PrintF(
          "# ERROR: failed to allocate new SparseMatrix object\n"
          "#        for grid point %d (z = %.18g [m])\n",
          j, z_[j]);
        exit(-1); // TODO: add recoverable failure
      }
    }

    if(store_jacobian_) {
      // TODO: add allocation check in case we run out of memory for large
      //       mechanisms and grids
      saved_jacobian_.assign(num_nonzeros*num_reactors, 0.0);
    }

  }

  if(integrator_type_ == 3 && implicit_transport_) {
    num_off_diagonals_ = 1;
    int storage = 4*num_off_diagonals_ + 1;
    banded_jacobian_.assign(num_local_points*storage * num_states, 0.0);

    num_states_per_proc_ = (num_states + npes_ - 1)/npes_;
    if(num_states - (npes_-1)*num_states_per_proc_ < 1) {
      cerr << "Too few states per processor. Try using fewer processors.\n";
      exit(-1);
    }
    if(my_pe_ == npes_-1) {
      num_states_local_ = num_states - (npes_-1)*num_states_per_proc_;
    } else {
      num_states_local_ = num_states_per_proc_;
    }
    banded_jacobian2_.assign(num_points*storage * num_states_local_, 0.0);
    banded_jacobian_serial_.assign(num_points*4*num_states_local_, 0.0);
    pivots_serial_.assign(num_points*num_states_local_, 0);

  }

}

static double NormalizeComposition(const size_t num_elements,
                                   double composition[])
{
  double sum = 0.0;
  for(size_t j=0; j<num_elements; ++j) {
    sum += composition[j];
  }
  if(fabs(sum) > 1.0e-300) {
    sum = 1.0/sum;
    for(size_t j=0; j<num_elements; ++j) {
      composition[j] *= sum;
    }
    sum = 1.0/sum;
  }
  return sum;
}

double GetDissipationRate(double mixture_fraction)
{
  if(mixture_fraction <= 0.0 || mixture_fraction >= 1.0) {
    return 0.0;
  } else {
    double two_mixture_fraction = 2.0*mixture_fraction;
    double inv_erfc = zerork::utilities::erfc_inv(two_mixture_fraction);
    return exp(-2.0*inv_erfc*inv_erfc);
  }
}

double GetDissipationRateYSI(double mixture_fraction)
{
  if(mixture_fraction <= 0.0 || mixture_fraction >= 1.0) {
    return 0.0;
  } else {
    double a=0.0;
    if (mixture_fraction < 0.15 )
      a=20.0 * pow(mixture_fraction,3.0) * pow(1.0-mixture_fraction,1.7);
    if (mixture_fraction > 0.15)
      a=120*exp(-0.6/(mixture_fraction-0.11))*
        pow(1.0-mixture_fraction,1.5)
        +0.1*exp(-50.0*(mixture_fraction-0.3)*(mixture_fraction-0.7))
        +0.1*exp(-50.0*(mixture_fraction-0.2)*(mixture_fraction-0.4));

    double b=-2e-14*exp(-150*(mixture_fraction+0.2)*(mixture_fraction-0.7));

    double f=1.75*a-b;

    return f;

  }
}
