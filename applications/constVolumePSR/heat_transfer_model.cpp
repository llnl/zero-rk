
#ifdef _WIN32
#define _USE_MATH_DEFINES //for M_PI
#endif
#include <cmath>

#include "heat_transfer_model.h"

#include <transport/mass_transport_factory.h>

namespace zerork {

class NullHeatTransferModel : public HeatTransferModel {
  public:
    NullHeatTransferModel(zerork::mechanism* mech, idt_sweep_IFP& input_db)
     : HeatTransferModel(mech, input_db) {};
    ~NullHeatTransferModel() {};
    void Update(double T, double p, double rho, double cp, double tau_res, double* mf) {};
    double Nu() {return 0.0;};
    double h() {return 0.0;};
    double H() {return 0.0;};
    double reactor_area() {return 1.0;};
    double reactor_volume() {return 1.0;};
    double wall_temperature() {return 373.0;};
};

class PrHeatTransferModel : public HeatTransferModel {
  public:
    PrHeatTransferModel(zerork::mechanism* mech, idt_sweep_IFP& input_db)
     : HeatTransferModel(mech, input_db) {
      // setup transport interface
      trans_ = std::unique_ptr<transport::MassTransportInterface>(transport::InterfaceFactory::CreateMassBased(input_db.transport_model()));
      std::vector<std::string> transport_files;
      transport_files.push_back(input_db.transport_file());

      int error_code = trans_->Initialize(mech, transport_files, std::string("transport_parser_log"));
      if(error_code != transport::NO_ERROR) {
        printf("# ERROR: Could not Initialize MassTransportInterface for files:\n"
               "#            transport      file = %s\n"
               "#        Initialize returned error code = %d\n",
               input_db.transport_file().c_str(),
               error_code);
        exit(-1);
      }
      reactor_radius_ = input_db.reactor_radius();
      Nj_ = input_db.njets();
      dj_ = input_db.djet();
      Re_ = 0.0;
      Pr_ = 1.0;
      Twall_ = input_db.wall_temperature();
    };
    
    ~PrHeatTransferModel() {};
    void Update(double T, double p, double rho, double cp, double tau_res, double* mf) {
      transport_input_.temperature_ = T;
      transport_input_.pressure_ = p;
      transport_input_.mass_fraction_ = mf;
      trans_->GetMixtureViscosity(transport_input_,&mu_);
      trans_->GetMixtureConductivity(transport_input_,&k_);
      Re_ = 4*rho*this->reactor_volume()/(tau_res * M_PI * Nj_ * dj_ * mu_);
      Pr_ = cp*mu_/k_;
    };
    virtual double Nu() = 0;
    virtual double h() {
      return this->Nu()*k_/(2*reactor_radius_);
    }
    virtual double H() {
      return this->h()*this->reactor_area();
    }
    virtual double reactor_area() {return 4*M_PI*reactor_radius_*reactor_radius_;};
    virtual double reactor_volume() {return 4/3*M_PI*reactor_radius_*reactor_radius_*reactor_radius_;};
    virtual double wall_temperature() {return Twall_;};
  protected:

    std::unique_ptr<transport::MassTransportInterface> trans_;
    transport::MassTransportInput transport_input_;
    double Re_;
    double Pr_;
    double reactor_radius_;
    double Nj_;
    double dj_;
    double mu_;
    double k_;
    double Twall_;
};

class LRHeatTransferModel : public PrHeatTransferModel {
  public:
    LRHeatTransferModel(zerork::mechanism* mech, idt_sweep_IFP& input_db) :
       PrHeatTransferModel(mech, input_db) {
         a_ = 0.89;
	 b_ = 0.683;
	 c_ = 0.333333;
       };
    
    ~LRHeatTransferModel() {};
    double Nu() {
      return a_*pow(Re_,b_)*pow(Pr_,c_);
    };
  private:
    double a_, b_, c_;
};


class DHVHeatTransferModel : public PrHeatTransferModel {
  public:
    DHVHeatTransferModel(zerork::mechanism* mech, idt_sweep_IFP& input_db) :
       PrHeatTransferModel(mech, input_db) {
         a_ = 12;
	 b_ = 0.0460;
	 c_ = -1.5;
    };

    ~DHVHeatTransferModel() {};
    double Nu() {
      return a_+b_*Re_*pow(Pr_,c_);
    };
  private:
    double a_, b_, c_;
};

std::unique_ptr<HeatTransferModel> HeatTransferModelFactory(zerork::mechanism* mech, idt_sweep_IFP& input_db) {
  std::string type = input_db.heat_transfer_model();
  if(type == std::string("none")) {
    return std::unique_ptr<HeatTransferModel>(new NullHeatTransferModel(mech, input_db));
  } else if (type == std::string("Lignola-Reverchon")) {
    return std::unique_ptr<HeatTransferModel>(new LRHeatTransferModel(mech, input_db));
  } else if (type == std::string("David-Houzelot-Villermaux")) {
    return std::unique_ptr<HeatTransferModel>(new DHVHeatTransferModel(mech, input_db));
  } else {
    printf("Unrecognized heat transfer model. Using \"null\" model.\n");
    return std::unique_ptr<HeatTransferModel>(new NullHeatTransferModel(mech, input_db));
  }
}

} //namespace zerork

