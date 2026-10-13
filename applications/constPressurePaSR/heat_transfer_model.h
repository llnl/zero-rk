#ifndef HEAT_TRANSFER_MODEL_H
#define HEAT_TRANSFER_MODEL_H

#include <memory>

#include <zerork/mechanism.h>
#include "idt_sweep_IFP.h"

namespace zerork {

class HeatTransferModel {
  public:
    HeatTransferModel(zerork::mechanism* mech, idt_sweep_IFP& input_db) {};
    ~HeatTransferModel () {};
    virtual void Update(double T, double p, double rho, double cp, double* mf) = 0;
    virtual double Nu() = 0;
    virtual double h() = 0;
    virtual double H() = 0;
    virtual double reactor_area() = 0;
    virtual double reactor_volume() = 0;
    virtual double wall_temperature() = 0;
};

std::unique_ptr<HeatTransferModel> HeatTransferModelFactory(zerork::mechanism* mech, idt_sweep_IFP& input_db);

} //namespace zerork

#endif
