
#include "optionable.h"

optionable_status_t Optionable::GetIntOption(const std::string option_name, int* option_value) {
  std::unordered_map<std::string,int>::iterator it;
  it = int_options_.find(option_name);
  if( it != int_options_.end() ) {
    *option_value = it->second;
    return OPTIONABLE_STATUS_SUCCESS;
  } else {
    return OPTIONABLE_STATUS_OPTION_NOT_DEFINED;
  }
}

optionable_status_t Optionable::GetDoubleOption(const std::string option_name, double* option_value) {
  std::unordered_map<std::string,double>::iterator it;
  it = double_options_.find(option_name);
  if( it != double_options_.end() ) {
    *option_value = it->second;
    return OPTIONABLE_STATUS_SUCCESS;
  } else {
    return OPTIONABLE_STATUS_OPTION_NOT_DEFINED;
  }
}

optionable_status_t Optionable::GetStringOption(const std::string option_name, std::string* option_value) {
  std::unordered_map<std::string,std::string>::iterator it;
  it = string_options_.find(option_name);
  if( it != string_options_.end() ) {
    *option_value = it->second;
    return OPTIONABLE_STATUS_SUCCESS;
  } else {
    return OPTIONABLE_STATUS_OPTION_NOT_DEFINED;
  }
}

void Optionable::GetIntOptions(IntOptions* options) {
  *options = int_options_;
}

void Optionable::GetDoubleOptions(DoubleOptions* options) {
  *options = double_options_;
}

void Optionable::GetStringOptions(StringOptions* options) {
  *options = string_options_;
}

optionable_status_t Optionable::SetIntOption(const std::string option_name, int option_value) {
  int_options_[option_name] = option_value;
  return OPTIONABLE_STATUS_SUCCESS;
}

optionable_status_t Optionable::SetDoubleOption(const std::string option_name, double option_value) {
  double_options_[option_name] = option_value;
  return OPTIONABLE_STATUS_SUCCESS;
}

optionable_status_t Optionable::SetStringOption(const std::string option_name, std::string option_value) {
  string_options_[option_name] = option_value;
  return OPTIONABLE_STATUS_SUCCESS;
}

void Optionable::SetIntOptions(const IntOptions& options) {
  int_options_ = options;
}

void Optionable::SetDoubleOptions(const DoubleOptions& options) {
  double_options_ = options;
}

void Optionable::SetStringOptions(const StringOptions& options) {
  string_options_ = options;
}


