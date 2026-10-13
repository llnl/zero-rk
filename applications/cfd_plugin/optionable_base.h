#ifndef OPTIONABLE_BASE_H
#define OPTIONABLE_BASE_H

#include <string>
#include <unordered_map>

typedef enum _optionable_status_t {
  OPTIONABLE_STATUS_SUCCESS = 0,
  OPTIONABLE_STATUS_OPTION_NOT_DEFINED
} optionable_status_t;

class OptionableBase
{
 public:
  OptionableBase() {};
  virtual ~OptionableBase() {};

  typedef std::unordered_map<std::string,int> IntOptions;
  typedef std::unordered_map<std::string,double> DoubleOptions;
  typedef std::unordered_map<std::string,std::string> StringOptions;

  virtual optionable_status_t GetIntOption(const std::string option_name, int* option_value) = 0;
  virtual optionable_status_t GetDoubleOption(const std::string option_name, double* option_value) = 0;
  virtual optionable_status_t GetStringOption(const std::string option_name, std::string* option_value) = 0;

  virtual void GetIntOptions(IntOptions* options) = 0;
  virtual void GetDoubleOptions(DoubleOptions* options) = 0;
  virtual void GetStringOptions(StringOptions* options) = 0;

  virtual optionable_status_t SetIntOption(const std::string option_name, int option_value) = 0;
  virtual optionable_status_t SetDoubleOption(const std::string option_name, double option_value) = 0;
  virtual optionable_status_t SetStringOption(const std::string option_name, std::string option_value) = 0;

  virtual void SetIntOptions(const IntOptions& options) = 0;
  virtual void SetDoubleOptions(const DoubleOptions& options) = 0;
  virtual void SetStringOptions(const StringOptions& options) = 0;

 protected:
  IntOptions int_options_;
  DoubleOptions double_options_;
  StringOptions string_options_;
};


#endif
