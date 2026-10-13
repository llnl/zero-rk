#ifndef ZERORK_DEVICE_VECTOR_H
#define ZERORK_DEVICE_VECTOR_H

#ifdef ZERORK_HAVE_RMM_DEVICE_VECTOR
#include "rmm/device_vector.hpp"
#else
#include "thrust/device_vector.h"
#endif
namespace zerork {
#ifdef ZERORK_HAVE_RMM_DEVICE_VECTOR
    template<typename T> using device_vector = rmm::device_vector<T>;
#else
    template<typename T> using device_vector = thrust::device_vector<T>;
#endif
}

#endif
