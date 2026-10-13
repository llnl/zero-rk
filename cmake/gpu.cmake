
option(ENABLE_GPU "Enable CUDA libraries and applications." OFF)
cmake_dependent_option(ENABLE_MAGMA "Enable MAGMA GPU library" ON "ENABLE_GPU" OFF)
if(${ENABLE_GPU})
  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  set(CMAKE_CUDA_STANDARD 14)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 90)
  endif()
  enable_language(CUDA)
  #N.B. this is for configuring external projects that use CMAKE_CUDA_ARCHITECTURES
  #     as semi-colon separated lists get munged when passing through ExternalProject_Add
  string(REPLACE ";" "|" CMAKE_CUDA_ARCHITECTURES_EXT "${CMAKE_CUDA_ARCHITECTURES}")

  set(ZERORK_HAVE_MAGMA OFF)
  if(${ENABLE_MAGMA})
    include(cmake/magma.cmake)
    set(ZERORK_HAVE_MAGMA ON)
  endif()

  option(ZERORK_HAVE_RMM_DEVICE_VECTOR "Toggle for RMM Device Vector in cfd_plugin" OFF)
endif()

