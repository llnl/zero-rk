
#Disable in-source builds to prevent source tree corruption.
if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR
"In-source builds disabled.
    NOTE: You must delete CMakeCache.txt and CMakeFiles/* in
          the source folder if they exist.")
endif()

cmake_minimum_required(VERSION 3.15)

include(cmake/policies.cmake)

project(zerork LANGUAGES C CXX VERSION 3.0.8)
include(CMakeDependentOption)

find_package (Python3 REQUIRED COMPONENTS Interpreter)

set(ZERORK_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})
set(ZERORK_DATA_DIR ${ZERORK_SOURCE_DIR}/data)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/inst_dir" CACHE PATH "Path to install zerork" FORCE)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

option(ENABLE_SHARED_LIBS "" OFF)
option(ENABLE_MPI "" ON)
option(ENABLE_OPENMP "Enable OpenMP (currently only used in cfd_plugin_tester)" OFF)
option(ZERORK_TESTS "Enable Zero-RK Tests" ON)
option(ZERORK_EXP_LIBC "Use libc exponential function instead of platform fast exponential" OFF)

if(WIN32)
  set(ZERORK_EXTERNALS_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING "Build type for external dependencies built during config.")
else()
  set(ZERORK_EXTERNALS_BUILD_TYPE "Release" CACHE STRING "Build type for external dependencies built during config.")
endif()
set_property(CACHE ZERORK_EXTERNALS_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")

if(${ENABLE_SHARED_LIBS})
  set(BUILD_SHARED_LIBS ON CACHE INTERNAL "")
else()
  set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "")
endif()

if(${ENABLE_MPI})
  include(cmake/mpi.cmake)
endif()

if(${ENABLE_OPENMP})
find_package(OpenMP)
endif()

include(cmake/platform.cmake)
include(cmake/git.cmake)
include(cmake/gpu.cmake)
include(cmake/spify.cmake)
include(cmake/blas.cmake)
include(cmake/lapack.cmake)
include(cmake/vmath.cmake)
include(cmake/superlu.cmake)
if(${ENABLE_MPI})
  include(cmake/superlu_dist.cmake)
endif()
include(cmake/sundials.cmake)
include(cmake/ceq.cmake)
include(cmake/cantera.cmake)

add_subdirectory(src)
add_subdirectory(tools)
add_subdirectory(applications)
add_subdirectory(python)
add_subdirectory(examples)
add_subdirectory(data)

if(${ZERORK_TESTS})
  if(${CMAKE_CROSSCOMPILING})
    message(WARNING "Disabling tests when cross-compiling.")
  else()
    include(cmake/tests.cmake)
    add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure)
    add_subdirectory(tests)
  endif()
endif()

