

#This module will make a header that contains
#the appropriate Windows definitions for
#creating/using a shared library
include(GenerateExportHeader)


set(COMMON_SRCS utility_funcs.cpp
       optionable.cpp reactor_base.cpp
       reactor_constant_volume_cpu.cpp
       reactor_constant_pressure_cpu.cpp
       reactor_nvector_serial.cpp solver_cvode.cpp 
       solver_seulex.cpp solver_sodex.cpp 
       solver_radau.cpp utility_funcs.cpp
       zerork_reactor_manager.cpp
       interfaces/superlu_manager/superlu_manager.cpp
       interfaces/superlu_manager/superlu_manager_z.cpp
       interfaces/lapack_manager/lapack_manager.cpp
       interfaces/lapack_manager/lapack_manager_z.cpp
       interfaces/seulex_cpp/seulex_cpp.cpp
       interfaces/sodex_cpp/sodex_cpp.cpp
       interfaces/radau5_cpp/radau_cpp.cpp)


if(ZERORK_ENABLE_SUNDIALS_LAPACK)
  list(APPEND COMMON_SRCS interfaces/zrkmatrix/zrkmatrix_lapackdense.c)
endif()

if(ENABLE_MPI)
  add_mpi_library(zerork_cfd_plugin SHARED ${COMMON_SRCS} zerork_cfd_plugin.cpp)
  target_compile_definitions(zerork_cfd_plugin PRIVATE USE_MPI)
  add_mpi_executable(zerork_cfd_plugin_tester.x zerork_cfd_plugin_tester.cpp ZeroRKCFDPluginTesterIFP.cpp)
  target_compile_definitions(zerork_cfd_plugin_tester.x PRIVATE USE_MPI)
else()
  add_library(zerork_cfd_plugin SHARED ${COMMON_SRCS} zerork_cfd_plugin.cpp)
  add_executable(zerork_cfd_plugin_tester.x zerork_cfd_plugin_tester.cpp ZeroRKCFDPluginTesterIFP.cpp)
endif()

target_link_libraries(zerork_cfd_plugin zerork zerorkutilities superlu spify sundials_cvode
                      sundials_nvecserial)
if(${ZERORK_ENABLE_SUNDIALS_LAPACK})
  target_link_libraries(zerork_cfd_plugin sundials_sunlinsollapackdense)
endif()

if(NOT WIN32 AND NOT APPLE)
  target_link_options(zerork_cfd_plugin PRIVATE -Wl,-Bsymbolic-functions -Wl,-Bsymbolic)
endif()
target_link_libraries(zerork_cfd_plugin_tester.x zerork_cfd_plugin)

if(ENABLE_OPENMP)
  target_compile_definitions(zerork_cfd_plugin_tester.x PRIVATE USE_OMP)
  target_link_libraries(zerork_cfd_plugin_tester.x OpenMP::OpenMP_CXX)
endif()

set(SPIFY_PLUGIN_TARGETS "zerork_cfd_plugin")
set(SPIFY_TESTER_TARGETS "zerork_cfd_plugin_tester.x")

GENERATE_EXPORT_HEADER(zerork_cfd_plugin
                       BASE_NAME zerork_cfd_plugin
                       EXPORT_MACRO_NAME ZERORK_CFD_PLUGIN_EXPORTS
                       EXPORT_FILE_NAME zerork_cfd_plugin_exports.h)

set_target_properties(zerork_cfd_plugin PROPERTIES PUBLIC_HEADER
                      "zerork_cfd_plugin.h;${CMAKE_CURRENT_BINARY_DIR}/zerork_cfd_plugin_exports.h")

target_include_directories(zerork_cfd_plugin
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS zerork_cfd_plugin PUBLIC_HEADER DESTINATION include
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION lib
        ARCHIVE DESTINATION lib)
install(TARGETS zerork_cfd_plugin_tester.x
        RUNTIME DESTINATION bin)

if(ENABLE_GPU)
if(${SUNDIALS_VERSION} EQUAL "5")
  set(GPU_SRCS reactor_constant_volume_gpu.cu
               reactor_constant_pressure_gpu.cu
               reactor_nvector_serial_cuda.cu
               interfaces/cuda_la_manager/cuda_la_manager.cpp
               interfaces/cublas_manager/cublas_manager.cu
               interfaces/cusolver_rf_manager/cusolver_rf_manager.cu
               cuda_transpose.cu)
  if(${ZERORK_HAVE_MAGMA})
    list(APPEND GPU_SRCS interfaces/zrklinsol/zrklinsol_magmadense.cu
                         interfaces/magma_manager/magma_manager.cu)
  endif()
  if(ENABLE_MPI)
    add_mpi_library(zerork_cfd_plugin_gpu SHARED ${COMMON_SRCS} ${GPU_SRCS} zerork_cfd_plugin.cpp)
    add_mpi_executable(zerork_cfd_plugin_tester_gpu.x zerork_cfd_plugin_tester.cpp ZeroRKCFDPluginTesterIFP.cpp)
    target_compile_definitions(zerork_cfd_plugin_gpu PRIVATE USE_MPI ZERORK_GPU)
    target_compile_definitions(zerork_cfd_plugin_tester_gpu.x PRIVATE USE_MPI ZERORK_GPU)
  else()
    add_library(zerork_cfd_plugin_gpu SHARED ${COMMON_SRCS} ${GPU_SRCS} zerork_cfd_plugin.cpp)
    target_compile_definitions(zerork_cfd_plugin_gpu PRIVATE ZERORK_GPU)
    add_executable(zerork_cfd_plugin_tester_gpu.x zerork_cfd_plugin_tester.cpp ZeroRKCFDPluginTesterIFP.cpp)
    target_compile_definitions(zerork_cfd_plugin_tester_gpu.x PRIVATE ZERORK_GPU)
  endif()
  target_include_directories(zerork_cfd_plugin_gpu PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

  target_link_libraries(zerork_cfd_plugin_gpu zerork_cuda zerork zerorkutilities superlu spify
                        sundials_nveccuda sundials_cvode sundials_nvecserial cublas cusolver cudart)
  if(${ZERORK_ENABLE_SUNDIALS_LAPACK})
    target_link_libraries(zerork_cfd_plugin_gpu sundials_sunlinsollapackdense)
  endif()

  if(${ZERORK_HAVE_MAGMA})
    target_link_libraries(zerork_cfd_plugin_gpu sundials_sunlinsolmagmadense sundials_sunmatrixmagmadense magma)
  endif()

  target_link_options(zerork_cfd_plugin_gpu PRIVATE -Wl,-Bsymbolic-functions -Wl,-Bsymbolic)
  target_link_libraries(zerork_cfd_plugin_tester_gpu.x zerork_cfd_plugin_gpu)
  set(SPIFY_PLUGIN_TARGETS "${SPIFY_PLUGIN_TARGETS};zerork_cfd_plugin_gpu")
  set(SPIFY_TESTER_TARGETS "${SPIFY_TESTER_TARGETS};zerork_cfd_plugin_tester_gpu.x")
  set_target_properties(zerork_cfd_plugin_gpu PROPERTIES PUBLIC_HEADER
                        "zerork_cfd_plugin.h;${CMAKE_CURRENT_BINARY_DIR}/zerork_cfd_plugin_exports.h")

  target_include_directories(zerork_cfd_plugin_gpu
          PUBLIC
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
          $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

  install(TARGETS zerork_cfd_plugin_gpu LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION include
          RUNTIME DESTINATION bin)
  install(TARGETS zerork_cfd_plugin_tester_gpu.x
          RUNTIME DESTINATION bin)
else()
  message(NOTICE "cfd_plugin_gpu disabled for SUNDIALS_VERSION!=5")
endif()
endif()

add_spifyIFP_target(ZeroRKCFDPluginIFP ${SPIFY_PLUGIN_TARGETS})
add_spifyIFP_target(ZeroRKCFDPluginTesterIFP ${SPIFY_TESTER_TARGETS})

