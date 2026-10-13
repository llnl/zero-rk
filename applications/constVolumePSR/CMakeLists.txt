
set(MAIN_SRC      cvIDT_sweep.cpp)
set(COMMON_SRC    matrix_funcs.cpp ode_funcs.cpp cv_param_sparse.cpp utility_funcs.cpp heat_transfer_model.cpp)
set(SPIFY_SRC     sweep_util_yml.cpp idt_sweep_IFP.cpp)
set(SPIFY_APPS    constVolumePSR.x)

add_executable(constVolumePSR.x ${MAIN_SRC} ${COMMON_SRC} ${SPIFY_SRC})
target_link_libraries(constVolumePSR.x zerork zerork_equilibrium zerorktransport sundials_nvecserial sundials_cvode superlu spify)
target_compile_definitions(constVolumePSR.x PRIVATE SPIFY)
install(TARGETS constVolumePSR.x
        RUNTIME DESTINATION bin)

if(ENABLE_MPI)
add_mpi_executable(constVolumePSR_mpi.x ${MAIN_SRC} ${COMMON_SRC} ${SPIFY_SRC})
target_link_libraries(constVolumePSR_mpi.x zerork zerork_equilibrium zerorktransport zerorkmpiutilities sundials_nvecserial sundials_cvode superlu spify)
target_compile_definitions(constVolumePSR_mpi.x PRIVATE SPIFY)
install(TARGETS constVolumePSR_mpi.x
        RUNTIME DESTINATION bin)
set(SPIFY_APPS    "${SPIFY_APPS};constVolumePSR_mpi.x")
endif()

add_spifyIFP_target(idt_sweep_IFP ${SPIFY_APPS})
