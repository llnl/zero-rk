
set(SRCS cvIDT_sweep.cpp matrix_funcs.cpp ode_funcs.cpp
         cv_param_sparse.cpp sweep_util_spify.cpp
	 heat_transfer_model.cpp
	 idt_sweep_IFP.cpp)
set(SPIFY_APPS constPressurePaSR.x)

add_executable(constPressurePaSR.x ${SRCS})
target_link_libraries(constPressurePaSR.x zerork_equilibrium zerorktransport zerork sundials_nvecserial sundials_cvode superlu spify)
target_compile_definitions(constPressurePaSR.x PRIVATE SPIFY)
install(TARGETS constPressurePaSR.x)

if(ENABLE_MPI)
add_mpi_executable(constPressurePaSR_mpi.x ${SRCS})
target_link_libraries(constPressurePaSR_mpi.x zerorkmpiutilities zerorktransport zerork sundials_nvecserial sundials_cvode superlu spify)
target_link_libraries(constPressurePaSR_mpi.x zerork_equilibrium)
install(TARGETS constPressurePaSR_mpi.x RUNTIME DESTINATION bin)
set(SPIFY_APPS "${SPIFY_APPS};constPressurePaSR_mpi.x")
endif()

add_spifyIFP_target(idt_sweep_IFP ${SPIFY_APPS})
