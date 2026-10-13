
set(MAIN_SRC    premixed_unsteady_flame_solver.cpp)
set(COMMON_SRC  cvode_functions.cpp set_initial_conditions.cpp flame_params.cpp sparse_matrix.cpp)
set(SPIFY_SRC   UnsteadyFlameIFP.cpp)
set(SPIFY_APPS  premixed_unsteady_flame_solver.x)

add_executable(premixed_unsteady_flame_solver.x ${MAIN_SRC} ${COMMON_SRC} ${SPIFY_SRC})
target_link_libraries(premixed_unsteady_flame_solver.x zerork_restart_writer zerorkutilities mechanisminfo reactor
                      zerorktransport zerork superlu spify sundials_cvode sundials_nvecserial)
target_link_libraries(premixed_unsteady_flame_solver.x zerork_equilibrium)
install(TARGETS premixed_unsteady_flame_solver.x
        RUNTIME DESTINATION bin)

if(ENABLE_MPI)
add_mpi_executable(premixed_unsteady_flame_solver_mpi.x ${MAIN_SRC} ${COMMON_SRC} ${SPIFY_SRC})
target_link_libraries(premixed_unsteady_flame_solver_mpi.x zerork_restart_writer zerorkutilities mechanisminfo reactor
                      zerorktransport zerork superlu spify sundials_cvode sundials_nvecparallel)
target_link_libraries(premixed_unsteady_flame_solver_mpi.x zerork_equilibrium)
install(TARGETS premixed_unsteady_flame_solver_mpi.x
        RUNTIME DESTINATION bin)
set(SPIFY_APPS  "${SPIFY_APPS};premixed_unsteady_flame_solver_mpi.x")
endif()

add_spifyIFP_target(UnsteadyFlameIFP ${SPIFY_APPS})
