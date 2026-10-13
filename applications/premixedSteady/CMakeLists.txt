
set(MAIN_SRC    premixed_steady_flame_solver.cpp)
set(COMMON_SRC  kinsol_functions.cpp set_initial_conditions.cpp flame_params.cpp sparse_matrix.cpp)
set(SPIFY_SRC   UnsteadyFlameIFP.cpp)
set(SPIFY_APPS  premixed_steady_flame_solver.x)

add_executable(premixed_steady_flame_solver.x ${MAIN_SRC} ${COMMON_SRC} ${SPIFY_SRC})
target_link_libraries(premixed_steady_flame_solver.x zerorkutilities mechanisminfo reactor
                      zerorktransport zerork superlu spify sundials_kinsol sundials_nvecserial)
target_link_libraries(premixed_steady_flame_solver.x zerork_equilibrium)
target_link_libraries(premixed_steady_flame_solver.x zerork_restart_writer)

install(TARGETS premixed_steady_flame_solver.x
        RUNTIME DESTINATION bin)

if(ENABLE_MPI)
add_mpi_executable(premixed_steady_flame_solver_mpi.x ${MAIN_SRC} ${COMMON_SRC} sparse_matrix_dist.cpp ${SPIFY_SRC})
target_link_libraries(premixed_steady_flame_solver_mpi.x zerorkmpiutilities zerorkutilities mechanisminfo reactor
                      zerorktransport zerork superlu_dist superlu spify sundials_kinsol sundials_nvecparallel)
target_link_libraries(premixed_steady_flame_solver_mpi.x zerork_equilibrium)
target_link_libraries(premixed_steady_flame_solver.x zerork_restart_writer)
install(TARGETS premixed_steady_flame_solver_mpi.x
        RUNTIME DESTINATION bin)
set(SPIFY_APPS "${SPIFY_APPS};premixed_steady_flame_solver_mpi.x")
endif()

add_spifyIFP_target(UnsteadyFlameIFP ${SPIFY_APPS})
