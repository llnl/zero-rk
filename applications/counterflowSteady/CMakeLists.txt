if(${SUNDIALS_VERSION} EQUAL "5")
if(ENABLE_MPI)
add_mpi_executable(counterflow_steady_flame_solver.x counterflow_steady_flame_solver.cpp
                   kinsol_functions.cpp set_initial_conditions.cpp
                   flame_params.cpp sparse_matrix.cpp sparse_matrix_dist.cpp SteadyFlameIFP.cpp)

target_link_libraries(counterflow_steady_flame_solver.x zerorkmpiutilities zerorkutilities mechanisminfo reactor
                      zerorktransport zerork superlu_dist superlu spify sundials_kinsol sundials_nvecparallel)

target_link_libraries(counterflow_steady_flame_solver.x zerork_restart_writer)

install(TARGETS counterflow_steady_flame_solver.x
        RUNTIME DESTINATION bin)

add_spifyIFP_target(SteadyFlameIFP counterflow_steady_flame_solver.x)
endif()
else()
message(NOTICE "steadyCounterFlow Flame solver disabled for SUNDIALS_VERSION!=5")
endif()
