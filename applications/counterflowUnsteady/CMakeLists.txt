
if(${SUNDIALS_VERSION} EQUAL "5")
if(ENABLE_MPI)
add_mpi_executable(counterflow_unsteady_flame_solver.x counterflow_unsteady_flame_solver.cpp
                   cvode_functions.cpp set_initial_conditions.cpp
                   flame_params.cpp sparse_matrix.cpp UnsteadyFlameIFP.cpp)

target_link_libraries(counterflow_unsteady_flame_solver.x zerorkutilities mechanisminfo reactor
                      zerorktransport zerork superlu spify sundials_cvode sundials_nvecparallel)
target_link_libraries(counterflow_unsteady_flame_solver.x zerork_equilibrium)
target_link_libraries(counterflow_unsteady_flame_solver.x zerork_restart_writer)

install(TARGETS counterflow_unsteady_flame_solver.x
        RUNTIME DESTINATION bin)

add_spifyIFP_target(UnsteadyFlameIFP counterflow_unsteady_flame_solver.x)
endif()
else()
message(NOTICE "unsteadyCounterFlow Flame solver disabled for SUNDIALS_VERSION!=5")
endif()
