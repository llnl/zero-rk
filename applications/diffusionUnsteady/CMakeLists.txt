
if(ENABLE_MPI)
add_mpi_executable(diffusion_unsteady_flame_solver.x diffusion_unsteady_flame_solver.cpp
                   cvode_functions.cpp set_initial_conditions.cpp
                   flame_params.cpp sparse_matrix.cpp sparse_matrix_dist.cpp
                   UnsteadyFlameIFP.cpp soot.cpp)

target_link_libraries(diffusion_unsteady_flame_solver.x zerorkutilities mechanisminfo reactor
                      zerorktransport zerork superlu_dist superlu spify sundials_cvode sundials_nvecparallel)
target_link_libraries(diffusion_unsteady_flame_solver.x zerork_restart_writer)

install(TARGETS diffusion_unsteady_flame_solver.x
        RUNTIME DESTINATION bin)

add_spifyIFP_target(UnsteadyFlameIFP diffusion_unsteady_flame_solver.x)
endif()
