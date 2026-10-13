
set(INPUT_FILES constVolumeWSR/hydrogen_TSweep.yml.in
  constVolumeWSR/run.sh.in
  constVolumeWSR_TLA/run.sh.in
  constVolumeWSR_TLA/hydrogen_TSweep.yml.in
  constVolumePSR/input.yml.in
  constVolumePSR/run.sh.in
  idt_diagnostic/hydrogen_stats.yml.in
  idt_diagnostic/run.sh.in
  thermo_check/thermo_check.yml.in
  thermo_check/run.sh.in
  variable_volume/cv/h2_cv.yml.in
  variable_volume/cv/run.sh.in
  variable_volume/rcm/h2_rcm.yml.in
  variable_volume/rcm/run.sh.in
  variable_volume_batch/hydrogen.yml.in
  variable_volume_batch/run.sh.in
  perturbAFactor/perturbAFactor_example.yml.in
  perturbAFactor/run.sh.in
  cfd_plugin_tester/run.sh.in
  cfd_plugin_tester/run_gpu.sh.in
  rate_optimization/run.sh.in
  rate_optimization/optimize_rate_h2_simple.py.in
  negative_rate_refitter/run.sh.in
  surrogate_optimizer/diesel/run.sh.in
  surrogate_optimizer/tprf/run.sh.in
  flame_api_tester/input.yml.in
  flame_api_tester/run.sh.in
  mechanism_parser/hydrogen.yml.in
  mechanism_parser/run.sh.in
  )

if(ENABLE_MPI)
get_filename_component(MPIEXEC_NAME ${MPIEXEC_EXECUTABLE} NAME)
set(MPIEXEC_EXAMPLE_NUMPROCS ${MPIEXEC_MAX_NUMPROCS})
if(${MPIEXEC_EXAMPLE_NUMPROCS} GREATER 32)
  set(MPIEXEC_EXAMPLE_NUMPROCS 32)
endif()
set(MPIEXEC_EXAMPLE_NUMPROCS_SMALL ${MPIEXEC_MAX_NUMPROCS})
if(${MPIEXEC_EXAMPLE_NUMPROCS_SMALL} GREATER 8)
  set(MPIEXEC_EXAMPLE_NUMPROCS_SMALL 8)
endif()

set(INPUT_FILES ${INPUT_FILES}
  diffusion_steady_flame_solver/input.yml.in
  diffusion_steady_flame_solver/run.sh.in
  diffusion_unsteady_flame_solver/input.yml.in
  diffusion_unsteady_flame_solver/run.sh.in
  premixed_unsteady_flame_solver/input.yml.in
  premixed_unsteady_flame_solver/run.sh.in
  premixed_steady_flame_solver/input.yml.in
  premixed_steady_flame_solver/run.sh.in
  counterflow_unsteady_flame_solver/input.yml.in
  counterflow_unsteady_flame_solver/run.sh.in
  counterflow_steady_flame_solver/input.yml.in
  counterflow_steady_flame_solver/run.sh.in
  lewisGenerator/lewis_generator.yml.in
  lewisGenerator/run.sh.in
  variable_volume_gsa/h2_rcm_bruteforce.yml.in
  variable_volume_gsa/h2_rcm_lognormal.yml.in
  flame_speed/README.in
  flame_speed/steady/input.yml.in
  flame_speed/steady/sweep_steady.py.in
  flame_speed/steady/run.sh.in
  flame_speed/unsteady/input.yml.in
  flame_speed/unsteady/sweep_unsteady.py.in
  flame_speed/unsteady/run.sh.in
  flame_speed/pseudounsteady/input.yml.in
  flame_speed/pseudounsteady/sweep_pseudounsteady.py.in
  flame_speed/pseudounsteady/run.sh.in
  counterflow_flame_sweeps/premixed_unsteady/input.yml.in
  counterflow_flame_sweeps/premixed_unsteady/sweep_unsteady.py.in
  counterflow_flame_sweeps/premixed_unsteady/run.sh.in
  counterflow_flame_sweeps/premixed_steady/input.yml.in
  counterflow_flame_sweeps/premixed_steady/sweep_steady.py.in
  counterflow_flame_sweeps/premixed_steady/run.sh.in
  counterflow_flame_sweeps/diffusion_unsteady/input.yml.in
  counterflow_flame_sweeps/diffusion_unsteady/sweep_unsteady.py.in
  counterflow_flame_sweeps/diffusion_unsteady/run.sh.in
  counterflow_flame_sweeps/diffusion_steady/input.yml.in
  counterflow_flame_sweeps/diffusion_steady/sweep_steady.py.in
  counterflow_flame_sweeps/diffusion_steady/run.sh.in
  perturbAFactorGSA/perturbAFactorGSA_example.yml.in
  perturbAFactorGSA/run.sh.in
  variable_volume_gsa/run.sh.in
  flame_api_tester/run_parallel.sh.in
  surrogate_optimizer/idt_sweep/run.sh.in)

install(FILES parallel_examples.txt run_parallel.sh DESTINATION share/zerork/examples)
install(FILES run_all.sh DESTINATION share/zerork/examples)
endif(ENABLE_MPI)

foreach(INPUT_FILE ${INPUT_FILES})
string(REGEX REPLACE "\.in$" "" OUTPUT_FILE ${INPUT_FILE})
configure_file(${INPUT_FILE} ${OUTPUT_FILE} @ONLY)
get_filename_component(OUT_DIR ${INPUT_FILE} DIRECTORY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE} DESTINATION share/zerork/examples/${OUT_DIR})
get_filename_component(OUT_DIR_PARENT ${OUT_DIR} DIRECTORY)
install(DIRECTORY ${OUT_DIR} DESTINATION share/zerork/examples/${OUT_DIR_PARENT}
        PATTERN "CMakeLists.txt" EXCLUDE
        REGEX "\.in$" EXCLUDE)
endforeach()

install(FILES serial_examples.txt run_serial.sh DESTINATION share/zerork/examples)
