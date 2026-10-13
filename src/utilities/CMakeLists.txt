
add_library(zerorkutilities distribution.cpp sort_vector.cpp sequential_file_matrix.cpp
            file_utilities.cpp math_utilities.cpp string_utilities.cpp)

target_include_directories(zerorkutilities PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                                  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
                                                  $<INSTALL_INTERFACE:include>)

set(public_headers distribution.h
    sequential_file_matrix.h sort_vector.h
    file_utilities.h math_utilities.h string_utilities.h)

set_target_properties(zerorkutilities PROPERTIES
                      PUBLIC_HEADER  "${public_headers}")
install(TARGETS zerorkutilities
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include/utilities)

add_library(zerork_restart_writer INTERFACE)
target_include_directories(zerork_restart_writer INTERFACE .)

if(ENABLE_MPI)
add_mpi_library(zerorkmpiutilities mpi_utilities.cpp)

target_include_directories(zerorkmpiutilities PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                                  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
                                                  $<INSTALL_INTERFACE:include>)

set_target_properties(zerorkmpiutilities PROPERTIES
                      PUBLIC_HEADER  "mpi_utilities.h")

install(TARGETS zerorkmpiutilities
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include/utilities)
endif(ENABLE_MPI)
