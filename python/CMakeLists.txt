

if(Python3_FOUND)
add_subdirectory(flame_solvers)
endif()

set(INPUT_FILES zerork/config.py.in)

foreach(INPUT_FILE ${INPUT_FILES})
string(REGEX REPLACE "\.in$" "" OUTPUT_FILE ${INPUT_FILE})
configure_file(${INPUT_FILE} ${OUTPUT_FILE} @ONLY)
get_filename_component(OUT_DIR ${INPUT_FILE} DIRECTORY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE} DESTINATION share/zerork/python/${OUT_DIR})
endforeach()

install(DIRECTORY zerork DESTINATION share/zerork/python
        REGEX "\.in$" EXCLUDE)

install(FILES pyproject.toml MANIFEST.in DESTINATION share/zerork/python)


install(CODE "MESSAGE(\"-- Installing: Making python environment ...\")")
install(CODE "execute_process(COMMAND \"${Python3_EXECUTABLE}\" -m venv env
                              WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/share/zerork/python)")
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E env PYTHONNOUSERSITE=1 env/bin/python -m pip install -I pip .
                              OUTPUT_FILE package_build.log
                              WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/share/zerork/python)")
