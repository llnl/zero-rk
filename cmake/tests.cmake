
Include(GoogleTest)
#https://cliutils.gitlab.io/modern-cmake/chapters/testing/googletest.html
# (accessed 20200513)

enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.11.0
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

add_custom_target(tests_build ${CMAKE_CTEST_COMMAND} -C $<CONFIG>)

function(zerork_add_gtests TESTNAME)
    #message(STATUS "Adding new gtest target ${TESTNAME}")

    cmake_parse_arguments(ZERORK_ADD_GTESTS "" "" "SOURCES;ARGUMENTS;LINK_LIBRARIES" ${ARGN} )

    if(NOT DEFINED ZERORK_ADD_GTESTS_SOURCES)
      message(FATAL_ERROR "zerork_add_gtests needs SOURCES")
    endif()

    # create an exectuable in which the tests will be stored
    add_executable(${TESTNAME} ${ZERORK_ADD_GTESTS_SOURCES})
    # link the Google test infrastructure, mocking library, and a default main fuction to
    # the test executable.  Remove g_test_main if writing your own main function.
    target_link_libraries(${TESTNAME} gtest gmock gtest_main)
    if(DEFINED ZERORK_ADD_GTESTS_LINK_LIBRARIES)
      target_link_libraries(${TESTNAME} ${ZERORK_ADD_GTESTS_LINK_LIBRARIES})
    endif()

    if(NOT DEFINED ZERORK_ADD_GTESTS_ARGUMENTS)
      set(ZERORK_ADD_GTESTS_ARGUMENTS "")
    endif()
    # gtest_discover_tests replaces gtest_add_tests,
    # see https://cmake.org/cmake/help/v3.10/module/GoogleTest.html for more options to pass to it
    gtest_discover_tests(${TESTNAME}
                         EXTRA_ARGS ${ZERORK_ADD_GTESTS_ARGUMENTS}
			 WORKING_DIRECTORY $<TARGET_FILE_DIR:${TESTNAME}>
			 DISCOVERY_TIMEOUT 30
                         PROPERTIES ENVIRONMENT "ZERORK_DATA_DIR=${ZERORK_DATA_DIR}")

    add_dependencies(tests_build ${TESTNAME})
endfunction()

function(zerork_add_test TESTNAME)
    #message(STATUS "Adding new test target ${TESTNAME}")
    cmake_parse_arguments(ZERORK_ADD_TEST "" "" "SOURCES;ARGUMENTS;LINK_LIBRARIES" ${ARGN})

    if(NOT DEFINED ZERORK_ADD_TEST_SOURCES)
      message(FATAL_ERROR "zerork_add_test needs SOURCES")
    endif()

    # create an exectuable in which the tests will be stored
    add_executable(${TESTNAME} ${ZERORK_ADD_TEST_SOURCES})

    if(DEFINED ZERORK_ADD_TEST_LINK_LIBRARIES)
      target_link_libraries(${TESTNAME} ${ZERORK_ADD_TEST_LINK_LIBRARIES})
    endif()

    if(NOT DEFINED ZERORK_ADD_TEST_ARGUMENTS)
      set(ZERORK_ADD_TEST_ARGUMENTS "")
    endif()
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME} ${ZERORK_ADD_TEST_ARGUMENTS})

    set_tests_properties(${TESTNAME} PROPERTIES 
                        ENVIRONMENT "ZERORK_DATA_DIR=${ZERORK_DATA_DIR}")

    add_dependencies(tests_build ${TESTNAME})
endfunction()

include(CTest)


