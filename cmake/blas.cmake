
if(NOT DEFINED ZERORK_HAVE_MKL)
  if(DEFINED ENV{MKLROOT})
    set(ZERORK_HAVE_MKL ON CACHE BOOL "" FORCE)
    mark_as_advanced(ZERORK_HAVE_MKL)
    set(ZERORK_MKLROOT "$ENV{MKLROOT}" CACHE FILEPATH "" FORCE)
    mark_as_advanced(ZERORK_MKLROOT)
  elseif(DEFINED ENV{ONEAPI_ROOT})
    set(ZERORK_HAVE_MKL ON CACHE BOOL "" FORCE)
    mark_as_advanced(ZERORK_HAVE_MKL)
    set(ZERORK_MKLROOT "$ENV{ONEAPI_ROOT}/mkl/latest" CACHE FILEPATH "" FORCE)
    mark_as_advanced(ZERORK_MKLROOT)
  else()
    set(ZERORK_HAVE_MKL OFF CACHE BOOL "" FORCE)
    mark_as_advanced(ZERORK_HAVE_MKL)
  endif()
endif()

if(${ZERORK_HAVE_MKL})
  add_library(mkl INTERFACE IMPORTED GLOBAL)
  #These removals are to work around an issue with mkl `module`s on LLNL's
  #clusters that set environment variables which cmake acts upon on first configure
  #but does not refresh on re-configuration.  This can mean that different paths
  #are used depending on environment state (independent of configuration variables)
  #which can cause the build to fail if the user forgets to `module load mkl` when
  #(re-)building the code.
  if(DEFINED CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES)
    list(REMOVE_ITEM CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES "${ZERORK_MKLROOT}/include")
  endif()
  if(DEFINED CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
    list(REMOVE_ITEM CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "${ZERORK_MKLROOT}/include")
  endif()
  if(DEFINED CMAKE_C_IMPLICIT_LINK_DIRECTORIES)
    list(REMOVE_ITEM CMAKE_C_IMPLICIT_LINK_DIRECTORIES "${ZERORK_MKLROOT}/lib/intel64")
  endif()
  if(DEFINED CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES)
    list(REMOVE_ITEM CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES "${ZERORK_MKLROOT}/lib/intel64")
  endif()
  target_include_directories(mkl INTERFACE "${ZERORK_MKLROOT}/include")
  #Below pulled from MKL Link Advisor 2018.0.
  if(WIN32)
    set(MKL_LIBS_DIR lib)
    if( NOT(EXISTS "${ZERORK_MKLROOT}/${MKL_LIBS_DIR}/mkl_core.lib") )
      set(MKL_LIBS_DIR win-x64)
      if( NOT(EXISTS "${ZERORK_MKLROOT}/${MKL_LIBS_DIR}/mkl_core.lib") )
        message( FATAL_ERROR "MKLROOT set but can't find MKL libs." )
      endif()
    endif()
    target_link_libraries(mkl INTERFACE
      "${ZERORK_MKLROOT}/${MKL_LIBS_DIR}/mkl_intel_lp64.lib"
      "${ZERORK_MKLROOT}/${MKL_LIBS_DIR}/mkl_sequential.lib"
      "${ZERORK_MKLROOT}/${MKL_LIBS_DIR}/mkl_core.lib")
  else()
    target_link_libraries(mkl INTERFACE
      -Wl,--start-group
      "${ZERORK_MKLROOT}/lib/intel64/libmkl_intel_lp64.a"
      "${ZERORK_MKLROOT}/lib/intel64/libmkl_sequential.a"
      "${ZERORK_MKLROOT}/lib/intel64/libmkl_core.a"
      -Wl,--end-group
      pthread m dl)
  endif()
  add_library(blas ALIAS mkl)
else()
  find_package(BLAS REQUIRED)
  add_library(blas INTERFACE IMPORTED GLOBAL)
  target_link_libraries(blas INTERFACE ${BLAS_LIBRARIES})
endif()

