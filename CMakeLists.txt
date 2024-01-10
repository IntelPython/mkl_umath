cmake_minimum_required(VERSION 3.27...3.28 FATAL_ERROR)

cmake_policy(SET CMP0135 NEW)

project(mkl_umath
  LANGUAGES C
  DESCRIPTION "mkl_umath module"
)

option(OPTIMIZATION_REPORT
  "Whether to generate optimization vectorization report"
  OFF
)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(NumPy REQUIRED)

set(CYTHON_FLAGS "-t -w \"${CMAKE_SOURCE_DIR}\"")
find_package(Cython REQUIRED)

set(MKL_ARCH intel64)
set(MKL_LINK sdl)
set(MKL_THREADING intel_thread)
set(MKL_INTERFACE ilp64)
find_package(MKL REQUIRED)

if(WIN32)
 string(CONCAT WARNING_FLAGS
     "-Wall "
     "-Wextra "
     "-Winit-self "
     "-Wunused-function "
     "-Wuninitialized "
     "-Wmissing-declarations "
     "-Wstrict-prototypes "
     "-Wno-unused-parameter "
     "-Wno-implicit-function-declaration "
   )
   string(CONCAT SDL_FLAGS
     "/GS "
     "/DynamicBase "
   )
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Ox ${WARNING_FLAGS} ${SDL_FLAGS}")
   set(CMAKE_C_FLAGS_DEBUG
     "${CMAKE_C_FLAGS_DEBUG} ${WARNING_FLAGS} ${SDL_FLAGS} -O0 -g1 -DDEBUG"
   )
  set(MKL_UMATH_LINKER_OPTIONS "LINKER:/NXCompat;LINKER:/DynamicBase")
elseif(UNIX)
   string(CONCAT WARNING_FLAGS
     "-Wall "
     "-Wextra "
     "-Winit-self "
     "-Wunused-function "
     "-Wuninitialized "
     "-Wmissing-declarations "
     "-Wstrict-prototypes "
     "-Wno-unused-parameter "
     "-fdiagnostics-color=auto "
   )
   string(CONCAT SDL_FLAGS
     "-fstack-protector "
     "-fstack-protector-all "
     "-fpic "
     "-fPIC "
     "-D_FORTIFY_SOURCE=2 "
     "-Wformat "
     "-Wformat-security "
#     "-fno-strict-overflow "    # no-strict-overflow is implied by -fwrapv
     "-fno-delete-null-pointer-checks "
     "-fwrapv "
   )
   string(CONCAT CFLAGS
     "${WARNING_FLAGS}"
     "${SDL_FLAGS}"
   )
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 ${CFLAGS}")
   set(CMAKE_C_FLAGS_DEBUG
     "${CMAKE_C_FLAGS_DEBUG} ${CFLAGS} -O0 -g1 -DDEBUG"
   )
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-incompatible-function-pointer-types ${CFLAGS}")
  set(MKL_UMATH_LINKER_OPTIONS "LINKER:-z,noexecstack,-z,relro,-z,now")
else()
  message(FATAL_ERROR "Unsupported system.")
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
# set_property(GLOBAL PROPERTY GLOBAL_DEPENDS_DEBUG_MODE 1)
set(_linker_options ${MKL_UMATH_LINKER_OPTIONS})

set(_trgt mkl_umath_loops)
add_library(${_trgt} SHARED "mkl_umath/src/mkl_umath_loops.c")
set_target_properties(${_trgt} PROPERTIES CMAKE_POSITION_INDEPENDENT_CODE ON)
target_include_directories(${_trgt} PRIVATE "mkl_umath/src/" ${NumPy_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR})
target_link_libraries(${_trgt} PRIVATE mkl_rt ${Python_LIBRARIES})
target_link_options(${_trgt} PRIVATE ${_linker_options})
target_compile_options(${_trgt} PRIVATE -fveclib=SVML)
target_compile_options(${_trgt} PRIVATE -fvectorize)
if(OPTIMIZATION_REPORT)
  target_compile_options(${_trgt} PRIVATE -qopt-report=3)
endif()
install(TARGETS ${_trgt}
  LIBRARY DESTINATION mkl_umath
  ARCHIVE DESTINATION mkl_umath
  RUNTIME DESTINATION mkl_umath
)

Python_add_library(_ufuncs MODULE WITH_SOABI "mkl_umath/src/ufuncsmodule.c" "mkl_umath/src/__umath_generated.c")
target_include_directories(_ufuncs PRIVATE "mkl_umath/src" ${NumPy_INCLUDE_DIR} ${MKL_INCLUDE_DIR})
target_compile_definitions(_ufuncs PUBLIC NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
target_link_options(_ufuncs PRIVATE ${_linker_options})
target_link_libraries(_ufuncs PRIVATE mkl_umath_loops)
if (UNIX)
  set_target_properties(_ufuncs PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
install(TARGETS _ufuncs LIBRARY DESTINATION mkl_umath)

add_cython_target(_patch "mkl_umath/src/_patch.pyx" C OUTPUT_VAR _generated_src)
Python_add_library(_patch MODULE WITH_SOABI ${_generated_src})
target_include_directories(_patch PRIVATE "mkl_umath/src/" ${NumPy_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR})
target_compile_definitions(_patch PUBLIC NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
target_link_libraries(_patch PRIVATE mkl_umath_loops)
if (UNIX)
  set_target_properties(_patch PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
install(TARGETS _patch LIBRARY DESTINATION mkl_umath)
