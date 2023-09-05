cmake_minimum_required(VERSION 3.21...3.25 FATAL_ERROR)

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24")
  cmake_policy(SET CMP0135 NEW)
endif()

project(mkl_umath
  LANGUAGES C
  DESCRIPTION "mkl_umath module"
)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(NumPy REQUIRED)
find_package(PythonExtensions REQUIRED)

set(CYTHON_FLAGS "-t -w \"${CMAKE_SOURCE_DIR}\"")
find_package(Cython REQUIRED)

set(MKL_ARCH intel64)
set(MKL_LINK sdl)
set(MKL_THREADING intel_thread)
set(MKL_INTERFACE ilp64)
# MKL_ARCH: None, set to ` intel64` by default
# MKL_ROOT /localdisk/work/aguzmanb/Development/miniconda3.py310/envs/numpy_umath_prefix.v5
# MKL_DPCPP_LINK: None, set to ` dynamic` by default
# MKL_LINK: None, set to ` dynamic` by default
# MKL_DPCPP_INTERFACE_FULL: None, set to ` intel_ilp64` by default
# MKL_INTERFACE_FULL: None, set to ` intel_ilp64` by default
# MKL_DPCPP_THREADING: None, set to ` tbb_thread` by default
# MKL_THREADING: None, set to ` intel_thread` by default
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
target_include_directories(${_trgt} PRIVATE "mkl_umath/src/" ${NumPy_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR} "C:/Users/aguzmanb/Development/mambaforge/envs/mkl_umath_prefix/Library/include")
target_link_libraries(${_trgt} PRIVATE mkl_rt ${Python_LIBRARIES})
if (WIN32)
   target_link_directories(${_trgt} PRIVATE "C:/Users/aguzmanb/Development/mambaforge/envs/mkl_umath_prefix/Libs")
endif()
target_link_options(${_trgt} PRIVATE ${_linker_options})
install(TARGETS ${_trgt} LIBRARY DESTINATION mkl_umath)

add_library(_ufuncs MODULE "mkl_umath/src/ufuncsmodule.c" "mkl_umath/src/__umath_generated.c")
target_include_directories(_ufuncs PRIVATE "mkl_umath/src" ${NumPy_INCLUDE_DIR} ${MKL_INCLUDE_DIR})
target_compile_definitions(_ufuncs PUBLIC NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
target_link_options(_ufuncs PRIVATE ${_linker_options})
target_link_libraries(_ufuncs mkl_umath_loops)
python_extension_module(_ufuncs)
if (UNIX)
  set_target_properties(_ufuncs PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
install(TARGETS _ufuncs LIBRARY DESTINATION mkl_umath)

add_cython_target(_patch "mkl_umath/src/_patch.pyx" C OUTPUT_VAR _generated_src)
add_library(_patch MODULE ${_generated_src})
target_include_directories(_patch PRIVATE "mkl_umath/src/" ${NumPy_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR})
target_compile_definitions(_patch PUBLIC NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
target_link_libraries(_patch mkl_umath_loops)
python_extension_module(_patch)
if (UNIX)
  set_target_properties(_patch PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
install(TARGETS _patch LIBRARY DESTINATION mkl_umath)
