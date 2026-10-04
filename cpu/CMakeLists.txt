cmake_minimum_required(VERSION 3.18)

project(tandem_jax_cpu LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import jaxlib, os; print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))"
  OUTPUT_VARIABLE XLA_FFI_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)

# A module loads with dlopen only, and CMake names it .so on macOS as well as Linux.
add_library(tandem_ffi MODULE tandem_ffi.cc tandem/tandem.c)
target_include_directories(tandem_ffi PRIVATE .)
target_include_directories(tandem_ffi SYSTEM PRIVATE "${XLA_FFI_INCLUDE}")
target_compile_features(tandem_ffi PRIVATE cxx_std_17 c_std_11)
# The flags of tandem-c's Makefile: unfused arithmetic keeps the normals bit exact on every compiler.
set_source_files_properties(tandem/tandem.c PROPERTIES COMPILE_OPTIONS "-O2;-ffp-contract=off")
set_target_properties(tandem_ffi PROPERTIES PREFIX "lib")
install(TARGETS tandem_ffi LIBRARY DESTINATION tandem_jax_cpu)
