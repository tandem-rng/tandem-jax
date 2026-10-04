cmake_minimum_required(VERSION 3.24)

# all-major runs on any GPU the toolkit supports. Set CMAKE_CUDA_ARCHITECTURES (e.g. 80) to build faster.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()

project(tandem_jax_cuda LANGUAGES CXX CUDA)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import jaxlib, os; print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))"
  OUTPUT_VARIABLE XLA_FFI_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)

add_library(tandem_ffi SHARED tandem_ffi.cu)
target_include_directories(tandem_ffi PRIVATE include)
target_include_directories(tandem_ffi SYSTEM PRIVATE "${XLA_FFI_INCLUDE}")
target_compile_features(tandem_ffi PRIVATE cuda_std_20)
target_compile_options(tandem_ffi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
# A static runtime keeps the library independent of the CUDA runtime that jaxlib loads.
set_target_properties(tandem_ffi PROPERTIES CUDA_RUNTIME_LIBRARY Static)
install(TARGETS tandem_ffi LIBRARY DESTINATION tandem_jax_cuda)
