project(libshm LANGUAGES CXX)
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(TORCH_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../../)

add_library(shm OBJECT core.cpp)
set_target_properties(shm PROPERTIES CXX_STANDARD 17)

target_include_directories(shm PUBLIC
  ${TORCH_ROOT}/torch/lib # provides "libshm/libshm.h"
)
target_link_libraries(shm INTERFACE torch_cpu rt)

add_executable(torch_shm_manager manager.cpp)

target_link_libraries(torch_shm_manager PUBLIC shm)

set_target_properties(torch_shm_manager PROPERTIES
  INSTALL_RPATH "${_rpath_portable_origin}/../lib")

install(TARGETS torch_shm_manager DESTINATION "bin")
