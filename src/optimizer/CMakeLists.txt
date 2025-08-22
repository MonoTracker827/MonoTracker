cmake_minimum_required(VERSION 3.10)
project(PnPMutualRefine LANGUAGES CUDA CXX)
# debug
set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -g")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -dc 
    -gencode=arch=compute_60,code=sm_60 
    -gencode=arch=compute_61,code=sm_61 
    -gencode=arch=compute_70,code=sm_70 
    -gencode=arch=compute_75,code=sm_75 
    -gencode=arch=compute_80,code=sm_80 
    -gencode=arch=compute_86,code=sm_86 
    -O3 -std=c++14 -use_fast_math --default-stream per-thread 
    --expt-relaxed-constexpr"
)


find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(
    ${CERES_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIR}
)

# show the include directories
message(STATUS "CERES_INCLUDE_DIRS: ${CERES_INCLUDE_DIRS}")
message(STATUS "EIGEN3_INCLUDE_DIR: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "pybind11_INCLUDE_DIR: ${pybind11_INCLUDE_DIR}")

# ********************** Pybind **********************
# add_library(PnPMutualRefine SHARED PnPMutualRefineOpt.cpp)
pybind11_add_module(pyPnPMutualRefine PnPMutualRefineOpt.cpp)

target_link_libraries(
    pyPnPMutualRefine PRIVATE Ceres::ceres Eigen3::Eigen ${CUDA_LIBRARIES}
)
