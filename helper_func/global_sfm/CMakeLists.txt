cmake_minimum_required(VERSION 3.10)
project(PnPMutualRefine LANGUAGES CXX)
# debug
set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -g")

find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(
    ${CERES_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${pybind11_INCLUDE_DIR}
)

file(GLOB_RECURSE SRC_PYRECUTILS
    "*.cpp"
)

# show the include directories
message(STATUS "CERES_INCLUDE_DIRS: ${CERES_INCLUDE_DIRS}")
message(STATUS "EIGEN3_INCLUDE_DIR: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "pybind11_INCLUDE_DIR: ${pybind11_INCLUDE_DIR}")

# ********************** Pybind **********************
pybind11_add_module(pyglobalsfm bindings.cpp ${SRC_PYRECUTILS})

target_link_libraries(
    pyglobalsfm PRIVATE Ceres::ceres Eigen3::Eigen
)
