#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>

#include "pose_graph_alignment.h"

Eigen::Matrix3d align_orientations(
    const std::vector<Eigen::Matrix3d> &orientations,
    const std::vector<Eigen::Matrix3d> &reference_orientations,
    const double loss_threshold,
    const int maximum_iterations,
    const int num_threads,
    const bool initialization,
    const bool numerical_optimization,
    const bool log)
{
    return reconstruction::posegraph::alignOrientations(
        orientations, 
        reference_orientations,
        loss_threshold,
        maximum_iterations,
        num_threads,
        initialization,
        numerical_optimization,
        log);
}