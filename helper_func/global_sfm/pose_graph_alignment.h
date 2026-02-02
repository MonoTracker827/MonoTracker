#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <iostream>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "alignment_utils.h"
#include "alignment_losses.h"

namespace reconstruction
{
    namespace posegraph
    {        
        Eigen::Matrix3d alignOrientations(
            const std::vector<Eigen::Matrix3d> &kOrientations_,
            const std::vector<Eigen::Matrix3d> &kReferenceOrientations_,
            const double kLossThreshold_,
            const int kMaximumIterations_,
            const int kNumThreads_,
            const bool kInitialization_,
            const bool kNumericalOptimization_,
            const bool kLog_)
        {
            if (!kInitialization_ && !kNumericalOptimization_)
            {
                std::cerr << "Neither the initialization nor the numerical optimization was turned on. No alignment is performed." << std::endl;
                return Eigen::Matrix3d::Identity();
            }

            // Set up the ceres problem and specify the ownership of the loss function
            ceres::Problem::Options ceresOptions;
            // Because the loss function is constructed by Python, ceres should not destruct it.
            ceresOptions.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
            std::unique_ptr<ceres::Problem> problem(new ceres::Problem(ceresOptions));
            Eigen::Vector3d rotation;
            rotation << 0, 0, 0;

            // Calculate a pre-alignment by taking the average relative rotation in its angle-axis representation
            if (kInitialization_)
            {
                Eigen::Matrix3d relativeRotation;
                for (size_t idx = 0; idx < kOrientations_.size(); ++idx)
                {
                    relativeRotation = kReferenceOrientations_[idx].transpose() * kOrientations_[idx];
                    Eigen::AngleAxisd angleAxis(relativeRotation);
                    rotation += angleAxis.angle() * angleAxis.axis();
                }
                rotation /= kOrientations_.size();
            }

            if (!kNumericalOptimization_)
            {
                Eigen::AngleAxisd angleAxis(rotation.norm(), rotation.normalized());
                return angleAxis.toRotationMatrix();
            }

            std::unique_ptr<ceres::LossFunction> lossFunction;
            lossFunction.reset(new ceres::CauchyLoss(kLossThreshold_));

            for (size_t idx = 0; idx < kOrientations_.size(); ++idx)
            {
                // Create the cost function
                ceres::CostFunction *costFunction = 
                    RotationErrorAngleAxis::Create(kOrientations_[idx], kReferenceOrientations_[idx]);

                // Add the cost function as a residual block to the problem
                problem->AddResidualBlock(
                    costFunction, lossFunction.get(), rotation.data());
            }

            // The problem should be relatively sparse so sparse cholesky is a good
            // choice.
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            options.max_num_iterations = kMaximumIterations_;
            options.num_threads = kNumThreads_;

            // Solve the optimization problem
            ceres::Solver::Summary summary;
            ceres::Solve(options, problem.get(), &summary);
            if (kLog_)
                std::cout << summary.FullReport();
            
            Eigen::AngleAxisd angleAxis(rotation.norm(), rotation.normalized());
            return angleAxis.toRotationMatrix();
        }
    }
}