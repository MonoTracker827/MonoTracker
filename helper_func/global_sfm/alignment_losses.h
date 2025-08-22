#pragma once

#include <Eigen/Dense>
#include <vector>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace reconstruction
{
    namespace posegraph
    {        
        struct PositionError
        {
            PositionError(
                const Eigen::Vector3d &kPosition_, 
                const Eigen::Vector3d &kPositionReference_);

            PositionError(
                const Eigen::Vector3d &kPosition_, 
                const Eigen::Vector3d &kPositionReference_,
                const double &kWeight_);

            // The error is given by the rotation loop error as specified above. We return
            // 3 residuals to give more opportunity for optimization.
            template <typename T>
            bool operator()(const T *rotation,
                const T *translation,
                const T *scale, 
                T *residuals) const;

            static ceres::CostFunction *Create(
                const Eigen::Vector3d &kPosition_, 
                const Eigen::Vector3d &kPositionReference_);

            static ceres::CostFunction *Create(
                const Eigen::Vector3d &kPosition_, 
                const Eigen::Vector3d &kPositionReference_,
                const double &kWeight_);

            const Eigen::Vector3d position_;
            const Eigen::Vector3d positionReference_;
            const double weight_;
        };
        
        template <typename T>
        bool PositionError::operator()(
            const T *rotation,
            const T *translation,
            const T *scale,
            T *residuals) const
        {
            // Convert angle axis rotations to rotation matrices.
            Eigen::Matrix<T, 3, 3> rotationMat;
            ceres::AngleAxisToRotationMatrix(
                rotation, ceres::ColumnMajorAdapter3x3(rotationMat.data()));
                
            // Convert to translation vector
            Eigen::Matrix<T, 3, 1> translationVec;
            translationVec << translation[0], translation[1], translation[2];

            Eigen::Matrix<T, 3, 1> newPosition =
                (*scale) * rotationMat * position_.cast<T>() + translationVec;
            Eigen::Matrix<T, 3, 1> refPosition =
                positionReference_.cast<T>();

            residuals[0] = weight_ * (newPosition(0) - refPosition(0));
            residuals[1] = weight_ * (newPosition(1) - refPosition(1));
            residuals[2] = weight_ * (newPosition(2) - refPosition(2));

            return true;
        }

        PositionError::PositionError(
            const Eigen::Vector3d &kPosition_, 
            const Eigen::Vector3d &kPositionReference_) : 
            position_(kPosition_), 
            positionReference_(kPositionReference_),
            weight_(1.0)
        {
            
        }

        PositionError::PositionError(
            const Eigen::Vector3d &kPosition_, 
            const Eigen::Vector3d &kPositionReference_,
            const double &kWeight_) : 
            position_(kPosition_), 
            positionReference_(kPositionReference_),
            weight_(kWeight_)
        {
            
        }

        ceres::CostFunction *PositionError::Create(
            const Eigen::Vector3d &kPosition_, 
            const Eigen::Vector3d &kPositionReference_)
        {
            return new ceres::AutoDiffCostFunction<PositionError, 3, 3, 3, 1>(
                new PositionError(kPosition_, kPositionReference_));
        }

        ceres::CostFunction *PositionError::Create(
            const Eigen::Vector3d &kPosition_, 
            const Eigen::Vector3d &kPositionReference_,
            const double &kWeight_)
        {
            return new ceres::AutoDiffCostFunction<PositionError, 3, 3, 3, 1>(
                new PositionError(kPosition_, kPositionReference_, kWeight_));
        }

        struct RotationErrorAngleAxis
        {
            RotationErrorAngleAxis(
                const Eigen::Matrix3d &kRotation_, 
                const Eigen::Matrix3d &kRotationReference_);

            RotationErrorAngleAxis(
                const Eigen::Matrix3d &kRotation_, 
                const Eigen::Matrix3d &kRotationReference_,
                const double &kWeight_);

            // The error is given by the rotation loop error as specified above. We return
            // 3 residuals to give more opportunity for optimization.
            template <typename T>
            bool operator()(const T *rotation, T *residuals) const;

            static ceres::CostFunction *Create(
                const Eigen::Matrix3d &kRotation_, 
                const Eigen::Matrix3d &kRotationReference_);

            static ceres::CostFunction *Create(
                const Eigen::Matrix3d &kRotation_, 
                const Eigen::Matrix3d &kRotationReference_,
                const double &kWeight_);

            const Eigen::Matrix3d rotation_;
            const Eigen::Matrix3d rotationReference_;
            const double weight_;
        };

        template <typename T>
        bool RotationErrorAngleAxis::operator()(
            const T *rotation,
            T *residuals) const
        {
            // Convert angle axis rotations to rotation matrices.
            Eigen::Matrix<T, 3, 3> rotationMat;
            ceres::AngleAxisToRotationMatrix(
                rotation, ceres::ColumnMajorAdapter3x3(rotationMat.data()));
            // Calculate the new rotation matrix
            Eigen::Matrix<T, 3, 3> newRotation = 
                rotation_.cast<T>() * rotationMat.transpose();
            // Calculate the loop rotation matrix
            const Eigen::Matrix<T, 3, 3> errorRotationMat =
                newRotation * rotationReference_.cast<T>().transpose();
           Eigen::Matrix<T, 3, 1> errorRotation;
            ceres::RotationMatrixToAngleAxis(
                ceres::ColumnMajorAdapter3x3(errorRotationMat.data()),
                errorRotation.data());

            residuals[0] = weight_ * errorRotation(0);
            residuals[1] = weight_ * errorRotation(1);
            residuals[2] = weight_ * errorRotation(2);

            return true;
        }

        RotationErrorAngleAxis::RotationErrorAngleAxis(
            const Eigen::Matrix3d &kRotation_, 
            const Eigen::Matrix3d &kRotationReference_) : 
            rotation_(kRotation_), 
            rotationReference_(kRotationReference_),
            weight_(1.0)
        {
            
        }

        RotationErrorAngleAxis::RotationErrorAngleAxis(
            const Eigen::Matrix3d &kRotation_, 
            const Eigen::Matrix3d &kRotationReference_,
            const double &kWeight_) : 
            rotation_(kRotation_), 
            rotationReference_(kRotationReference_),
            weight_(kWeight_)
        {
            
        }

        ceres::CostFunction *RotationErrorAngleAxis::Create(
            const Eigen::Matrix3d &kRotation_, 
            const Eigen::Matrix3d &kRotationReference_)
        {
            return new ceres::AutoDiffCostFunction<RotationErrorAngleAxis, 3, 3>(
                new RotationErrorAngleAxis(kRotation_, kRotationReference_));
        }

        ceres::CostFunction *RotationErrorAngleAxis::Create(
            const Eigen::Matrix3d &kRotation_, 
            const Eigen::Matrix3d &kRotationReference_,
            const double &kWeight_)
        {
            return new ceres::AutoDiffCostFunction<RotationErrorAngleAxis, 3, 3>(
                new RotationErrorAngleAxis(kRotation_, kRotationReference_, kWeight_));
        }
    }
}
