#pragma once

#include <Eigen/Dense>
#include <vector>

namespace reconstruction
{
    namespace posegraph
    {        
        std::tuple<Eigen::Matrix3d, Eigen::Vector3d, double> findOptimalAlignment(
            const std::vector<Eigen::Vector3d> &kPositions_,
            const std::vector<Eigen::Vector3d> &kReferencePositions_)
        {
            // Ensure the sizes of the position vectors match
            assert(kPositions_.size() == kReferencePositions_.size());

            // 1. Compute the centroids of both point sets.
            Eigen::Vector3d centroid1 = Eigen::Vector3d::Zero();
            Eigen::Vector3d centroid2 = Eigen::Vector3d::Zero();

            for (size_t i = 0; i < kPositions_.size(); i++) 
            {
                centroid1 += kPositions_[i];
                centroid2 += kReferencePositions_[i];
            }

            centroid1 /= double(kPositions_.size());
            centroid2 /= double(kReferencePositions_.size());

            // 2. Center both point sets.
            std::vector<Eigen::Vector3d> centered1(kPositions_.size());
            std::vector<Eigen::Vector3d> centered2(kReferencePositions_.size());
            for (size_t i = 0; i < kPositions_.size(); i++) 
            {
                centered1[i] = kPositions_[i] - centroid1;
                centered2[i] = kReferencePositions_[i] - centroid2;
            }

            // 3. Compute the norms of centered point sets.
            double norm1 = 0.0;
            double norm2 = 0.0;
            for (size_t i = 0; i < centered1.size(); i++) 
            {
                norm1 += centered1[i].squaredNorm();
                norm2 += centered2[i].squaredNorm();
            }

            // 4. Compute the scale.
            double scale = std::sqrt(norm2 / norm1);

            // 5. Apply the scale to the first point set.
            for (Eigen::Vector3d& point : centered1) 
                point *= scale;

            // 6. Compute the cross-covariance matrix.
            Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
            for (size_t i = 0; i < kPositions_.size(); i++) {
                H += centered1[i] * centered2[i].transpose();
            }

            // 7. Compute the singular value decomposition (SVD) of H.
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);

            // 8. Compute the rotation matrix.
            Eigen::Matrix3d rotation = (svd.matrixU() * svd.matrixV().transpose()).transpose();

            // 9. Compute the translation.
            Eigen::Vector3d translation = centroid2 - scale * rotation * centroid1;

            return std::make_tuple(rotation, translation, scale);
        }
    }
}