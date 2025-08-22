#pragma once
#include <cmath>
#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/rotation.h>
#include <ceres/autodiff_cost_function.h>
// reprojection error
// point fixed, camera pose variable
typedef Eigen::Vector3d uvd_feature;
typedef Eigen::Vector3d coeff_list;
typedef Eigen::Matrix<double, 3, 3> intrinsic_matrix;
typedef Eigen::Vector3d Translation;
typedef std::array<int, 4> neighbor_index; // x1 x2 y1 y2


struct BilinerInterpolationError
{
    const double q_u, q_v;
    const int grid_size;
    const int x1, y1, x2, y2;
    BilinerInterpolationError(
        const double q_u, const double q_v, const int grid_size, 
        const int x1, const int y1, const int x2, const int y2)
        : q_u(q_u), q_v(q_v), grid_size(grid_size), x1(x1), y1(y1), x2(x2), y2(y2) {}
    
    template <typename T>
    bool operator()(
        const T *const value_x1y1, const T *const value_x2y1, 
        const T *const value_x1y2, const T *const value_x2y2, T *out_value
        ) const 
    {
        double u1 = x1 * grid_size;
        double v1 = y1 * grid_size;
        double u2 = x2 * grid_size;
        double v2 = y2 * grid_size;
        T value_y1 = ( value_x1y1[0] * (u2 - q_u) + value_x2y1[0] * (q_u - u1) )/ (u2 - u1);
        T value_y2 = ( value_x1y2[0] * (u2 - q_u) + value_x2y2[0] * (q_u - u1) )/ (u2 - u1);
        out_value[0] = ( value_y1 * (v2 - q_v) + value_y2 * (q_v - v1) )/ (v2 - v1);
        // print the bilinear interpolation result
        // std::cout << "BilinerInterpolationError: " << out_value[0] << std::endl;
        return true;
    }    
};



class PnPReprojectionErrorGlobalPoseGridVersion
{
private:
    const double u0, v0, mono_d0;
    const double u1, v1, mono_d1;
    const int frame0_x1, frame0_y1, frame0_x2, frame0_y2;
    const int frame1_x1, frame1_y1, frame1_x2, frame1_y2;
    const double fx, fy, cx, cy;
    const double w_spatial, w_disparity, w_reproj;
    const int grid_size;

    explicit PnPReprojectionErrorGlobalPoseGridVersion(
        const double u0, const double v0, const double mono_d0,
        const double u1, const double v1, const double mono_d1,
        const int frame0_x1, const int frame0_x2, const int frame0_y1, const int frame0_y2,
        const int frame1_x1, const int frame1_x2, const int frame1_y1, const int frame1_y2,
        const double fx, const double fy, const double cx, const double cy, 
        const double w_spatial, const double w_disparity, const double w_reproj, const int grid_size)
        : u0(u0), v0(v0), mono_d0(mono_d0), u1(u1), v1(v1), mono_d1(mono_d1), 
          frame0_x1(frame0_x1), frame0_y1(frame0_y1), frame0_x2(frame0_x2), frame0_y2(frame0_y2),
          frame1_x1(frame1_x1), frame1_y1(frame1_y1), frame1_x2(frame1_x2), frame1_y2(frame1_y2), 
          fx(fx), fy(fy), cx(cx), cy(cy), w_spatial(w_spatial), w_disparity(w_disparity), w_reproj(w_reproj), 
          grid_size(grid_size) {}

public: 
    static ceres::CostFunction *Create(
        const uvd_feature &uvd0, const uvd_feature &uvd1, 
        const intrinsic_matrix K, const coeff_list weights, const int grid_size, 
        const neighbor_index &frame0_neighbor, const neighbor_index &frame1_neighbor)
    {
        return (new ceres::AutoDiffCostFunction<PnPReprojectionErrorGlobalPoseGridVersion, 5, 4, 3, 4, 3, 
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 >(
            new PnPReprojectionErrorGlobalPoseGridVersion(
                uvd0[0], uvd0[1], uvd0[2], uvd1[0], uvd1[1], uvd1[2], 
                frame0_neighbor[0], frame0_neighbor[1], frame0_neighbor[2], frame0_neighbor[3],
                frame1_neighbor[0], frame1_neighbor[1], frame1_neighbor[2], frame1_neighbor[3],
                K(0, 0), K(1, 1), K(0, 2), K(1, 2), weights[0], weights[1], weights[2], grid_size
                )
            )   
        );
    }


    template <typename T>
    bool operator()(
        const T *const q_0, const T *const t_0,
        const T *const q_1, const T *const t_1,
        // frame 0 - 4 neighbors scale and shift
        const T *const a0_x1y1, const T *const b0_x1y1, const T *const a0_x2y1, const T *const b0_x2y1,
        const T *const a0_x1y2, const T *const b0_x1y2, const T *const a0_x2y2, const T *const b0_x2y2,
        // frame 1 - 4 neighbors scale and shift
        const T *const a1_x1y1, const T *const b1_x1y1, const T *const a1_x2y1, const T *const b1_x2y1,
        const T *const a1_x1y2, const T *const b1_x1y2, const T *const a1_x2y2, const T *const b1_x2y2, 
        T *residuals
        ) const
    {
        T q_0to1[4]; // q_0to1 = q_1 * q_0^-1
        T inv_q_0[4] = {q_0[0], -q_0[1], -q_0[2], -q_0[3]};
        ceres::QuaternionProduct(q_1, inv_q_0, q_0to1);
        T t_0in1[3];
        ceres::QuaternionRotatePoint(q_0to1, t_0, t_0in1);
        T t_0to1[3] = {t_1[0] - t_0in1[0], t_1[1] - t_0in1[1], t_1[2] - t_0in1[2]};

        // bilinear interpolation, u,v is query point
        BilinerInterpolationError bErr_a0(u0, v0, grid_size, frame0_x1, frame0_y1, frame0_x2, frame0_y2);
        BilinerInterpolationError bErr_b0(u0, v0, grid_size, frame0_x1, frame0_y1, frame0_x2, frame0_y2);
        BilinerInterpolationError bErr_a1(u1, v1, grid_size, frame1_x1, frame1_y1, frame1_x2, frame1_y2);
        BilinerInterpolationError bErr_b1(u1, v1, grid_size, frame1_x1, frame1_y1, frame1_x2, frame1_y2);

        T a0[1]; bErr_a0(a0_x1y1, a0_x2y1, a0_x1y2, a0_x2y2, a0);
        T b0[1]; bErr_b0(b0_x1y1, b0_x2y1, b0_x1y2, b0_x2y2, b0);
        T a1[1]; bErr_a1(a1_x1y1, a1_x2y1, a1_x1y2, a1_x2y2, a1);
        T b1[1]; bErr_b1(b1_x1y1, b1_x2y1, b1_x1y2, b1_x2y2, b1);

        T d0 = a0[0] * mono_d0 + b0[0];
        T d1 = a1[0] * mono_d1 + b1[0];

        T p3d_0[3];
        p3d_0[0] = d0 * (u0 - cx) / fx;
        p3d_0[1] = d0 * (v0 - cy) / fy;
        p3d_0[2] = d0;
        T p3d_1[3];
        p3d_1[0] = d1 * (u1 - cx) / fx;
        p3d_1[1] = d1 * (v1 - cy) / fy;
        p3d_1[2] = d1;

        T p3d_0_projTo_1[3];
        ceres::QuaternionRotatePoint(q_0to1, p3d_0, p3d_0_projTo_1);
        p3d_0_projTo_1[0] += t_0to1[0];
        p3d_0_projTo_1[1] += t_0to1[1];
        p3d_0_projTo_1[2] += t_0to1[2];

        // project the 3D point to the image plane
        T u1_proj = p3d_0_projTo_1[0] * fx / p3d_0_projTo_1[2] + cx;
        T v1_proj = p3d_0_projTo_1[1] * fy / p3d_0_projTo_1[2] + cy;


        // 3d point align error
        residuals[0] = w_spatial* (p3d_0_projTo_1[0] - p3d_1[0]);
        residuals[1] = w_spatial* (p3d_0_projTo_1[1] - p3d_1[1]);
        residuals[2] = w_spatial* (p3d_0_projTo_1[2] - p3d_1[2]);
        // 2d reprojection error
        residuals[3] = w_reproj* (u1_proj - u1);
        residuals[4] = w_reproj* (v1_proj - v1);

        return true;
    }
};



class PnPReprojectionError
{
private:
    const double u0, v0, mono_d0, u1, v1, mono_d1;
    const double fx, fy, cx, cy;
    const double w_spatial, w_disparity, w_reproj;
    PnPReprojectionError(
        const double u0, const double v0, const double mono_d0,
        const double u1, const double v1, const double mono_d1,
        const double fx, const double fy, const double cx, const double cy, 
        const double w_spatial, const double w_disparity, const double w_reproj)
        : u0(u0), v0(v0), mono_d0(mono_d0), u1(u1), v1(v1), mono_d1(mono_d1), 
          fx(fx), fy(fy), cx(cx), cy(cy), w_spatial(w_spatial), w_disparity(w_disparity), w_reproj(w_reproj){}

public:
    static ceres::CostFunction *Create(
        const uvd_feature &uvd0, const uvd_feature &uvd1, const intrinsic_matrix K, const coeff_list &weights)
    {
        return (new ceres::AutoDiffCostFunction<PnPReprojectionError, 5, 4, 3, 1, 1, 1, 1>(
            new PnPReprojectionError(uvd0[0], uvd0[1], uvd0[2], uvd1[0], uvd1[1], uvd1[2],
                K(0, 0), K(1, 1), K(0, 2), K(1, 2), weights[0], weights[1], weights[2])
            )
        );
    }

    template <typename T>
    bool operator()(
        const T *const q_0to1, const T *const t_0to1, 
        const T *const a0, const T *const b0, const T *const a1, const T *const b1, T *residuals) const
    {
        T d0 = a0[0] * mono_d0 + b0[0];
        T d1 = a1[0] * mono_d1 + b1[0];

        T p3d_0[3];
        p3d_0[0] = d0 * (u0 - cx) / fx;
        p3d_0[1] = d0 * (v0 - cy) / fy;
        p3d_0[2] = d0;
        T p3d_1[3];
        p3d_1[0] = d1 * (u1 - cx) / fx;
        p3d_1[1] = d1 * (v1 - cy) / fy;
        p3d_1[2] = d1;

        T p3d_0_projTo_1[3];
        ceres::QuaternionRotatePoint(q_0to1, p3d_0, p3d_0_projTo_1);
        p3d_0_projTo_1[0] += t_0to1[0];
        p3d_0_projTo_1[1] += t_0to1[1];
        p3d_0_projTo_1[2] += t_0to1[2];
        // project the 3D point to the image plane
        T u1_proj = p3d_0_projTo_1[0] * fx / p3d_0_projTo_1[2] + cx;
        T v1_proj = p3d_0_projTo_1[1] * fy / p3d_0_projTo_1[2] + cy;

        // 3d point align error
        residuals[0] = w_spatial* (p3d_0_projTo_1[0] - p3d_1[0]) / p3d_1[2];
        residuals[1] = w_spatial* (p3d_0_projTo_1[1] - p3d_1[1]) / p3d_1[2];
        residuals[2] = w_spatial* (p3d_0_projTo_1[2] - p3d_1[2]) / p3d_1[2];
        // 2d reprojection error
        residuals[3] = w_reproj* (u1_proj - u1) / cx;
        residuals[4] = w_reproj* (v1_proj - v1) / cy;

        return true;
    }

};


/*
class GlobalPoseErrorCUDA
{
private:
    const double u0, v0, mono_d0;
    const double u1, v1, mono_d1;
    const double fx, fy, cx, cy;
    const double w_spatial, w_disparity, w_reproj;
    HOST_DEVICE GlobalPoseErrorCUDA(
        const double u0, const double v0, const double mono_d0,
        const double u1, const double v1, const double mono_d1,
        const double fx, const double fy, const double cx, const double cy,
        const double w_spatial, const double w_disparity, const double w_reproj)
        : u0(u0), v0(v0), mono_d0(mono_d0), u1(u1), v1(v1), mono_d1(mono_d1), 
          fx(fx), fy(fy), cx(cx), cy(cy), w_spatial(w_spatial), w_disparity(w_disparity), w_reproj(w_reproj){}

public:
    static ceres::CostFunction *Create(
        const uvd_feature &uvd0, const uvd_feature &uvd1, const intrinsic_matrix K, const coeff_list weights)
    {
        return (new ceres::AutoDiffCostFunction<GlobalPoseErrorCUDA, 5, 4, 3, 4, 3, 1, 1, 1, 1>(
            new GlobalPoseErrorCUDA(uvd0[0], uvd0[1], uvd0[2], uvd1[0], uvd1[1], uvd1[2],
                K(0, 0), K(1, 1), K(0, 2), K(1, 2), weights[0], weights[1], weights[2])
            )
        );
    }


    template <typename T>
    HOST_DEVICE bool operator()(
        const T *const q_0, const T *const t_0, const T *const q_1, const T *const t_1, 
        const T *const a0, const T *const b0, const T *const a1, const T *const b1, T *residuals) const
    {
        T q_0to1[4]; 
        T inv_q_0[4] = {q_0[0], -q_0[1], -q_0[2], -q_0[3]};
        ceres::QuaternionProduct(q_1, inv_q_0, q_0to1);
        T t_0in1[3];
        ceres::QuaternionRotatePoint(q_0to1, t_0, t_0in1);
        T t_0to1[3] = {t_1[0] - t_0in1[0], t_1[1] - t_0in1[1], t_1[2] - t_0in1[2]};


        T d0 = a0[0] * mono_d0 + b0[0];
        T d1 = a1[0] * mono_d1 + b1[0];

        T p3d_0[3];
        p3d_0[0] = d0 * (u0 - cx) / fx;
        p3d_0[1] = d0 * (v0 - cy) / fy;
        p3d_0[2] = d0;
        T p3d_1[3];
        p3d_1[0] = d1 * (u1 - cx) / fx;
        p3d_1[1] = d1 * (v1 - cy) / fy;
        p3d_1[2] = d1;

        T p3d_0_projTo_1[3];
        ceres::QuaternionRotatePoint(q_0to1, p3d_0, p3d_0_projTo_1);
        p3d_0_projTo_1[0] += t_0to1[0];
        p3d_0_projTo_1[1] += t_0to1[1];
        p3d_0_projTo_1[2] += t_0to1[2];

        // project the 3D point to the image plane
        T u1_proj = p3d_0_projTo_1[0] * fx / p3d_0_projTo_1[2] + cx;
        T v1_proj = p3d_0_projTo_1[1] * fy / p3d_0_projTo_1[2] + cy;

        // 3d point align error
        residuals[0] = w_spatial* (p3d_0_projTo_1[0] - p3d_1[0])/ p3d_1[2]; //  
        residuals[1] = w_spatial* (p3d_0_projTo_1[1] - p3d_1[1])/ p3d_1[2];
        residuals[2] = w_spatial* (p3d_0_projTo_1[2] - p3d_1[2])/ p3d_1[2];
        // re-projection error
        residuals[3] = w_reproj* (u1_proj - u1) / cx; //
        residuals[4] = w_reproj* (v1_proj - v1) / cy; //

        return true;
    }

};
*/


class GlobalPose3DError
{
private:
    const double u0, v0, mono_d0;
    const double u1, v1, mono_d1;
    const double fx, fy, cx, cy;
    const double w_spatial;
    GlobalPose3DError(
        const double u0, const double v0, const double mono_d0,
        const double u1, const double v1, const double mono_d1,
        const double fx, const double fy, const double cx, const double cy,
        const double w_spatial)
        : u0(u0), v0(v0), mono_d0(mono_d0), u1(u1), v1(v1), mono_d1(mono_d1), 
          fx(fx), fy(fy), cx(cx), cy(cy), w_spatial(w_spatial){}

public:
    static ceres::CostFunction *Create(
        const uvd_feature &uvd0, const uvd_feature &uvd1, const intrinsic_matrix K, const double weight)
    {
        return (new ceres::AutoDiffCostFunction<GlobalPose3DError, 3, 4, 3, 4, 3, 1, 1, 1, 1>(
            new GlobalPose3DError(uvd0[0], uvd0[1], uvd0[2], uvd1[0], uvd1[1], uvd1[2],
                K(0, 0), K(1, 1), K(0, 2), K(1, 2), weight)
            )
        );
    }


    template <typename T>
    bool operator()(
        const T *const q_0, const T *const t_0, const T *const q_1, const T *const t_1, 
        const T *const a0, const T *const b0, const T *const a1, const T *const b1, T *residuals) const
    {
        // Cam1_R_Cam0: q_0to1 = q_1 * q_0^-1
        T q_0to1[4]; 
        T inv_q_0[4] = {q_0[0], -q_0[1], -q_0[2], -q_0[3]};
        ceres::QuaternionProduct(q_1, inv_q_0, q_0to1);
        // t_0to1 = Cam1_R_obj * obj_t_Cam0 + Cam1_t_obj 
        //        = Cam1_R_obj * (-obj_R_Cam0 * Cam0_t_obj) + Cam1_t_obj
        T t_0in1[3];
        ceres::QuaternionRotatePoint(q_0to1, t_0, t_0in1);
        T t_0to1[3] = {t_1[0] - t_0in1[0], t_1[1] - t_0in1[1], t_1[2] - t_0in1[2]};

        T d0 = a0[0] * mono_d0 + b0[0];
        T d1 = a1[0] * mono_d1 + b1[0];
        T p3d_0[3];
        p3d_0[0] = d0 * (u0 - cx) / fx;
        p3d_0[1] = d0 * (v0 - cy) / fy;
        p3d_0[2] = d0;
        T p3d_1[3];
        p3d_1[0] = d1 * (u1 - cx) / fx;
        p3d_1[1] = d1 * (v1 - cy) / fy;
        p3d_1[2] = d1;

        T p3d_0_projTo_1[3];
        ceres::QuaternionRotatePoint(q_0to1, p3d_0, p3d_0_projTo_1);
        p3d_0_projTo_1[0] += t_0to1[0];
        p3d_0_projTo_1[1] += t_0to1[1];
        p3d_0_projTo_1[2] += t_0to1[2];

        // 3d point align error
        residuals[0] = w_spatial* (p3d_0_projTo_1[0] - p3d_1[0]); //  / p3d_1[2]
        residuals[1] = w_spatial* (p3d_0_projTo_1[1] - p3d_1[1]);
        residuals[2] = w_spatial* (p3d_0_projTo_1[2] - p3d_1[2]);

        // T p3d_obj0[3];
        // ceres::QuaternionRotatePoint(inv_q_0, p3d_0, p3d_obj0);
        // T p3d_obj1[3];
        // ceres::QuaternionRotatePoint(inv_q_1, p3d_1, p3d_obj1);
        // residuals[0] = w_spatial* (p3d_obj0[0] - p3d_obj1[0]);
        // residuals[1] = w_spatial* (p3d_obj0[1] - p3d_obj1[1]);
        // residuals[2] = w_spatial* (p3d_obj0[2] - p3d_obj1[2]);

        return true;
    }

};

class GlobalPoseReproj2DError
{
private:
    const double u0, v0, mono_d0;
    const double u1, v1, mono_d1;
    const double fx, fy, cx, cy;
    const double w_reproj;
    GlobalPoseReproj2DError(
        const double u0, const double v0, const double mono_d0,
        const double u1, const double v1, const double mono_d1,
        const double fx, const double fy, const double cx, const double cy,
        const double w_reproj)
        : u0(u0), v0(v0), mono_d0(mono_d0), u1(u1), v1(v1), mono_d1(mono_d1), 
          fx(fx), fy(fy), cx(cx), cy(cy), w_reproj(w_reproj){}

public:
    static ceres::CostFunction *Create(
        const uvd_feature &uvd0, const uvd_feature &uvd1, const intrinsic_matrix K, const double weight)
    {
        return (new ceres::AutoDiffCostFunction<GlobalPoseReproj2DError, 2, 4, 3, 4, 3, 1, 1, 1, 1>(
            new GlobalPoseReproj2DError(uvd0[0], uvd0[1], uvd0[2], uvd1[0], uvd1[1], uvd1[2],
                K(0, 0), K(1, 1), K(0, 2), K(1, 2), weight)
            )
        );
    }


    template <typename T>
    bool operator()(
        const T *const q_0, const T *const t_0, const T *const q_1, const T *const t_1, 
        const T *const a0, const T *const b0, const T *const a1, const T *const b1, T *residuals) const
    {
        // Cam1_R_Cam0: q_0to1 = q_1 * q_0^-1
        T q_0to1[4]; 
        T inv_q_0[4] = {q_0[0], -q_0[1], -q_0[2], -q_0[3]};
        ceres::QuaternionProduct(q_1, inv_q_0, q_0to1);
        T t_0in1[3];
        ceres::QuaternionRotatePoint(q_0to1, t_0, t_0in1);
        T t_0to1[3] = {t_1[0] - t_0in1[0], t_1[1] - t_0in1[1], t_1[2] - t_0in1[2]};

        T d0 = a0[0] * mono_d0 + b0[0];
        T d1 = a1[0] * mono_d1 + b1[0];
        T p3d_0[3];
        p3d_0[0] = d0 * (u0 - cx) / fx;
        p3d_0[1] = d0 * (v0 - cy) / fy;
        p3d_0[2] = d0;
        T p3d_0_projTo_1[3];
        ceres::QuaternionRotatePoint(q_0to1, p3d_0, p3d_0_projTo_1);
        p3d_0_projTo_1[0] += t_0to1[0];
        p3d_0_projTo_1[1] += t_0to1[1];
        p3d_0_projTo_1[2] += t_0to1[2];

        // project the 3D point to the image plane
        T u1_proj = p3d_0_projTo_1[0] * fx / p3d_0_projTo_1[2] + cx;
        T v1_proj = p3d_0_projTo_1[1] * fy / p3d_0_projTo_1[2] + cy;
        // re-projection error
        residuals[0] = w_reproj* (u1_proj - u1) / cx;
        residuals[1] = w_reproj* (v1_proj - v1) / cy;

        return true;
    }

};

class GlobalPoseDispError
{
private:
    const double u0, v0, mono_d0;
    const double u1, v1, mono_d1;
    const double fx, fy, cx, cy;
    const double w_disparity;
    GlobalPoseDispError(
        const double u0, const double v0, const double mono_d0,
        const double u1, const double v1, const double mono_d1,
        const double fx, const double fy, const double cx, const double cy,
        const double w_disparity)
        : u0(u0), v0(v0), mono_d0(mono_d0), u1(u1), v1(v1), mono_d1(mono_d1), 
          fx(fx), fy(fy), cx(cx), cy(cy), w_disparity(w_disparity){}

public:
    static ceres::CostFunction *Create(
        const uvd_feature &uvd0, const uvd_feature &uvd1, const intrinsic_matrix K, const double weight)
    {
        return (new ceres::AutoDiffCostFunction<GlobalPoseDispError, 1, 4, 3, 4, 3, 1, 1, 1, 1>(
            new GlobalPoseDispError(uvd0[0], uvd0[1], uvd0[2], uvd1[0], uvd1[1], uvd1[2],
                K(0, 0), K(1, 1), K(0, 2), K(1, 2), weight)
            )
        );
    }


    template <typename T>
    bool operator()(
        const T *const q_0, const T *const t_0, const T *const q_1, const T *const t_1, 
        const T *const a0, const T *const b0, const T *const a1, const T *const b1, T *residuals) const
    {
        // Cam1_R_Cam0: q_0to1 = q_1 * q_0^-1
        T q_0to1[4]; 
        T inv_q_0[4] = {q_0[0], -q_0[1], -q_0[2], -q_0[3]};
        ceres::QuaternionProduct(q_1, inv_q_0, q_0to1);
        T t_0in1[3];
        ceres::QuaternionRotatePoint(q_0to1, t_0, t_0in1);
        T t_0to1[3] = {t_1[0] - t_0in1[0], t_1[1] - t_0in1[1], t_1[2] - t_0in1[2]};

        T d0 = a0[0] * mono_d0 + b0[0];
        T d1 = a1[0] * mono_d1 + b1[0];
        T p3d_0[3];
        p3d_0[0] = d0 * (u0 - cx) / fx;
        p3d_0[1] = d0 * (v0 - cy) / fy;
        p3d_0[2] = d0;
        T p3d_1[3];
        p3d_1[0] = d1 * (u1 - cx) / fx;
        p3d_1[1] = d1 * (v1 - cy) / fy;
        p3d_1[2] = d1;

        T p3d_0_projTo_1[3];
        ceres::QuaternionRotatePoint(q_0to1, p3d_0, p3d_0_projTo_1);
        p3d_0_projTo_1[0] += t_0to1[0];
        p3d_0_projTo_1[1] += t_0to1[1];
        p3d_0_projTo_1[2] += t_0to1[2];

        // disparity error
        residuals[0] = w_disparity* ((1.0/ p3d_0_projTo_1[2]) - (1.0/ p3d_1[2]));

        return true;
    }

};



class PnPReprojectionErrorGlobalPoseTest
{
private:
    const double u0, v0, mono_d0;
    const double u1, v1, mono_d1;
    const double fx, fy, cx, cy;
    const double w_spatial, w_disparity, w_reproj;
    PnPReprojectionErrorGlobalPoseTest(
        const double u0, const double v0, const double mono_d0,
        const double u1, const double v1, const double mono_d1,
        const double fx, const double fy, const double cx, const double cy,
        const double w_spatial, const double w_disparity, const double w_reproj)
        : u0(u0), v0(v0), mono_d0(mono_d0), u1(u1), v1(v1), mono_d1(mono_d1), 
          fx(fx), fy(fy), cx(cx), cy(cy), 
          w_spatial(w_spatial), w_disparity(w_disparity), w_reproj(w_reproj){}

public:
    static ceres::CostFunction *Create(
        const uvd_feature &uvd0, const uvd_feature &uvd1, const intrinsic_matrix K, const coeff_list weights)
    {
        return (new ceres::AutoDiffCostFunction<PnPReprojectionErrorGlobalPoseTest, 6, 4, 3, 4, 3, 1, 1, 1, 1>(
            new PnPReprojectionErrorGlobalPoseTest(uvd0[0], uvd0[1], uvd0[2], uvd1[0], uvd1[1], uvd1[2],
                K(0, 0), K(1, 1), K(0, 2), K(1, 2), weights[0], weights[1], weights[2])
            )
        );
    }


    template <typename T>
    bool operator()(
        const T *const q_0, const T *const t_0, const T *const q_1, const T *const t_1, 
        const T *const a0, const T *const b0, const T *const a1, const T *const b1, T *residuals) const
    {
        T q_0to1[4]; // q_0to1 = q_1 * q_0^-1
        T inv_q_0[4] = {q_0[0], -q_0[1], -q_0[2], -q_0[3]};
        ceres::QuaternionProduct(q_1, inv_q_0, q_0to1);
        T t_0in1[3];
        ceres::QuaternionRotatePoint(q_0to1, t_0, t_0in1);
        T t_0to1[3] = {t_1[0] - t_0in1[0], t_1[1] - t_0in1[1], t_1[2] - t_0in1[2]};

        T d0 = a0[0] * mono_d0 + b0[0];
        T d1 = a1[0] * mono_d1 + b1[0];

        T p3d_0[3];
        p3d_0[0] = d0 * (u0 - cx) / fx;
        p3d_0[1] = d0 * (v0 - cy) / fy;
        p3d_0[2] = d0;

        T p3d_0_projTo_1[3];
        ceres::QuaternionRotatePoint(q_0to1, p3d_0, p3d_0_projTo_1);
        p3d_0_projTo_1[0] += t_0to1[0];
        p3d_0_projTo_1[1] += t_0to1[1];
        p3d_0_projTo_1[2] += t_0to1[2];

        T p3d_1[3];
        p3d_1[0] = d1 * (u1 - cx) / fx;
        p3d_1[1] = d1 * (v1 - cy) / fy;
        p3d_1[2] = d1;

        // project the 3D point to the image plane
        T u1_proj = p3d_0_projTo_1[0] * fx / p3d_0_projTo_1[2] + cx;
        T v1_proj = p3d_0_projTo_1[1] * fy / p3d_0_projTo_1[2] + cy;

        // 3d point align error
        residuals[0] = w_spatial* (p3d_0_projTo_1[0] - p3d_1[0]);
        residuals[1] = w_spatial* (p3d_0_projTo_1[1] - p3d_1[1]);
        residuals[2] = w_spatial* (p3d_0_projTo_1[2] - p3d_1[2]);
        // re-projection error
        residuals[3] = w_reproj* (u1_proj - u1);
        residuals[4] = w_reproj* (v1_proj - v1);
        // disparity error
        residuals[5] = w_disparity* ((1.0 / p3d_0_projTo_1[2]) - (1.0 / p3d_1[2]));

        return true;
    }

};




class PnPErrorGlobalPoseJacobian
{
private:
    const double u0, v0, mono_d0;
    const double u1, v1, mono_d1;
    const double fx, fy, cx, cy;
    const double w_spatial, w_disparity, w_reproj;

    PnPErrorGlobalPoseJacobian(
        const double u0, const double v0, const double mono_d0,
        const double u1, const double v1, const double mono_d1,
        const double fx, const double fy, const double cx, const double cy,
        const double w_spatial, const double w_disparity, const double w_reproj)
        : u0(u0), v0(v0), mono_d0(mono_d0), u1(u1), v1(v1), mono_d1(mono_d1),
          fx(fx), fy(fy), cx(cx), cy(cy), w_spatial(w_spatial), w_disparity(w_disparity), w_reproj(w_reproj){}

// public:
//     static ceres::CostFunction *Create(
//         const uvd_feature &uvd0, const uvd_feature &uvd1, const intrinsic_matrix K, const coeff_list weights)
//     {
//         return (new ceres::SizedCostFunction<5, 4, 3, 4, 3, 1, 1, 1, 1>(
//             new PnPErrorGlobalPoseJacobian(uvd0[0], uvd0[1], uvd0[2], uvd1[0], uvd1[1], uvd1[2],
//                 K(0, 0), K(1, 1), K(0, 2), K(1, 2), weights[0], weights[1], weights[2])
//             )
//         );
//     }

    /*
    // Define the cost function operator
    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        const double* q_0 = parameters[0];  // 4-element quaternion
        const double* t_0 = parameters[1];  // 3-element translation
        const double* q_1 = parameters[2];  // 4-element quaternion
        const double* t_1 = parameters[3];  // 3-element translation
        const double* a0 = parameters[4];   // scalar
        const double* b0 = parameters[5];   // scalar
        const double* a1 = parameters[6];   // scalar
        const double* b1 = parameters[7];   // scalar

        
        T q_0to1[4]; // q_0to1 = q_1 * q_0^-1
        T inv_q_0[4] = {q_0[0], -q_0[1], -q_0[2], -q_0[3]};
        ceres::QuaternionProduct(q_1, inv_q_0, q_0to1);
        T t_0in1[3];
        ceres::QuaternionRotatePoint(q_0to1, t_0, t_0in1);
        T t_0to1[3] = {t_1[0] - t_0in1[0], t_1[1] - t_0in1[1], t_1[2] - t_0in1[2]};

        T d0 = a0[0] * mono_d0 + b0[0];
        T d1 = a1[0] * mono_d1 + b1[0];

        T p3d_0[3];
        p3d_0[0] = d0 * (u0 - cx) / fx;
        p3d_0[1] = d0 * (v0 - cy) / fy;
        p3d_0[2] = d0;
        T p3d_1[3];
        p3d_1[0] = d1 * (u1 - cx) / fx;
        p3d_1[1] = d1 * (v1 - cy) / fy;
        p3d_1[2] = d1;

        T p3d_0_projTo_1[3];
        // trandform the point cloud to frame 1
        ceres::QuaternionRotatePoint(q_0to1, p3d_0, p3d_0_projTo_1);
        p3d_0_projTo_1[0] += t_0to1[0];
        p3d_0_projTo_1[1] += t_0to1[1];
        p3d_0_projTo_1[2] += t_0to1[2];
        // project the 3D point to the image plane
        T u1_proj = p3d_0_projTo_1[0] * fx / p3d_0_projTo_1[2] + cx;
        T v1_proj = p3d_0_projTo_1[1] * fy / p3d_0_projTo_1[2] + cy;

        // ===================== compute residuals =====================
        // 3d point align error
        residuals[0] = w_spatial* (p3d_0_projTo_1[0] - p3d_1[0]) / p3d_1[2];
        residuals[1] = w_spatial* (p3d_0_projTo_1[1] - p3d_1[1]) / p3d_1[2];
        residuals[2] = w_spatial* (p3d_0_projTo_1[2] - p3d_1[2]) / p3d_1[2];
        // re-projection error
        residuals[3] = w_reproj* (u1_proj - u1);
        residuals[4] = w_reproj* (v1_proj - v1);
        // disparity error
        // residuals[5] = w_disparity* ((1.0 / p3d_0_projTo_1[2]) - (1.0 / p3d_1[2])) * p3d_1[2];


        // ===================== compute Jacobians =====================
        if (jacobians) {
            // Jacobian w.r.t q_0 [5, 4]
            if (jacobians[0]) {
                Eigen::Matrix<double, 5, 4> dq0_dq0_inv = Eigen::Matrix<double, 5, 4>::Identity();
                dq0_dq0_inv(1, 1) = -1;
                dq0_dq0_inv(2, 2) = -1;
                dq0_dq0_inv(3, 3) = -1;
            }
            // Jacobian w.r.t t_0 [5, 3]
            if (jacobians[1]) {
                // jacobians[1][...] = ...
            }
            // Jacobian w.r.t q_1 [5, 4]
            if (jacobians[2]) {
                // jacobians[2][...] = ...
            }
            // Jacobian w.r.t t_1 [5, 3]
            if (jacobians[3]) {
                // jacobians[3][...] = ...
            }
            // Jacobian w.r.t a0 and b0
            if (jacobians[4]) {
                // jacobians[4][...] = ...
            }
            if (jacobians[5]) {
                // jacobians[5][...] = ...
            }
            // Jacobian w.r.t a1 and b1
            if (jacobians[6]) {
                // jacobians[6][...] = ...
            }
            if (jacobians[7]) {
                // jacobians[7][...] = ...
            }
        }

        return true;
    }
    */

};


/*
class MonoDTriangulateErrorGlobalPose
{
private:
    const double u0, v0, tri_d0;
    const double u1, v1, mono_d1;
    const double fx, fy, cx, cy;
    const double weight_tri_d;
    MonoDTriangulateErrorGlobalPose(
        const double u0, const double v0, const double tri_d0,
        const double u1, const double v1, const double mono_d1,
        const double fx, const double fy, const double cx, const double cy, const double weight_tri_d
        )
        : u0(u0), v0(v0), tri_d0(tri_d0), u1(u1), v1(v1), mono_d1(mono_d1), fx(fx), fy(fy), cx(cx), cy(cy), weight_tri_d(weight_tri_d){}

public:
    static ceres::CostFunction *Create(const uvd_feature &uvd0, const uvd_feature &uvd1, const intrinsic_matrix K,  const coeff_list weights_tri)
    {
        return (new ceres::AutoDiffCostFunction<MonoDTriangulateErrorGlobalPose, 5, 4, 3, 4, 3, 1, 1, 1, 1>(
            new MonoDTriangulateErrorGlobalPose(
                uvd0[0], uvd0[1], uvd0[2], uvd1[0], uvd1[1], uvd1[2], 
                K(0, 0), K(1, 1), K(0, 2), K(1, 2), weights_tri[0], weights_tri[1], weights_tri[2]
            )
        ));
    }

public:
    template <typename T>
    bool operator()(
        const T *const q_0, const T *const t_0, const T *const q_1, const T *const t_1, 
        const T *const a0, const T *const b0, const T *const a1, const T *const b1, T *residuals) const
    {
        T q_0to1[4]; // q_0to1 = q_1 * q_0^-1
        T inv_q_0[4] = {q_0[0], -q_0[1], -q_0[2], -q_0[3]};
        ceres::QuaternionProduct(q_1, inv_q_0, q_0to1);
        T t_0in1[3];
        ceres::QuaternionRotatePoint(q_0to1, t_0, t_0in1);
        T t_0to1[3] = {t_1[0] - t_0in1[0], t_1[1] - t_0in1[1], t_1[2] - t_0in1[2]};

        T d0(tri_d0);
        T d1 = a1[0] * mono_d1 + b1[0];

        T p3d_0[3];
        p3d_0[0] = d0 * (u0 - cx) / fx;
        p3d_0[1] = d0 * (v0 - cy) / fy;
        p3d_0[2] = d0;

        T p3d_0_projTo_1[3];
        ceres::QuaternionRotatePoint(q_0to1, p3d_0, p3d_0_projTo_1);
        p3d_0_projTo_1[0] += t_0to1[0];
        p3d_0_projTo_1[1] += t_0to1[1];
        p3d_0_projTo_1[2] += t_0to1[2];

        T p3d_1[3];
        p3d_1[0] = d1 * (u1 - cx) / fx;
        p3d_1[1] = d1 * (v1 - cy) / fy;
        p3d_1[2] = d1;

        // 3d point align error
        residuals[0] = weight_tri_d * (p3d_0_projTo_1[0] - p3d_1[0]) / p3d_1[2];
        residuals[1] = weight_tri_d * (p3d_0_projTo_1[1] - p3d_1[1]) / p3d_1[2];
        residuals[2] = weight_tri_d * (p3d_0_projTo_1[2] - p3d_1[2]) / p3d_1[2];

        // project the 3D point to the image plane
        T u1_proj =( p3d_0_projTo_1[0] * fx / p3d_0_projTo_1[2] )+ cx;
        T v1_proj =( p3d_0_projTo_1[1] * fy / p3d_0_projTo_1[2] )+ cy;
        residuals[3] = u1_proj - u1;
        residuals[4] = v1_proj - v1;

        return true;
    }

};
*/


/*
class MonoDTriangulateErrorScale
{
private:
    const double u0, v0, tri_d0;
    const double u1, v1, mono_d0;
    const double fx, fy, cx, cy;
    const double weight_tri_d;
    MonoDTriangulateErrorScale(
        const double u0, const double v0, const double tri_d0,
        const double u1, const double v1, const double mono_d0,
        const double fx, const double fy, const double cx, const double cy, const double weights_tri)
        : u0(u0), v0(v0), tri_d0(tri_d0), u1(u1), v1(v1), mono_d0(mono_d0), 
            fx(fx), fy(fy), cx(cx), cy(cy), weight_tri_d(weight_tri_d){}

public:
    static ceres::CostFunction *Create(
        const uvd_feature &uvd0, const uvd_feature &uvd1, 
        const intrinsic_matrix K, const coeff_list weights_tri)
    {
        return (new ceres::AutoDiffCostFunction<MonoDTriangulateErrorScale, 3, 1, 1>(
            new MonoDTriangulateErrorScale(
                uvd0[0], uvd0[1], uvd0[2], uvd1[0], uvd1[1], uvd1[2], 
                K(0, 0), K(1, 1), K(0, 2), K(1, 2), weights_tri[0], weights_tri[1], weights_tri[2]
            )
        ));
    }

public:
    template <typename T>
    bool operator()(
        const T *const a0, const T *const b0, T *residuals) const
    {
        T d0_tri(tri_d0);
        T d0_mono = a0[0] * mono_d0 + b0[0];

        T p3d_0_tri[3];
        p3d_0_tri[0] = d0_tri * (u0 - cx) / fx;
        p3d_0_tri[1] = d0_tri * (v0 - cy) / fy;
        p3d_0_tri[2] = d0_tri;

        T p3d_0_mono[3];
        p3d_0_mono[0] = d0_mono * (u1 - cx) / fx;
        p3d_0_mono[1] = d0_mono * (v1 - cy) / fy;
        p3d_0_mono[2] = d0_mono;

        // 3d point align error
        residuals[0] = weight_tri_d * (p3d_0_tri[0] - p3d_0_mono[0]) / p3d_0_mono[2];
        residuals[1] = weight_tri_d * (p3d_0_tri[1] - p3d_0_mono[1]) / p3d_0_mono[2];
        residuals[2] = weight_tri_d * (p3d_0_tri[2] - p3d_0_mono[2]) / p3d_0_mono[2];

        return true;
    }

};
*/


class ConstantError
{
private:
    const double constant_value;
    const double weight_constant;

    ConstantError(const double constant_value, const double weight_constant)
        : constant_value(constant_value), weight_constant(weight_constant) {}

public:
    static ceres::CostFunction *Create(const double constant_value, const double weight_constant)
    {
        return (new ceres::AutoDiffCostFunction<ConstantError, 1, 1>(
            new ConstantError(constant_value, weight_constant)));
    }

public:
    template <typename T>
    bool operator()(const T *const constant, T *residuals) const
    {
        residuals[0] = (constant[0] - constant_value)*weight_constant;
        return true;
    }
};





