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





