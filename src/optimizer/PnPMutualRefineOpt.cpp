#include "cost_func.h"
#include <vector>
#include <cmath>
#include <utility>
#include <Eigen/Core>
#include <ceres/manifold.h>
// #include <ceres/problem_cuda.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#define SCALE_LOWBD 0.6
#define SCALE_UPBD 1.4
#define SHIFT_LOWBD -2.0
#define SHIFT_UPBD 2.0
// #define SCALE_LOWBD 0.95
// #define SCALE_UPBD 1.05
// #define SHIFT_LOWBD -0.2
// #define SHIFT_UPBD 0.2

class Point2D
{
public:
    int x, y;
    Point2D(int x, int y) : x(x), y(y) {}

    bool operator==(const Point2D &other) const
    {
        return x == other.x && y == other.y;
    }
};

// specialization of std::hash for Point2D
namespace std
{
    template <>
    struct hash<Point2D>
    {
        std::size_t operator()(const Point2D &p) const
        {
            using std::size_t;
            using std::hash;
            using std::string;

            return ((hash<double>()(p.x) ^ (hash<double>()(p.y) << 1)) >> 1);
        }
    };
} // namespace std


class BAsolver
{
private:
    ceres::Problem _problem;
    ceres::Solver::Options _options;
    inline void setUpperLowerBoundForGridScaleShift(
        double *a0_x1y1, double *b0_x1y1, double *a0_x2y1, double *b0_x2y1, 
        double *a0_x1y2, double *b0_x1y2, double *a0_x2y2, double *b0_x2y2,
        double *a1_x1y1, double *b1_x1y1, double *a1_x2y1, double *b1_x2y1, 
        double *a1_x1y2, double *b1_x1y2, double *a1_x2y2, double *b1_x2y2,
        double a_min, double a_max, double b_min, double b_max)
    {
        _problem.SetParameterLowerBound(a0_x1y1, 0, a_min);
        _problem.SetParameterUpperBound(a0_x1y1, 0, a_max);
        _problem.SetParameterLowerBound(b0_x1y1, 0, b_min);
        _problem.SetParameterUpperBound(b0_x1y1, 0, b_max);
        _problem.SetParameterLowerBound(a0_x2y1, 0, a_min);
        _problem.SetParameterUpperBound(a0_x2y1, 0, a_max);
        _problem.SetParameterLowerBound(b0_x2y1, 0, b_min);
        _problem.SetParameterUpperBound(b0_x2y1, 0, b_max);
        _problem.SetParameterLowerBound(a0_x1y2, 0, a_min);
        _problem.SetParameterUpperBound(a0_x1y2, 0, a_max);
        _problem.SetParameterLowerBound(b0_x1y2, 0, b_min);
        _problem.SetParameterUpperBound(b0_x1y2, 0, b_max);
        _problem.SetParameterLowerBound(a0_x2y2, 0, a_min);
        _problem.SetParameterUpperBound(a0_x2y2, 0, a_max);
        _problem.SetParameterLowerBound(b0_x2y2, 0, b_min);
        _problem.SetParameterUpperBound(b0_x2y2, 0, b_max);
        _problem.SetParameterLowerBound(a1_x1y1, 0, a_min);
        _problem.SetParameterUpperBound(a1_x1y1, 0, a_max);
        _problem.SetParameterLowerBound(b1_x1y1, 0, b_min);
        _problem.SetParameterUpperBound(b1_x1y1, 0, b_max);
        _problem.SetParameterLowerBound(a1_x2y1, 0, a_min);
        _problem.SetParameterUpperBound(a1_x2y1, 0, a_max);
        _problem.SetParameterLowerBound(b1_x2y1, 0, b_min);
        _problem.SetParameterUpperBound(b1_x2y1, 0, b_max);
        _problem.SetParameterLowerBound(a1_x1y2, 0, a_min);
        _problem.SetParameterUpperBound(a1_x1y2, 0, a_max);
        _problem.SetParameterLowerBound(b1_x1y2, 0, b_min);
        _problem.SetParameterUpperBound(b1_x1y2, 0, b_max);
        _problem.SetParameterLowerBound(a1_x2y2, 0, a_min);
        _problem.SetParameterUpperBound(a1_x2y2, 0, a_max);
        _problem.SetParameterLowerBound(b1_x2y2, 0, b_min);
        _problem.SetParameterUpperBound(b1_x2y2, 0, b_max);
    }

public:
    BAsolver()
    {
        _options.minimizer_type = ceres::TRUST_REGION;
        _options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        _options.function_tolerance = 1e-10;
        _options.parameter_tolerance = 1e-10;
        // _options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        // _options.linear_solver_type = ceres::DENSE_SCHUR;

        _options.max_num_iterations = 400;
        _options.num_threads = 11;
    }

    /**
     * @brief find the four neighbors of the query point
     * 
     * @param gridScaleShiftDict 
     * @param queryPoint 
     * @param gridSize sampling grid size
     * @return std::array<int, 4>
     *  x1, x2, y1, y2 is the four neighbors of the query point in
     *  downs-sampled coordinate system
     *      | x1 | x2 |
     * y1   |    |    |
     * y2   |    |    |
    */
    std::array<int, 4> bilinear_interpolation_find4neighbor(
        const Point2D &queryPoint, int gridSize
        )
    {
        const int halfGridSize = gridSize / 2;
        int x1, x2, y1, y2;

        int reminder_x = queryPoint.x % gridSize;
        if (reminder_x > halfGridSize) {
            x1 = queryPoint.x / gridSize;
            x2 = x1 + 1;
        }
        else {
            x2 = queryPoint.x / gridSize;
            x1 = x2 - 1;
        }
        int reminder_y = queryPoint.y % gridSize;
        if (reminder_y > halfGridSize) {
            y1 = queryPoint.y / gridSize;
            y2 = y1 + 1;
        }
        else {
            y2 = queryPoint.y / gridSize;
            y1 = y2 - 1;
        }
        assert(x2 > x1 && y2 > y1 && "x2 > x1 and y2 > y1");
        return {x1, x2, y1, y2};
    }


    void addAnEdgeLocal(
        bool fix_ab, bool is_first_frame, 
        const std::vector<uvd_feature> &uvd0s, const std::vector<uvd_feature> &uvd1s, 
        const intrinsic_matrix K, const coeff_list weights, 
        Eigen::Ref<Eigen::Vector4d> q_0to1_coeff, Eigen::Ref<Translation> t_0to1, 
        double &a0, double &b0, double &a1, double &b1
    ){
        assert(uvd0s.size() == uvd1s.size() && "uvd0 and uvd1 should have the same size");
        ceres::Manifold *quaternion_manifold = new ceres::QuaternionManifold;
        ceres::LossFunction *loss_function = new ceres::HuberLoss(9.0);
        // ceres::LossFunction *loss_function = nullptr;

        for (int i = 0; i < uvd0s.size(); i++)
        {
            ceres::CostFunction *cost_function = PnPReprojectionError::Create(
                uvd0s[i], uvd1s[i], K, weights);
            _problem.AddResidualBlock(cost_function, loss_function, 
            q_0to1_coeff.data(), t_0to1.data(), &a0, &b0, &a1, &b1);
        }

        _problem.SetManifold(q_0to1_coeff.data(), quaternion_manifold);

        if(fix_ab){
            // fix a0 and b0
            _problem.SetParameterBlockConstant(&a0);
            _problem.SetParameterBlockConstant(&b0);
            // fix a1 and b1
            _problem.SetParameterBlockConstant(&a1);
            _problem.SetParameterBlockConstant(&b1);
        }
        if(is_first_frame){
            // fix a0
            _problem.SetParameterBlockConstant(&a0);
            _problem.SetParameterBlockConstant(&b0);
        }

        _problem.SetParameterLowerBound(&a1, 0, SCALE_LOWBD);
        _problem.SetParameterUpperBound(&a1, 0, SCALE_UPBD);
        _problem.SetParameterLowerBound(&b1, 0, SHIFT_LOWBD);
        _problem.SetParameterUpperBound(&b1, 0, SHIFT_UPBD);

        _problem.SetParameterLowerBound(&a0, 0, SCALE_LOWBD);
        _problem.SetParameterUpperBound(&a0, 0, SCALE_UPBD);
        _problem.SetParameterLowerBound(&b0, 0, SHIFT_LOWBD);
        _problem.SetParameterUpperBound(&b0, 0, SHIFT_UPBD);
    }




    void addAnEdgeGlobal(
        bool fix_ab, bool is_first_frame, 
        const std::vector<uvd_feature> &uvd0s, const std::vector<uvd_feature> &uvd1s, 
        const intrinsic_matrix K, const coeff_list weights, const coeff_list loss_thres, 
        Eigen::Ref<Eigen::Vector4d> q_0_coeff, Eigen::Ref<Translation> t_0, 
        Eigen::Ref<Eigen::Vector4d> q_1_coeff, Eigen::Ref<Translation> t_1, 
        double &a0, double &b0, double &a1, double &b1
        )
    {
        assert(uvd0s.size() == uvd1s.size() && "uvd0 and uvd1 should have the same size");
        ceres::Manifold *quaternion_manifold = new ceres::QuaternionManifold; // without (), call the default constructor
        
        ceres::LossFunction *loss_func_3d = new ceres::HuberLoss(loss_thres[0]);
        ceres::LossFunction *loss_func_2d = new ceres::HuberLoss(loss_thres[1]);
        ceres::LossFunction *loss_func_disp = new ceres::HuberLoss(loss_thres[2]);
        ceres::LossFunction *loss_func_const = new ceres::TrivialLoss();
        // CauchyLoss(0.5); 

        for (int i = 0; i < uvd0s.size(); i++) 
        {   
            // 3d 
            ceres::CostFunction *cost_function_3d = GlobalPose3DError::Create(uvd0s[i], uvd1s[i], K, weights[0]);
            _problem.AddResidualBlock(cost_function_3d, loss_func_3d, 
                q_0_coeff.data(), t_0.data(),q_1_coeff.data(), t_1.data(), &a0, &b0, &a1, &b1);
            // 2d reprojection
            ceres::CostFunction *cost_function_2d = GlobalPoseReproj2DError::Create(uvd0s[i], uvd1s[i], K, weights[1]);
            _problem.AddResidualBlock(cost_function_2d, loss_func_2d, 
                q_0_coeff.data(), t_0.data(),q_1_coeff.data(), t_1.data(), &a0, &b0, &a1, &b1);
            // disparity
            ceres::CostFunction *cost_function_disp = GlobalPoseDispError::Create(uvd0s[i], uvd1s[i], K, weights[2]);
            _problem.AddResidualBlock(cost_function_disp, loss_func_disp, 
                q_0_coeff.data(), t_0.data(),q_1_coeff.data(), t_1.data(), &a0, &b0, &a1, &b1);
        }

        // if(is_first_frame){
        // }
        // else {
        //     for (int i = 0; i < uvd0s.size(); i++) 
        //     {   
        //         ceres::CostFunction *cost_function = PnPReprojectionErrorGlobalPoseTest::Create(uvd0s[i], uvd1s[i], K, weights);
        //         _problem.AddResidualBlock(cost_function, loss_function, 
        //             q_0_coeff.data(), t_0.data(),q_1_coeff.data(), t_1.data(), &a0, &b0, &a1, &b1);
        //     }
        // }

        _problem.SetManifold(q_0_coeff.data(), quaternion_manifold);
        _problem.SetManifold(q_1_coeff.data(), quaternion_manifold);

        // ***************** test *****************
        // _problem.SetParameterBlockConstant(&b0);
        // _problem.SetParameterBlockConstant(&b1);

        if(fix_ab){
            // fix a0 and b0
            _problem.SetParameterBlockConstant(&a0);
            _problem.SetParameterBlockConstant(&b0);
            // fix a1 and b1
            _problem.SetParameterBlockConstant(&a1);
            _problem.SetParameterBlockConstant(&b1);
        }

        if(is_first_frame){
            // fix q_0 and t_0
            _problem.SetParameterBlockConstant(q_0_coeff.data());
            _problem.SetParameterBlockConstant(t_0.data());
            // fix a0
            _problem.SetParameterBlockConstant(&a0);
            _problem.SetParameterBlockConstant(&b0);
        }
        // upper and lower bound
        _problem.SetParameterLowerBound(&a1, 0, SCALE_LOWBD);
        _problem.SetParameterUpperBound(&a1, 0, SCALE_UPBD);
        _problem.SetParameterLowerBound(&b1, 0, SHIFT_LOWBD);
        _problem.SetParameterUpperBound(&b1, 0, SHIFT_UPBD);

        _problem.SetParameterLowerBound(&a0, 0, SCALE_LOWBD);
        _problem.SetParameterUpperBound(&a0, 0, SCALE_UPBD);
        _problem.SetParameterLowerBound(&b0, 0, SHIFT_LOWBD);
        _problem.SetParameterUpperBound(&b0, 0, SHIFT_UPBD);
    }




    void addAnEdgeWithTriangulate(
        bool fix_ab, 
        const std::vector<uvd_feature> &uv_monoD0s, const std::vector<uvd_feature> &uv_monoD1s, 
        const std::vector<uvd_feature> &uv_triD0s_tri, const std::vector<uvd_feature> &uv_monoD0s_tri, 
        const intrinsic_matrix K, const coeff_list weights_mono, const coeff_list weights_tri, 
        Eigen::Ref<Eigen::Vector4d> q_0_coeff, Eigen::Ref<Translation> t_0, 
        Eigen::Ref<Eigen::Vector4d> q_1_coeff, Eigen::Ref<Translation> t_1, 
        double &a0, double &b0, double &a1, double &b1
        )
    {
        assert(uv_monoD0s.size() == uv_monoD1s.size() && "uvd0 and uvd1 from MonoD should have the same size");
        assert(uv_triD0s_tri.size() == uv_monoD0s_tri.size() && "uvd0 and uvd1 from tracking should have the same size");

        ceres::Manifold *quaternion_manifold = new ceres::QuaternionManifold;
        ceres::LossFunction *loss_function_1 = new ceres::HuberLoss(9.0);
        ceres::LossFunction *loss_function_2 = new ceres::HuberLoss(9.0);
        // ceres::LossFunction *loss_function = nullptr;

        // for (int i = 0; i < uv_monoD0s.size(); i++)
        // {
        //     ceres::CostFunction *cost_function = PnPReprojectionErrorGlobalPose::Create(
        //         uv_monoD0s[i], uv_monoD1s[i], K, weights_mono);
        //     _problem.AddResidualBlock(cost_function, loss_function_1, 
        //         q_0_coeff.data(), t_0.data(),q_1_coeff.data(), t_1.data(), &a0, &b0, &a1, &b1);
        // }
        // for (int i = 0; i < uv_triD0s_tri.size(); i++)
        // {
        //     ceres::CostFunction *cost_function = MonoDTriangulateErrorScale::Create(
        //         uv_triD0s_tri[i], uv_monoD0s_tri[i], K, weights_tri);
        //     _problem.AddResidualBlock(cost_function, loss_function_2, &a0, &b0);
        // }

        _problem.SetManifold(q_0_coeff.data(), quaternion_manifold);
        _problem.SetManifold(q_1_coeff.data(), quaternion_manifold);
        if(fix_ab) {
            // fix a1 and b1
            _problem.SetParameterBlockConstant(&a1);
            _problem.SetParameterBlockConstant(&b1);
            // fix a0 and b0
            _problem.SetParameterBlockConstant(&a0);
            _problem.SetParameterBlockConstant(&b0);
        }
        _problem.SetParameterLowerBound(&a1, 0, SCALE_LOWBD);
        _problem.SetParameterUpperBound(&a1, 0, SCALE_UPBD);
        _problem.SetParameterLowerBound(&b1, 0, SHIFT_LOWBD);
        _problem.SetParameterUpperBound(&b1, 0, SHIFT_UPBD);

        _problem.SetParameterLowerBound(&a0, 0, SCALE_LOWBD);
        _problem.SetParameterUpperBound(&a0, 0, SCALE_UPBD);
        _problem.SetParameterLowerBound(&b0, 0, SHIFT_LOWBD);
        _problem.SetParameterUpperBound(&b0, 0, SHIFT_UPBD);
    }


    void addAnEdgeGridVersion(
        bool fix_ab,bool is_first_frame, 
        const std::vector<uvd_feature> &uvd0s, const std::vector<uvd_feature> &uvd1s, 
        const intrinsic_matrix K, const coeff_list weights, 
        Eigen::Ref<Eigen::Vector4d> q_0_coeff, Eigen::Ref<Translation> t_0, 
        Eigen::Ref<Eigen::Vector4d> q_1_coeff, Eigen::Ref<Translation> t_1, 
        py::dict &gridScaleShiftDict0, py::dict &gridScaleShiftDict1, int gridSize
        )
    {
        // make sure uvd0.size() == uvd1.size()
        assert(uvd0s.size() == uvd1s.size() && "uvd0 and uvd1 should have the same size");
        ceres::Manifold *quaternion_manifold = new ceres::QuaternionManifold; // without (), call the default constructor
        ceres::LossFunction *loss_function = new ceres::HuberLoss(9.0);
        // ceres::LossFunction *loss_function = nullptr;

        for (int i = 0; i < uvd0s.size(); i++)
        {
            Point2D p0(uvd0s[i][0], uvd0s[i][1]);
            Point2D p1(uvd1s[i][0], uvd1s[i][1]);
            // find the four neighbors of p0 and p1
            std::array<int, 4> neigh_f0 = bilinear_interpolation_find4neighbor(p0, gridSize);
            std::array<int, 4> neigh_f1 = bilinear_interpolation_find4neighbor(p1, gridSize);
            // get the scale and shift ptr of the four neighbors
            py::array_t<double> s_factor;
            /// frame 0
            auto key0_x1y1 = py::make_tuple(neigh_f0[0], neigh_f0[2]);
            auto key0_x2y1 = py::make_tuple(neigh_f0[1], neigh_f0[2]);
            auto key0_x1y2 = py::make_tuple(neigh_f0[0], neigh_f0[3]);
            auto key0_x2y2 = py::make_tuple(neigh_f0[1], neigh_f0[3]);
            s_factor = gridScaleShiftDict0[key0_x1y1].cast<py::array_t<double>>();
            double *a0_x1y1 = s_factor.mutable_data(0);
            double *b0_x1y1 = s_factor.mutable_data(1);
            s_factor = gridScaleShiftDict0[key0_x2y1].cast<py::array_t<double>>();
            double *a0_x2y1 = s_factor.mutable_data(0);
            double *b0_x2y1 = s_factor.mutable_data(1);
            s_factor = gridScaleShiftDict0[key0_x1y2].cast<py::array_t<double>>();
            double *a0_x1y2 = s_factor.mutable_data(0);
            double *b0_x1y2 = s_factor.mutable_data(1);
            s_factor = gridScaleShiftDict0[key0_x2y2].cast<py::array_t<double>>();
            double *a0_x2y2 = s_factor.mutable_data(0);
            double *b0_x2y2 = s_factor.mutable_data(1);
            /// frame 1
            auto key1_x1y1 = py::make_tuple(neigh_f1[0], neigh_f1[2]);
            auto key1_x2y1 = py::make_tuple(neigh_f1[1], neigh_f1[2]);
            auto key1_x1y2 = py::make_tuple(neigh_f1[0], neigh_f1[3]);
            auto key1_x2y2 = py::make_tuple(neigh_f1[1], neigh_f1[3]);
            s_factor = gridScaleShiftDict1[key1_x1y1].cast<py::array_t<double>>();
            double *a1_x1y1 = s_factor.mutable_data(0);
            double *b1_x1y1 = s_factor.mutable_data(1);
            s_factor = gridScaleShiftDict1[key1_x2y1].cast<py::array_t<double>>();
            double *a1_x2y1 = s_factor.mutable_data(0);
            double *b1_x2y1 = s_factor.mutable_data(1);
            s_factor = gridScaleShiftDict1[key1_x1y2].cast<py::array_t<double>>();
            double *a1_x1y2 = s_factor.mutable_data(0);
            double *b1_x1y2 = s_factor.mutable_data(1);
            s_factor = gridScaleShiftDict1[key1_x2y2].cast<py::array_t<double>>();
            double *a1_x2y2 = s_factor.mutable_data(0);
            double *b1_x2y2 = s_factor.mutable_data(1);

            // @TODO: add cost function
            ceres::CostFunction *cost_function = PnPReprojectionErrorGlobalPoseGridVersion::Create(
                uvd0s[i], uvd1s[i], K, weights, gridSize, neigh_f0, neigh_f1
                );

            _problem.AddResidualBlock(cost_function, loss_function,
                q_0_coeff.data(), t_0.data(), q_1_coeff.data(), t_1.data(),
                a0_x1y1, b0_x1y1, a0_x2y1, b0_x2y1, a0_x1y2, b0_x1y2, a0_x2y2, b0_x2y2,
                a1_x1y1, b1_x1y1, a1_x2y1, b1_x2y1, a1_x1y2, b1_x1y2, a1_x2y2, b1_x2y2);
            
            // set lower and upper bound for a0 and a1, b0 and b1
            this->setUpperLowerBoundForGridScaleShift(
                    a0_x1y1, b0_x1y1, a0_x2y1, b0_x2y1,
                    a0_x1y2, b0_x1y2, a0_x2y2, b0_x2y2,
                    a1_x1y1, b1_x1y1, a1_x2y1, b1_x2y1,
                    a1_x1y2, b1_x1y2, a1_x2y2, b1_x2y2,
                    0.8, 1.2, -1.0, 1.0);

            if(fix_ab) {
                // fix a0 and b0
                _problem.SetParameterBlockConstant(a0_x1y1);
                _problem.SetParameterBlockConstant(b0_x1y1);
                _problem.SetParameterBlockConstant(a0_x2y1);
                _problem.SetParameterBlockConstant(b0_x2y1);
                _problem.SetParameterBlockConstant(a0_x1y2);
                _problem.SetParameterBlockConstant(b0_x1y2);
                _problem.SetParameterBlockConstant(a0_x2y2);
                _problem.SetParameterBlockConstant(b0_x2y2);
                // fix a1 and b1
                _problem.SetParameterBlockConstant(a1_x1y1);
                _problem.SetParameterBlockConstant(b1_x1y1);
                _problem.SetParameterBlockConstant(a1_x2y1);
                _problem.SetParameterBlockConstant(b1_x2y1);
                _problem.SetParameterBlockConstant(a1_x1y2);
                _problem.SetParameterBlockConstant(b1_x1y2);
                _problem.SetParameterBlockConstant(a1_x2y2);
                _problem.SetParameterBlockConstant(b1_x2y2);
            }

            if (is_first_frame) {
                // fix a0 and b0
                _problem.SetParameterBlockConstant(a0_x1y1);
                _problem.SetParameterBlockConstant(b0_x1y1);
                _problem.SetParameterBlockConstant(a0_x2y1);
                _problem.SetParameterBlockConstant(b0_x2y1);
                _problem.SetParameterBlockConstant(a0_x1y2);
                _problem.SetParameterBlockConstant(b0_x1y2);
                _problem.SetParameterBlockConstant(a0_x2y2);
                _problem.SetParameterBlockConstant(b0_x2y2);
            }
        }

        _problem.SetManifold(q_0_coeff.data(), quaternion_manifold);
        _problem.SetManifold(q_1_coeff.data(), quaternion_manifold);
        if(is_first_frame){
            // fix q_0 and t_0
            _problem.SetParameterBlockConstant(q_0_coeff.data());
            _problem.SetParameterBlockConstant(t_0.data());
        }

    }

    void solve() {
        ceres::Solver::Summary summary;
        ceres::Solve(_options, &_problem, &summary);
        std::cout << summary.BriefReport() << std::endl;
    }
    
    void solve(int max_num_iterations) {
        ceres::Solver::Summary summary;
        _options.max_num_iterations = max_num_iterations;
        ceres::Solve(_options, &_problem, &summary);
        std::cout << summary.BriefReport() << std::endl;
    }
  
    void reset() {
        _problem = ceres::Problem();
    }
 
};


// class BAsolverCUDA
// {
// private:
//     ceres::ProblemCUDA _problemCUDA;
//     ceres::Solver::Options _options;

// public:
//     BAsolverCUDA()
//     {
//         _options.minimizer_type = ceres::TRUST_REGION;
//         _options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
//         _options.function_tolerance = 1e-10;
//         _options.parameter_tolerance = 1e-10;
//         // _options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
//         // _options.linear_solver_type = ceres::DENSE_SCHUR;
//         _options.max_num_iterations = 400;
//         _options.num_threads = 11;
//     }

//     void addAnEdgeGlobalCUDA(
//         bool fix_ab, bool is_first_frame, 
//         const std::vector<uvd_feature> &uvd0s, const std::vector<uvd_feature> &uvd1s, 
//         const intrinsic_matrix K, const coeff_list weights, 
//         Eigen::Ref<Eigen::Vector4d> q_0_coeff, Eigen::Ref<Translation> t_0, 
//         Eigen::Ref<Eigen::Vector4d> q_1_coeff, Eigen::Ref<Translation> t_1, 
//         double &a0, double &b0, double &a1, double &b1
//         )
//     {
//         ceres::Manifold *quat_manifold = new ceres::QuaternionManifold;
//         ceres::LossFunction *loss_function = new ceres::HuberLossCUDA(4.0);
//         //ceres::LossFunction *loss_function = new ceres::CauchyLossCUDA(9.0);

//         for (int i = 0; i < uvd0s.size(); i++) 
//         {   
//             ceres::CostFunction *cost_function = GlobalPoseErrorCUDA::Create(
//                 uvd0s[i], uvd1s[i], K, weights);
//             _problem.AddResidualBlock(cost_function, loss_function, 
//                 q_0_coeff.data(), t_0.data(),q_1_coeff.data(), t_1.data(), &a0, &b0, &a1, &b1);
//         }

//         _problem.SetManifold(q_0_coeff.data(), quat_manifold);
//         _problem.SetManifold(q_1_coeff.data(), quat_manifold);
//         if(fix_ab){
//             // fix a0 and b0
//             _problem.SetParameterBlockConstant(&a0);
//             _problem.SetParameterBlockConstant(&b0);
//             // fix a1 and b1
//             _problem.SetParameterBlockConstant(&a1);
//             _problem.SetParameterBlockConstant(&b1);
//         }

//         if(is_first_frame){
//             // fix q_0 and t_0
//             _problem.SetParameterBlockConstant(q_0_coeff.data());
//             _problem.SetParameterBlockConstant(t_0.data());
//             // fix a0
//             _problem.SetParameterBlockConstant(&a0);
//             _problem.SetParameterBlockConstant(&b0);
//         }
//         // upper and lower bound
//         _problem.SetParameterLowerBound(&a1, 0, SCALE_LOWBD);
//         _problem.SetParameterUpperBound(&a1, 0, SCALE_UPBD);
//         _problem.SetParameterLowerBound(&b1, 0, SHIFT_LOWBD);
//         _problem.SetParameterUpperBound(&b1, 0, SHIFT_UPBD);

//         _problem.SetParameterLowerBound(&a0, 0, SCALE_LOWBD);
//         _problem.SetParameterUpperBound(&a0, 0, SCALE_UPBD);
//         _problem.SetParameterLowerBound(&b0, 0, SHIFT_LOWBD);
//         _problem.SetParameterUpperBound(&b0, 0, SHIFT_UPBD);
//     }


//     void solve() {
//         ceres::Solver::Summary summary;
//         ceres::Solve(_options, &_problemCUDA, &summary);
//         std::cout << summary.FullReport() << std::endl;
//     }

//     void reset() {
//         _problemCUDA = ceres::ProblemCUDA();
//     }

// };



std::vector<Eigen::Matrix<double, 3, 1>> convert_to_vector(py::array_t<double> input_array)
{
    py::buffer_info buf_info = input_array.request();
    if (buf_info.ndim != 2 || buf_info.shape[1] != 3)
    {
        throw std::runtime_error("Input should be a two-dimensional array with 3 columns");
    }
    double *ptr = static_cast<double *>(buf_info.ptr);
    std::vector<Eigen::Matrix<double, 3, 1>> result;
    for (ssize_t i = 0; i < buf_info.shape[0]; ++i) {
        Eigen::Matrix<double, 3, 1> vec;
        for (ssize_t j = 0; j < 3; ++j) {
            vec[j] = ptr[i * buf_info.shape[1] + j];
        }
        result.push_back(vec);
    }
    return result;
}

    
PYBIND11_MODULE(pyPnPMutualRefine, m)
{
    m.doc() = "PnP Refine Optimization";
    // solver class
    py::class_<BAsolver>(m, "BAsolver")
        .def(py::init<>())
        .def("addAnEdgeLocal", [](
            BAsolver& self, bool is_fix_ab, bool is_1st_frame, 
            py::array_t<double> &uvd0s_arr, py::array_t<double> &uvd1s_arr, 
            Eigen::Matrix<double,3,3> K, Eigen::Vector3d weights, 
            Eigen::Ref<Eigen::Vector4d> q_0to1_coeff, Eigen::Ref<Translation> t_0to1, 
            Eigen::Ref<Eigen::Vector2d> scaleAndShift0, Eigen::Ref<Eigen::Vector2d> scaleAndShift1
            ){
                auto uvd0s = convert_to_vector(uvd0s_arr);
                auto uvd1s = convert_to_vector(uvd1s_arr);
                double &a0 = scaleAndShift0(0);
                double &b0 = scaleAndShift0(1);
                double &a1 = scaleAndShift1(0);
                double &b1 = scaleAndShift1(1);
                self.addAnEdgeLocal(is_fix_ab,is_1st_frame, uvd0s, uvd1s, 
                    K, weights, q_0to1_coeff, t_0to1, a0, b0, a1, b1);
            }
        )
        .def("addAnEdgeGlobal", [](
            BAsolver& self, bool is_fix_ab, bool is_1st_frame, 
            py::array_t<double> &uvd0s_arr, py::array_t<double> &uvd1s_arr, 
            Eigen::Matrix<double,3,3> K, Eigen::Vector3d weights, Eigen::Vector3d loss_thres,
            Eigen::Ref<Eigen::Vector4d> q_0_coeff, Eigen::Ref<Translation> t_0, 
            Eigen::Ref<Eigen::Vector4d> q_1_coeff, Eigen::Ref<Translation> t_1, 
            Eigen::Ref<Eigen::Vector2d> scaleAndShift0, Eigen::Ref<Eigen::Vector2d> scaleAndShift1
            ){
                // convert to vector of 3x1 matrix(s)
                auto uvd0s = convert_to_vector(uvd0s_arr);
                auto uvd1s = convert_to_vector(uvd1s_arr);
                double &a0 = scaleAndShift0(0);
                double &b0 = scaleAndShift0(1);
                double &a1 = scaleAndShift1(0);
                double &b1 = scaleAndShift1(1);
                self.addAnEdgeGlobal(is_fix_ab, is_1st_frame, uvd0s, uvd1s, K, 
                    weights, loss_thres, q_0_coeff, t_0, q_1_coeff, t_1, a0, b0, a1, b1);
            }
        )
        .def("addAnEdgeWithTriangulate", [](
            BAsolver& self, bool is_fix_ab, 
            py::array_t<double> &uvd0s_arr, py::array_t<double> &uvd1s_arr, 
            py::array_t<double> &uvd0s_tri_arr, py::array_t<double> &uvd1s_tri_arr, 
            Eigen::Matrix<double,3,3> K, Eigen::Vector3d weights_mono, Eigen::Vector3d weights_tri,
            Eigen::Ref<Eigen::Vector4d> q_0_coeff, Eigen::Ref<Translation> t_0, 
            Eigen::Ref<Eigen::Vector4d> q_1_coeff, Eigen::Ref<Translation> t_1, 
            Eigen::Ref<Eigen::Vector2d> scaleAndShift0, Eigen::Ref<Eigen::Vector2d> scaleAndShift1
            ){
                auto uvd0s = convert_to_vector(uvd0s_arr);
                auto uvd1s = convert_to_vector(uvd1s_arr);
                auto uvd0s_tri = convert_to_vector(uvd0s_tri_arr);
                auto uvd1s_tri = convert_to_vector(uvd1s_tri_arr);
                double &a0 = scaleAndShift0(0);
                double &b0 = scaleAndShift0(1);
                double &a1 = scaleAndShift1(0);
                double &b1 = scaleAndShift1(1);
                self.addAnEdgeWithTriangulate(is_fix_ab, uvd0s, uvd1s, uvd0s_tri, uvd1s_tri, 
                    K, weights_mono, weights_tri, q_0_coeff, t_0, q_1_coeff, t_1, a0, b0, a1, b1);
            }
        )
        .def("addAnEdgeGridVersion", [](
            BAsolver& self, bool is_fix_ab, bool is_1st_frame, 
            py::array_t<double> &uvd0s_arr, py::array_t<double> &uvd1s_arr, 
            Eigen::Matrix<double,3,3> K, Eigen::Vector3d weights, 
            Eigen::Ref<Eigen::Vector4d> q_0_coeff, Eigen::Ref<Translation> t_0, 
            Eigen::Ref<Eigen::Vector4d> q_1_coeff, Eigen::Ref<Translation> t_1, 
            py::dict &gridScaleShiftDict0, py::dict &gridScaleShiftDict1, int gridSize
            ){
                auto uvd0s = convert_to_vector(uvd0s_arr);
                auto uvd1s = convert_to_vector(uvd1s_arr);
                // x1 x2 y1 y2
                self.addAnEdgeGridVersion(
                    is_fix_ab, is_1st_frame, uvd0s, uvd1s, 
                    K, weights, q_0_coeff, t_0, q_1_coeff, t_1, 
                    gridScaleShiftDict0, gridScaleShiftDict1, gridSize
                    );
            }
        )
        .def("testModifyDict", [](BAsolver& self, py::dict &gridScaleShiftDict){
            for(auto item: gridScaleShiftDict)
            {
                auto key = item.first.cast<py::tuple>();
                auto value = item.second.cast<py::array_t<double>>();
                std::cout << "key: " << key[0].cast<int>() << " " << key[1].cast<int>() << " value: " << value.at(0) << " " << value.at(1) << std::endl;
                double *v0_ptr = value.mutable_data(0);
                double *v1_ptr = value.mutable_data(1);
                *v0_ptr = 100;
                *v1_ptr = 200;
                std::cout << "after modify: " << value.at(0) << " " << value.at(1) << std::endl;
            }
            }
        )
        .def("reset", &BAsolver::reset)
        .def("solve", (void (BAsolver::*)())&BAsolver::solve)
        .def("solve", (void (BAsolver::*)(int))&BAsolver::solve, py::arg("max_num_iterations"));
    // solver class for CUDA
    // py::class_<BAsolverCUDA>(m, "BAsolverCUDA")
    //     .def(py::init<>())
    //     .def("addAnEdgeGlobalCUDA", [](
    //         BAsolver& self, bool is_fix_ab, bool is_1st_frame, 
    //         py::array_t<double> &uvd0s_arr, py::array_t<double> &uvd1s_arr, 
    //         Eigen::Matrix<double,3,3> K, Eigen::Vector3d weights, 
    //         Eigen::Ref<Eigen::Vector4d> q_0to1_coeff, Eigen::Ref<Translation> t_0to1, 
    //         Eigen::Ref<Eigen::Vector2d> scaleAndShift0, Eigen::Ref<Eigen::Vector2d> scaleAndShift1
    //         ){
    //             auto uvd0s = convert_to_vector(uvd0s_arr);
    //             auto uvd1s = convert_to_vector(uvd1s_arr);
    //             double &a0 = scaleAndShift0(0);
    //             double &b0 = scaleAndShift0(1);
    //             double &a1 = scaleAndShift1(0);
    //             double &b1 = scaleAndShift1(1);
    //             self.addAnEdgeGlobalCUDA(is_fix_ab,is_1st_frame, uvd0s, uvd1s, 
    //                 K, weights, q_0to1_coeff, t_0to1, a0, b0, a1, b1);
    //         }
    //     )
    //     .def("reset", &BAsolver::reset)
    //     .def("solve", (void (BAsolver::*)())&BAsolver::solve);

}
