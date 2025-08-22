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
        .def("reset", &BAsolver::reset)
        .def("solve", (void (BAsolver::*)())&BAsolver::solve)
        .def("solve", (void (BAsolver::*)(int))&BAsolver::solve, py::arg("max_num_iterations"));

}
