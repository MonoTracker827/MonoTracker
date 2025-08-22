#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <glog/logging.h>

namespace py = pybind11; 

#include "utils_bindings.h"

PYBIND11_MODULE(pyglobalsfm, m) {
    m.doc() = "Python bindings for the reconstruction utility functions.";
    m.def("align_orientations", &align_orientations, "Align input orientations to the reference ones",
          py::arg("orientations"),
          py::arg("reference_orientations"),
          py::arg("loss_threshold") = 0.1,
          py::arg("maximum_iterations") = 1000,
          py::arg("num_threads") = 4,
          py::arg("initialization") = true,
          py::arg("numerical_optimization") = true,
          py::arg("log") = true);         
}
