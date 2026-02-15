#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "core/tensor.h"
#include "core/autograd.h"
#include "layers/linear.h"
#include "layers/conv2d.h"
#include "ops/activation.h"
#include "optim/sgd.h"
#include "loss/cross_entropy.h"
#include "layers/maxpool2d.h"
#include "ops/flatten.h"

namespace py = pybind11;

// ---------------------
// Global Seed Function
// ---------------------
void set_seed(int seed) {
    std::srand(seed);
}

namespace py = pybind11;

PYBIND11_MODULE(my_framework, m) {

    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<const std::vector<float>&,
                      const std::vector<int>&,
                      bool>(),
             py::arg("data"),
             py::arg("shape"),
             py::arg("requires_grad") = false)
        .def_readwrite("data", &Tensor::data)
        .def_readwrite("grad", &Tensor::grad)
        .def_readwrite("shape", &Tensor::shape)
        .def_readwrite("requires_grad", &Tensor::requires_grad)
        .def("numel", &Tensor::numel)
        .def("zero_grad", &Tensor::zero_grad)
        .def("backward", &Tensor::backward);
        
    py::class_<Linear>(m, "Linear")
        .def(py::init<int, int>())
        .def("forward", &Linear::forward)
        .def("parameters", &Linear::parameters);
        
    py::class_<Conv2D>(m, "Conv2D")
        .def(py::init<int, int, int, int, int>(),
            py::arg("in_channels"),
            py::arg("out_channels"),
            py::arg("kernel_size"),
            py::arg("stride") = 1,
            py::arg("padding") = 0)
        .def("forward", &Conv2D::forward)
        .def("parameters", &Conv2D::parameters);

    py::class_<MaxPool2D>(m, "MaxPool2D")
        .def(py::init<int, int>())
        .def("forward", &MaxPool2D::forward);

    py::class_<SGD>(m, "SGD")
        .def(py::init<const std::vector<std::shared_ptr<Tensor>>&,
                    float>())
        .def("step", &SGD::step)
        .def("zero_grad", &SGD::zero_grad);

    m.def("add", &add);
    m.def("mul", &mul);
    m.def("sum", &sum);
    m.def("matmul", &matmul);
    m.def("relu", &relu);
    m.def("flatten", &flatten);
    m.def("cross_entropy", &cross_entropy);
    m.def("set_seed", &set_seed);
}