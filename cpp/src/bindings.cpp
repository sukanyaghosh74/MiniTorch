#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../include/tensor.h"

namespace py = pybind11;
using namespace minitorch;

PYBIND11_MODULE(_C, m) {
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<int64_t>&, const std::string&, const std::string&>())
        .def("shape", &Tensor::shape)
        .def("dtype", &Tensor::dtype)
        .def("device", &Tensor::device)
        .def("add", &Tensor::add)
        .def("mul", &Tensor::mul)
        .def("matmul", &Tensor::matmul)
        .def("sum", &Tensor::sum)
        .def("data", [](Tensor& t) {
            auto shape = t.shape();
            return py::array_t<float>(shape, (float*)t.data());
        });
}
