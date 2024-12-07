#include "Swish.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(swish_binding, m) {
	m.doc() = R"doc(
		Python bindings for Swish activation function
	)doc";
	m.def("Swish", &Swish::Swish, "Swish activation function");
}