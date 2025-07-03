#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "OptQNewton.h"
#include "Opt.h"
#include "NLF.h"
#include "Teuchos_SerialDenseVector.hpp"
#include "Teuchos_SerialSymDenseMatrix.hpp"

#include <functional>

namespace py = pybind11;
using namespace OPTPP;

using T_SerialDenseVector = Teuchos::SerialDenseVector<int, double>;
using T_SerialSymDenseMatrix = Teuchos::SerialSymDenseMatrix<int, double>;

// Dummy C++ function for the default case
void default_update_model(int, int, T_SerialDenseVector) {}


// Trampoline class for NLF1
class PyNLF1 : public NLF1 {
public:
    // Inherit constructors
    using NLF1::NLF1;

    // Override virtual functions
    void initFcn() override {
        py::gil_scoped_acquire gil;
        py::function override = py::get_override(static_cast<const NLF1*>(this), "initFcn");
        if (override) {
            override();
        }
        // If no override is provided in Python, we do nothing.
        // The base NLF1::initFcn() is flawed for our use case as it calls a null function pointer.
    }

    // This eval() hides the non-virtual NLF1::eval() in the base class.
    // The optimizer calls this method. We then call the virtual methods
    // which are correctly routed to the Python overrides.
    void eval() {
        this->fvalue = this->evalF(this->mem_xc);
        this->mem_grad = this->evalG(this->mem_xc);
        this->nfevals++;
        this->ngevals++;
    }

    real evalF(const T_SerialDenseVector& x) override {
        py::gil_scoped_acquire gil;
        py::function override = py::get_override(static_cast<const NLF1*>(this), "evalF");
        if (override) {
            return override(x).cast<real>();
        }
        // If no override, return 0.0. The base class method is flawed.
        return 0.0;
    }

    T_SerialDenseVector evalG(const T_SerialDenseVector& x) override {
        py::gil_scoped_acquire gil;
        py::function override = py::get_override(static_cast<const NLF1*>(this), "evalG");
        if (override) {
            py::object result = override(x);
            py::array_t<double, py::array::c_style | py::array::forcecast> result_array = result.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
            py::buffer_info buf = result_array.request();
            if (buf.ndim != 1 || buf.shape[0] != this->getDim()) {
                throw std::runtime_error("evalG: Returned numpy array has wrong dimensions!");
            }
            // The optimizer depends on the internal state of mem_grad being updated.
            // We must copy the data from the Python override into mem_grad.
            if (mem_grad.length() != this->getDim()) {
                mem_grad.resize(this->getDim());
            }
            std::memcpy(mem_grad.values(), buf.ptr, this->getDim() * sizeof(double));
            return mem_grad;
        }
        // If no override, return an empty vector of the correct size.
        // The base class method is flawed.
        return T_SerialDenseVector(this->getDim());
    }

    T_SerialDenseVector evalG() override {
        // This override is crucial. The base NLF1::evalG() calls evalG(mem_xc),
        // but if the virtual method override has the wrong signature (e.g. const),
        // the base implementation of evalG(x) gets called, which crashes.
        // By overriding this, we ensure our safe evalG(x) is always called.
        return this->evalG(this->getXc());
    }

    T_SerialSymDenseMatrix evalH(T_SerialDenseVector& x) override {
        py::gil_scoped_acquire gil;
        py::function override = py::get_override(static_cast<const NLF1*>(this), "evalH");
        if (override) {
            py::print("Warning: evalH Python override is not fully implemented in the wrapper.");
        }
        // If no override, return an empty matrix of the correct size.
        // If no Python override, return the identity matrix
        int dim = this->getDim();
        T_SerialSymDenseMatrix H(dim);
        for (int i = 0; i < dim; ++i) {
            H(i, i) = 1.0;
        }
        return H;
    }
};


PYBIND11_MODULE(pyopttpp, m) {
    m.doc() = "Python bindings for OPTPP library";

    // Bind Teuchos::SerialDenseVector
    py::class_<T_SerialDenseVector>(m, "SerialDenseVector", py::buffer_protocol())
        .def(py::init<int>())
        .def(py::init([](py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            auto vec = new T_SerialDenseVector(arr.size());
            std::memcpy(vec->values(), arr.data(), arr.size() * sizeof(double));
            return vec;
        }))
        .def("size", &T_SerialDenseVector::length)
        .def("__len__", &T_SerialDenseVector::length)
        .def("__getitem__", [](const T_SerialDenseVector &v, int i) {
            if (i < 0) i += v.length();
            if (i < 0 || i >= v.length()) throw py::index_error();
            return v(i);
        })
        .def("__setitem__", [](T_SerialDenseVector &v, int i, double val) {
            if (i < 0) i += v.length();
            if (i < 0 || i >= v.length()) throw py::index_error();
            v(i) = val;
        })
        .def("to_numpy", [](T_SerialDenseVector &v) {
            return py::array_t<double>({(size_t)v.length()}, {sizeof(double)}, v.values(), py::cast(v));
        })
        .def_buffer([](T_SerialDenseVector &v) -> py::buffer_info {
            return py::buffer_info(
                v.values(),
                sizeof(double),
                py::format_descriptor<double>::format(),
                1,
                { (size_t)v.length() },
                { sizeof(double) }
            );
        });

    // Bind Teuchos::SerialSymDenseMatrix
    py::class_<Teuchos::SerialSymDenseMatrix<int, double>>(m, "SerialSymDenseMatrix")
        .def(py::init<int>());

    // Bind NLF1 using the trampoline class
    py::class_<NLF1, PyNLF1>(m, "NLF1")
        .def(py::init<int>(), py::arg("ndim"))
        .def("initFcn", &NLF1::initFcn)
        .def("evalF", static_cast<real (NLF1::*)(const T_SerialDenseVector&)>(&NLF1::evalF), py::arg("x"))
        .def("evalG", static_cast<T_SerialDenseVector (NLF1::*)(const T_SerialDenseVector&)>(&NLF1::evalG), py::arg("x"))
        .def("evalH", static_cast<T_SerialSymDenseMatrix (NLF1::*)(T_SerialDenseVector&)>(&NLF1::evalH), py::arg("x"))
        .def("getXc", &NLF1::getXc)
        .def("getF", &NLF1::getF)
        .def("setX", static_cast<void (NLF1::*)(const T_SerialDenseVector&)>(&NLF1::setX))
        .def("setIsExpensive", &NLF1::setIsExpensive, py::arg("is_expensive"));

    // Bind SearchStrategy enum
    py::enum_<SearchStrategy>(m, "SearchStrategy")
        .value("TrustRegion", SearchStrategy::TrustRegion)
        .value("LineSearch", SearchStrategy::LineSearch)
        .value("TrustPDS", SearchStrategy::TrustPDS)
        .export_values();

    // Bind OptQNewton
    py::class_<OptQNewton>(m, "OptQNewton")
        .def(py::init([](NLF1* p) {
            NLP1* p_base = static_cast<NLP1*>(p);
            return new OptQNewton(p_base, &default_update_model);
        }), py::arg("p"), py::keep_alive<0, 1>())
        .def("setDebug", &OPTPP::OptimizeClass::setDebug, "Set debug flag to true")
        .def("optimize", &OptQNewton::optimize)
        .def("printStatus", &OptQNewton::printStatus, py::arg("s"))
        .def("cleanup", &OptQNewton::cleanup)
        .def("setSearchStrategy", &OptQNewton::setSearchStrategy, py::arg("s"))
        .def("setOutputFile", static_cast<int (OptQNewton::*)(const char*, int)>(&OptQNewton::setOutputFile), py::arg("filename"), py::arg("mode") = 0)
        .def("setTRSize", &OptQNewton::setTRSize, py::arg("size"));
}
