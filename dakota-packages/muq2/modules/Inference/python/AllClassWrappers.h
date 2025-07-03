#ifndef INFERENCE_ALLCLASSWRAPPERS_H_
#define INFERENCE_ALLCLASSWRAPPERS_H_

#include <pybind11/pybind11.h>

namespace muq{
  namespace Inference{
    namespace PythonBindings{

      void KalmanWrapper(pybind11::module &m);
    }
  }
}


#endif
