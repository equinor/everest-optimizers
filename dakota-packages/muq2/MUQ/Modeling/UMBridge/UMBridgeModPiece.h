#ifndef UMBRIDGEMODPIECE
#define UMBRIDGEMODPIECE
#include "MUQ/Modeling/ModPiece.h"
#include "umbridge.h"

namespace muq {
  namespace Modeling {

    /**
     * @brief A ModPiece connecting to a model via the UM-Bridge HTTP protocol
     * @details This ModPiece connects to a (remote or local) model via UM-Bridge, a protocol based on HTTP.
     * A more in-depth documentation on the underlying HTTP interface and the server-side part can
     * be found at https://github.com/UM-Bridge/umbridge. Several models and benchmarks
     * are also available in the form of ready-to-use containers.
     *
     * The main use case is straightforward coupling of model and UQ codes across
     * different languages and frameworks. Since this allows treating the model
     * mostly as a black box, greater separation of concerns between model and UQ
     * developers can be achieved.
     *
     * In order to set up an UMBridgeModPiece, you need to specify the address it is to connect to.
     * For an HTTP model running locally, it typically looks like this:
     *
     * @code{.cpp}
     * auto umbridge_modpiece = std::make_shared<UMBridgeModPiece>("http://localhost:4242");
     * @endcode
     *
     * Passing additional configuration options to the model is supported through JSON structures.
     * For more examples on how to set up a JSON structure, refer to the documentation of json.hpp.
     *
     * @code{.cpp}
     * json config;
     * config["level"] = 1;
     * auto umbridge_modpiece = std::make_shared<UMBridgeModPiece>("http://localhost:4242", config);
     * @endcode
     *
     * For testing purposes, you can use a test benchmark hosted by us. It implements a
     * very simple Bayesian posterior:
     *
     * @code{.cpp}
     * auto benchmark_modpiece = std::make_shared<UMBridgeModPiece>("http://testbenchmark.linusseelinger.de");
     * @endcode
     *
     * Beyond initialization, UMBridgeModPiece behaves like any other ModPiece.
     *
     * The implementation makes use of the HTTP model c++ header-only library,
     * which in turn depends on json.hpp for JSON and httplib.h for HTTP support.
     */
    class UMBridgeModPiece : public muq::Modeling::ModPiece {
    public:

      /**
       * @brief Construct UMBridgeModPiece, connecting to model server
       *
       * @param host The host adress (and possibly port) to connect to. May be local.
       * @param name The name of the model to connect to.
       * @param config Configuration parameters may be passed to the model.
       * @param headers Additional headers may be passed to the server, e.g. access tokens.
       */
      UMBridgeModPiece(const std::string host, std::string name, json config = json(), httplib::Headers headers = httplib::Headers());

      static const std::vector<double> EigenvectordToStdVector(const Eigen::VectorXd& vector) {
        const std::vector<double> vec(vector.data(), vector.data() + vector.rows());
        return vec;
      }
      static Eigen::VectorXd StdVectorToEigenvectord(std::vector<double>& vector) {
        double* ptr_data = &vector[0];
        Eigen::VectorXd vec = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(ptr_data, vector.size());
        return vec;
      }
      static const Eigen::VectorXd StdVectorToEigenvectord(const std::vector<double>& vector) {
        const double* ptr_data = &vector[0];
        const Eigen::VectorXd vec = Eigen::Map<const Eigen::VectorXd, Eigen::Unaligned>(ptr_data, vector.size());
        return vec;
      }

      static std::vector<std::vector<double>> EigenvectordsToStdVectors(std::vector<Eigen::VectorXd> const& inputs) {
        std::vector<std::vector<double>> vecs(inputs.size());
        for (int i = 0; i < inputs.size(); i++)
          vecs[i] = EigenvectordToStdVector(inputs[i]);
        return vecs;
      }
      static std::vector<std::vector<double>> EigenvectordsToStdVectors(muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs) {
        std::vector<std::vector<double>> vecs(inputs.size());
        for (int i = 0; i < inputs.size(); i++)
          vecs[i] = EigenvectordToStdVector(inputs[i]);
        return vecs;
      }
      static std::vector<Eigen::VectorXd> StdVectorsToEigenvectords(std::vector<std::vector<double>> vector) {
        std::vector<Eigen::VectorXd> vec(vector.size());
        for (int i = 0; i < vector.size(); i++) {
          vec[i] = StdVectorToEigenvectord(vector[i]);
        }
        return vec;
      }

      static std::vector<std::size_t> EigenvectoriToStdVector(const Eigen::VectorXi& vector) {
        std::vector<std::size_t> vec(vector.data(), vector.data() + vector.rows());
        return vec;
      }
      static Eigen::VectorXi StdVectorToEigenvectori(const std::vector<std::size_t>& vector) {
        Eigen::VectorXi vec(vector.size());
        for (int i = 0; i < vector.size(); i++)
          vec[i] = vector[i];
        return vec;
      }

    private:

      Eigen::VectorXi read_input_size(const std::string host, std::string name, const httplib::Headers& headers);

      Eigen::VectorXi read_output_size(const std::string host, std::string name, const httplib::Headers& headers);

      void EvaluateImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs) override;

      void GradientImpl(unsigned int outWrt,
                        unsigned int inWrt,
                        muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs,
                        Eigen::VectorXd const& sens) override;

      void ApplyJacobianImpl(unsigned int outWrt,
                             unsigned int inWrt,
                             muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs,
                             Eigen::VectorXd const& vec) override;

      void ApplyHessianImpl(unsigned int outWrt,
                            unsigned int inWrt1,
                            unsigned int inWrt2,
                            muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs,
                            Eigen::VectorXd const& sens,
                            Eigen::VectorXd const& vec) override;

      json config;
      umbridge::HTTPModel client;
    };

  }
}

#endif
