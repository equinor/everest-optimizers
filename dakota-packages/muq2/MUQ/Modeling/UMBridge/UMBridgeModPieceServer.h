#include "MUQ/Modeling/UMBridge/UMBridgeModPiece.h"

namespace muq {
  namespace Modeling {

    /**
      @class UMBridgeModPieceWrapper
      @brief Wrap a ModPiece in an UM-Bridge Model
      @details This is needed in order to easily serve a MUQ ModPiece via UM-Bridge.
      */
    class UMBridgeModPieceWrapper : public umbridge::Model {
    public:

      /**
        @brief Constructor
        @param modPiece The ModPiece to wrap.
        @param name The name of the model as exposed by UM-Bridge.
        */
      UMBridgeModPieceWrapper(std::shared_ptr<muq::Modeling::ModPiece> modPiece, std::string name)
      : umbridge::Model(name),
        modPiece(modPiece)
      {}

      std::vector<std::size_t> GetInputSizes(const json& config_json = json()) const override {
        return UMBridgeModPiece::EigenvectoriToStdVector(modPiece->inputSizes);
      }

      std::vector<std::size_t> GetOutputSizes(const json& config_json = json()) const override {
        return UMBridgeModPiece::EigenvectoriToStdVector(modPiece->outputSizes);
      }

      std::vector<std::vector<double>> Evaluate(const std::vector<std::vector<double>>& inputs, json config) override {
        return UMBridgeModPiece::EigenvectordsToStdVectors(
                    modPiece->Evaluate(UMBridgeModPiece::StdVectorsToEigenvectords(inputs)));
      }

      std::vector<double> Gradient(unsigned int outWrt,
                    unsigned int inWrt,
                    const std::vector<std::vector<double>>& inputs,
                    const std::vector<double>& sens,
                    json config = json()) override {
        return UMBridgeModPiece::EigenvectordToStdVector(
                      modPiece->Gradient(outWrt,
                                         inWrt,
                                         UMBridgeModPiece::StdVectorsToEigenvectords(inputs),
                                         UMBridgeModPiece::StdVectorToEigenvectord(sens)));
      }

      std::vector<double> ApplyJacobian(unsigned int outWrt,
                         unsigned int inWrt,
                         const std::vector<std::vector<double>>& inputs,
                         const std::vector<double>& vec,
                         json config = json()) override {
        return UMBridgeModPiece::EigenvectordToStdVector(
                            modPiece->ApplyJacobian(outWrt,
                                                    inWrt,
                                                    UMBridgeModPiece::StdVectorsToEigenvectords(inputs),
                                                    UMBridgeModPiece::StdVectorToEigenvectord(vec)));
      }

      std::vector<double> ApplyHessian(unsigned int outWrt,
                        unsigned int inWrt1,
                        unsigned int inWrt2,
                        const std::vector<std::vector<double>>& inputs,
                        const std::vector<double>& sens,
                        const std::vector<double>& vec,
                        json config = json()) override {
        return UMBridgeModPiece::EigenvectordToStdVector(
                        modPiece->ApplyHessian(outWrt,
                                               inWrt1,
                                               inWrt2,
                                               UMBridgeModPiece::StdVectorsToEigenvectords(inputs),
                                               UMBridgeModPiece::StdVectorToEigenvectord(sens),
                                               UMBridgeModPiece::StdVectorToEigenvectord(vec)));
      }

      bool SupportsEvaluate() override {return true;}
      bool SupportsGradient() override {return true;} // Expose derivative information as well,
      bool SupportsApplyJacobian() override {return true;} // since we fall back to finite differences automatically
      bool SupportsApplyHessian() override {return true;}

    private:
      std::shared_ptr<muq::Modeling::ModPiece> modPiece;
    };

    /**
     * @brief Serve a ModPiece via network using UM-Bridge
     *
     * @param modPiece The modPiece to serve via UM-Bridge
     * @param host Bind address, may be 0.0.0.0
     * @param port Port at which to serve the modPiece
     */
    void serveModPiece(std::shared_ptr<ModPiece> modPiece, std::string name, std::string host, int port) {
      UMBridgeModPieceWrapper wrapper(modPiece, name);
      umbridge::serveModels({&wrapper}, host, port);
    }

  }
}
