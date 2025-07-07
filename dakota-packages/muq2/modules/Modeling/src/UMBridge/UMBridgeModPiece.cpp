#include "MUQ/Modeling/UMBridge/UMBridgeModPiece.h"

using namespace muq::Modeling;


UMBridgeModPiece::UMBridgeModPiece(const std::string host, std::string name, json config, httplib::Headers headers)
: config(config), client(host, name, headers),
  ModPiece(read_input_size(host, name, headers), read_output_size(host, name, headers))
{
  this->outputs.resize(this->numOutputs);
}


Eigen::VectorXi UMBridgeModPiece::read_input_size(const std::string host, std::string name, const httplib::Headers& headers){
  // Would prefer to reuse the existing client, circular dependency in constructor though...
  umbridge::HTTPModel client(host, name, headers);
  return StdVectorToEigenvectori(client.GetInputSizes(config));
}

Eigen::VectorXi UMBridgeModPiece::read_output_size(const std::string host, std::string name, const httplib::Headers& headers){
  umbridge::HTTPModel client(host, name, headers);
  return StdVectorToEigenvectori(client.GetOutputSizes(config));
}

void UMBridgeModPiece::EvaluateImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs) {
  if (!client.SupportsEvaluate())
    throw std::runtime_error("Model does not support evaluation!");
  std::vector<std::vector<double>> inputs_stdvec(this->numInputs);
  for (int i = 0; i < this->numInputs; i++) {
    inputs_stdvec[i] = EigenvectordToStdVector(inputs[i]);
  }
  std::vector<std::vector<double>> out = client.Evaluate(inputs_stdvec, config);
  for (int i = 0; i < this->numOutputs; i++) {
    outputs[i] = StdVectorToEigenvectord(out[i]);
  }
}

void UMBridgeModPiece::GradientImpl(unsigned int outWrt,
                                unsigned int inWrt,
                                muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs,
                                Eigen::VectorXd const& sens) {
  if (client.SupportsGradient()) {
    gradient = StdVectorToEigenvectord(client.Gradient(outWrt, inWrt, EigenvectordsToStdVectors(inputs), EigenvectordToStdVector(sens), config));
  } else
    gradient = GradientByFD(outWrt, inWrt, inputs, sens);
}

void UMBridgeModPiece::ApplyJacobianImpl(unsigned int outWrt,
                                    unsigned int inWrt,
                                    muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs,
                                    Eigen::VectorXd const& vec){
  if (client.SupportsApplyJacobian()) {
    jacobianAction = StdVectorToEigenvectord(client.ApplyJacobian(outWrt, inWrt, EigenvectordsToStdVectors(inputs), EigenvectordToStdVector(vec), config));
  } else
    jacobianAction = ApplyJacobianByFD(outWrt, inWrt, inputs, vec);
}

void UMBridgeModPiece::ApplyHessianImpl(unsigned int outWrt,
                                    unsigned int inWrt1,
                                    unsigned int inWrt2,
                                    muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs,
                                    Eigen::VectorXd const& sens,
                                    Eigen::VectorXd const& vec){
  if (client.SupportsApplyHessian()) {
    hessAction = StdVectorToEigenvectord(client.ApplyHessian(outWrt, inWrt1, inWrt2, EigenvectordsToStdVectors(inputs), EigenvectordToStdVector(sens), EigenvectordToStdVector(vec), config));
  } else
    hessAction = ApplyHessianByFD(outWrt, inWrt1, inWrt2, inputs, sens, vec);
}