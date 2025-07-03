#ifndef MUQ_MODELING_H
#define MUQ_MODELING_H

#include "MUQ/Modeling/ModPiece.h"

/**
@defgroup modeling Modeling

## Background and Motivation
UQ algorithms use repeated model evaluations to propagate uncertainties or to sample probability distributions.   The same implementation of a UQ algorithm needs to be used on many different models.  The Modeling module in MUQ provides a collection of classes that enable models to be defined in a way that MUQ can understand.   Many algorithms also require derivative information (such as gradients, Jacobian matrices, and Hessian matrices), which can also be defined and evaluated using this module.

To define models and support derivative evaluations, it is common for software frameworks to leverage domain specific languages or to use specialized numeric types in throughout the model definition.  These intrusive modeling approaches enable the calculation of derivative information using automatic differentiation and are therefore amenable to efficient structure-exploiting UQ algorithms that leverage gradient information.  Black-box approaches are an alternative that do not need to know anything about the inner workings of a model.  Only evaluations of the input-to-output mapping are necessary. This approach, in contrast to intrusive appraoches, provides incredible flexibility in how the models are defined and facilitates easy coupling with other software packages, system calls, humans-in-the-loop, etc...   However, pure black-box approaches do not expose gradient information and do not allow shared  modeling components to be easily reused.

MUQ adopts a hybrid approach that blends the computational graph concept used by AD packages with black-box modeling approaches.   MUQ defines a model through the connection of many small modeling components on a computational graph, but treats each component as a black box.  MUQ only needs to evaluate the input-output relationship and (optionally) evaluate derivatives of the component.  A model component could therefore make a system call to evaluate a complicated CFD simulation in a commercial CFD package like Fluent and evaluate gradients through sophisticated adjoint techniques.
*/

/**
@defgroup modpieces Model Components and the ModPiece class
@ingroup modeling

## Model Components in MUQ
Each model component in MUQ is defined as a child of the [ModPiece](\ref muq::Modeling::ModPiece) abstract base class.  The ModPiece class is the software analog of a function
\f\[
f : \mathbb{R}^{N_1}\times\cdots\times\mathbb{R}^{N_{D_{in}}}\rightarrow\mathbb{R}^{M_1}\times\cdots\times \mathbb{R}^{M_{D_{out}}},
\f\]
with \f$D_{in}\f$ vector-valued inputs and \f$D_{out}\f$ vector-valued outputs.  The ModPiece base class provides functions for evaluating the function \f$f\f$ and computing derivatives with respect to the input of \f$f\f$.   The code snippet below uses the [ExpOperator](\ref muq::Modeling::ExpOperator) child provided by MUQ to evaluate \f$f(x) = \exp(x)\f$.


@codeblock{cpp, C++}
#include <Eigen/Core>
#include <vector>
#include "MUQ/Modeling/CwiseOperators/CwiseUnaryOperator.h"

using namespace muq::Modeling;

int main(){
  // Set the dimension N_1 of the input vector
  int N1 = 2;

  // Create a ModPiece for evaluating f(x)=exp(x)
  auto expMod = std::make_shared<ExpOperator>(N1);

  // Define an input of all ones
  Eigen::VectorXd x = Eigen::VectorXd::Ones(N1);

  // Evaluate the model
  std::vector<Eigen::VectorXd> fx = expMod->Evaluate(x);
}
@endcodeblock
@codeblock{python,Python}
import numpy as np
import muq.Modeling as mm

# Set the dimension N_1 of the input vector
N1 = 2

# Create a ModPiece for evaluating f(x)=exp(x)
expMod = mm.ExpOperator(N1)

// Define an input of all ones
x = np.ones(N1)

// Evaluate the model
fx = expMod.Evaluate([x])
@endcodeblock

There are a few important things to note in this code snippet:
- MUQ uses the vector and matrix classes from [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page) in c++ but uses numpy objects in Python.
- The output of the <code>Evaluate</code> is a list (Python) or std::vector (c++) of vectors.  This is because the ModPiece class defines general functions with multiple inputs and outputs.  The square brackets <code>[...]</code> surrounding the input <code>x</code> in the Python code exists for the same reason.  The <code>x</code> variable is a vector, but the <code>Evaluate</code> function accepts one or more vectors in a list.   This isn't an issue in the c++ code because MUQ has shortcuts for transforming the <code>Eigen::VectorXd</code> into the expected list format.  It is also possible to skip these tricks by explicitly creating a std::vector input and passing that to the <code>Evaluate</code> function:
@codeblock{cpp,C++}
std::vector<Eigen::VectorXd> x(1);
x.at(0) = Eigen::VectorXd::Ones(N1);
std::vector<Eigen::VectorXd> fx = expMod->Evaluate(x);
@endcodeblock

### Using components with multiple inputs
Children of the <code>ModPiece</code> class (colloquially called "ModPieces") with multiple inputs are evaluated in the same way as the <code>ExpOperator</code> class above.   Here is an example using the [SumPiece](\ref muq::Modeling::SumPiece) class.

@codeblock{cpp,C++}
#include <vector>
#include <Eigen/Core>
#include "MUQ/Modeling/SumPiece.h"

int main(){

  // Set the dimension both input vectors
  int N = 2;

  // Create a ModPiece for evaluating f(x)=x_1 + x_2
  auto sumMod = std::make_shared<SumPiece>(N,2);

  // Define the inputs
  Eigen::VectorXd x1 = Eigen::VectorXd::Ones(N);
  Eigen::VectorXd x2 = Eigen::VectorXd::Ones(N);

  // Evaluate the model
  std::vector<Eigen::VectorXd> fx = sumMod->Evaluate(x1,x2);

  return 0;
}
@endcodeblock
@codeblock{python,Python}
import numpy as np
import muq.Modeling as mm

# Set the dimension both input vectors
N = 2

# Create a ModPiece for evaluating f(x)=x_1 + x_2
sumMod = mm.SumPiece>(N,2)

# Define the inputs
x1 = np.ones(N)
x2 = np.ones(N)

# Evaluate the model
fx = sumMod.Evaluate([x1,x2])

@endcodeblock
@codeblock{cpp,C++ (std::vector)}
#include <vector>
#include <Eigen/Core>
#include "MUQ/Modeling/SumPiece.h"

int main(){

  // Set the dimension both input vectors
  int N = 2;
  int numInputs = 4;

  // Create a ModPiece for evaluating f(x)=x_1 + x_2
  auto sumMod = std::make_shared<SumPiece>(N,2);

  // Define the inputs
  std::vector<Eigen::VectorXd> inputs(numInputs);
  for(int i=0; i<numInputs; ++i)
    inputs.at(i) = Eigen::VectorXd::Ones(N);

  // Evaluate the model
  std::vector<Eigen::VectorXd> fx = sumMod->Evaluate(inputs);

  return 0;
}
@endcodeblock

### Evaluating Derivatives
In addition to evaluating the model \f$y_1,y_2,\ldots,y_{D_{out}} = f(x_1,x_2,\ldots,x_{D_{in}})\f$, the ModPieces can also compute derivative information.   For example, to compute the Jacobian matrix of \f$y_i\f$ with respect to \f$x_j\f$, we can use the following code:
@codeblock{cpp,C++}
unsigned int outWrt = 0; // i in dy_i /dx_j
unsigned int inWrt = 0; // j in dy_i /dx_j
Eigen::MatrixXd jac = expMod->Jacobian(outWrt,inWrt,x);
@endcodeblock
@codeblock{python,Python}
outWrt = 0
inWrt = 0
jac = expMod.Jacobian(outWrt,inWrt,x)
@endcodeblock

Similarly, we can compute the application of the Jacobian to a vector, which corresponds to the directional derivative of \f$y_i\f$.  For some models, such as PDES with simple tangent operators, it can be much more efficient to compute Jacobian-vector products than constructing the entire Jacobian matrix and performing a matrix-vector multiplication.
@codeblock{cpp,C++}
Eigen::VectorXd dir = Eigen::VectorXd::Random(inDim);
Eigen::Matrix deriv = expMod->ApplyJacobian(outWrt,inWrt,x,dir);
@endcodeblock
@codeblock{python,Python}
dir = np.random.rand(inDim)
deriv = expMod.ApplyJacobian(outWrt,inWrt,x,dir)
@endcodeblock

Gradients can also be computed in MUQ.  Consider a situation where we have a scalar function \f$g(y_i)\f$ and are interested in the gradient of \f$g\f$ with respect to the model input \f$x_j\f$.  From the chain rule, we have \f$\nabla_{x_j} g = \nabla_{y_i} g D_{ij}f\f$, where \f$\nabla_{y_i} g\f$ is a row-vector containing the gradient of \f$g\f$ and \f$D_{ij}\f$ denotes the Jacobian matrix of the model \f$f\f$ using output \f$y_i\f$ and input \f$x_j\f$.     In terms of column vectors, we have
\f\[
( \nabla_{x_i} g)^T = D_{ij}^T s,
\f\]
where \f$s = \nabla_{y_i} g\f$.  In our context, the vector \f$s\f$ is called the sensitivity vector as it represents the first order sensitivity of \f$g\f$ with respect to the model output \f$y_i\f$.   Notice that the directional derivatives above, the gradient is simple a matrix-vector product involving the Jacobian matrix.  Here however, the transpose of the Jacobian matrix is used.

The <code>Gradient</code> function in the <code>ModPiece</code> class computes \f$D_{ij}^T s\f$.  This operation can be much faster than constructing the full Jacobian matrix when the dimension of the model input \f$x_j\f$ is large, such as when \f$x_j\f$ represents a spatially distributed field and \f$f(x)\f$ is a PDE model.  In this case, adjoint techniques can be used to efficiently compute \f$D_{ij}^T s\f$.
@codeblock{cpp,C++}
Eigen::VectorXd sens = Eigen::VectorXd::Random(outDim); // i.e., s= \nabla g^T
Eigen::VectorXd grad = expMod->Gradient(outWrt,inWrt,x,sens);
@endcodeblock
@codeblock{python,Python}
sens = np.random.rand(outDim)
grad = expMod.Gradient(outWrt,inWrt,x,sens)
@endcodeblock

Hessian-vector products, which are second order directional derivatives, can be with the <code>ApplyHessian</code> function in the <code>ModPiece</code> class.  Notice that the Hessian matrix is the Jacobian matrix of the function mapping \f$x_j\f$ to \f$\nabla_{x_j} g\f$. gradient function.  The <code>ModPiece::ApplyHessian</code> thus applies the Jacobian of the Gradient function to a vector.  To define the gradient, we need the same sensitivity vector \f$s\f$ used in the gradient calculation, as well as the vector we want to apply the Hessian to.
@codeblock{cpp,C++}
unsigned int inWrt1 = 0; // i in the second derivative d^2f / dx_i dx_j
unsigned int inWrt2 = 0; // j in the second derivative d^2f / dx_i dx_j

Eigen::VectorXd dir = Eigen::VectorXd::Random(inDim);
Eigen::VectorXd sens = Eigen::VectorXd::Random(outDim);

Eigen::VectorXd deriv2 = expMod->ApplyHessian(outWrt,inWrt1,inWrt2,x,sens,dir);
@endcodeblock
@codeblock{python,Python}
inWrt1 = 0 # i in the second derivative d^2f / dx_i dx_j
inWrt2 = 0 # j in the second derivative d^2f / dx_i dx_j

direction = np.random.rand(inDim)
sens = np.random.ran(outDim)

deriv2 = expMod.ApplyHessian(outWrt,inWrt1,inWrt2,x,sens,direction)
@endcodeblock
*/

/**
@defgroup usermods User-Defined Models
@ingroup modeling

## Creating Modeling Components
While MUQ has many built-in ModPieces, using MUQ on a new application typically requires creating one or more new model components.  This is accomplished by creating a new class that inherits from <code>ModPiece</code> and overides the evaluation method (and optionally the derivative functions).  To demonstrate this process, consider a simple <code>ModPiece</code> that evaluates a scalar linear regression model at several points to produce a single vector of predictions at those points.  Let \f$y_i\in\mathbb{R}\f$ denote the model prediction at a point \f$x_i\in\mathbb{R}\f$.  The model is given by
\f[
y_i = c_1 x_i + c_2,
\f]
where \f$c=[c_1,c_2]\f$ is a vector of coefficients.    The vectors \f$x\f$ and \f$c\f$ can be inputs to the ModPiece and the vector \f$y\f$ will be the only component in the vector of ModPiece outputs.   To define this ModPiece, we need to create a new class that inherits from <code>ModPiece</code>, tells the parent <code>ModPiece</code> constructor the sizes of the inputs and outputs, and then overrides the abstract <code>EvalauteImpl</code> function in the <code>ModPiece</code> base class.   Below is an example of how this could be accomplished
@codeblock{cpp,C++}
#include "MUQ/Modeling/ModPiece.h"

class SimpleModel : public muq::Modeling::ModPiece
{
public:
  SimpleModel(unsigned int numPts) : muq::Modeling::ModPiece({numPts,2},{numPts}) {};

protected:
  void EvaluateImpl(std::ref_vector<Eigen::VectorXd> const& inputs) override {
    Eigen::VectorXd const& x = inputs.at(0).get();
    Eigen::VectorXd const& c = inputs.at(1).get();

    Eigen::VectorXd y = c(0)*x + c(1)*Eigen::VectorXd::Ones(x.size());

    outputs.resize(1);
    outputs.at(0) = y;
  }
};
@endcodeblock
@codeblock{python,Python}
class SimpleModel : public muq::Modeling::ModPiece
{
public:
  SimpleModel(unsigned int numPts) : muq::Modeling::ModPiece({numPts,2},{numPts}) {};

protected:
  virtual void EvaluateImpl(std::ref_vector<Eigen::VectorXd> const& inputs) override {
    Eigen::VectorXd const& x = inputs.at(0).get();
    Eigen::VectorXd const& c = inputs.at(1).get();

    Eigen::VectorXd y = c(0)*x + c(1)*Eigen::VectorXd::Ones(x.size());

    outputs.resize(1);
    outputs.at(0) = y;
  }
};
@endcodeblock
The ModPiece constructor takes two vectors: one containing the dimensions of each input \f$N_1, N_2\f$ and one containing the dimensions of each output \f$M_1\f$.  Here we have used the [c++11 list initialization](https://en.cppreference.com/w/cpp/language/list_initialization) feature to specify the vectors (curly brackets) in a concise way.  In general however, the <code>ModPiece</code> constructor expects either an <code>Eigen::VectorXi</code> or a <code>std::vector<int></code> in c++ and a list or numpy array of ints in python.

Note that <code>EvaluateImpl</code> function does not return the result.  Instead, it sets the member variable <code>outputs</code>, which is a <code>std::vector</code> of <code>Eigen::VectorXd</code> vectors in c++ and a list of numpy arrays in Python.  Setting <code>outputs</code> instead of returning a vector allows MUQ to reduce the number of times data is copied in large computational graphs.  It also enables easy one-step caching, which prevents consecutive calls to the <code>EvaluateImpl</code> with the same inputs.

Using the <code>SimpleModel</code> ModPiece is identical to using the native ModPieces provided by MUQ:
@codeblock{cpp,C++}
unsigned int numPts = 10;
Eigen::VectorXd x = Eigen::VectorXd::Random(numPts);

Eigen::VectorXd c(2);
c << 1.0, 0.5;

auto mod = std::make_shared<SimpleModel>(numPts);

Eigen::VectorXd y = mod->Evaluate(x,c).at(0);
@endcodeblock
@codeblock{python,Python}
numPts = 10
x = np.random.randn(numPts)
c = np.array([1.0,0.5])

mod = SimpleModel(numPts)

y = mod.Evaluate(x,c)[0]
@endcodeblock

Notice that the <code>EvaluateImpl</code> function is defined in the <code>SimpleModel</code> class, but we call the <code>Evaluate</code> function when evaluating the model.  The <code>Evaluate</code> function is defined in the parent <code>ModPiece</code> class and calls the <code>EvaluateImpl</code> function.  The parent <code>Evaluate</code> function also checks the size of the input, supports one-step caching of the evaluation, and keeps track of average runtimes.

### Defining Derivative Information
The <code>EvaluateImpl</code> function is the only thing that must be implemented to define a new ModPiece.  All unimplemented derivative functions (e.g., <code>Jacobian</code>, <code>Gradient</code>, etc...) will default to finite difference implementations.  However, overriding the finite difference implementation is advantageous if derivatives can be computed analytically.   This can be accomplished in MUQ by implementing one or more of the <code>JacobianImpl</code>, <code>ApplyJacobianImpl</code>, <code>GradientImpl</code>, or <code>ApplyHessianImpl</code> functions.   Extending the <code>SimpleModel</code> definition above, all of the derivative functions could be implemented as
@codeblock{cpp,C++}
class SimpleModel : public muq::Modeling::ModPiece
{
public:
  SimpleModel(int numPts) : muq::Modeling::ModPiece({numPts,2},{numPts}) {};

protected:
  void EvaluateImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs) override
  {
    Eigen::VectorXd const& x = inputs.at(0).get();
    Eigen::VectorXd const& c = inputs.at(1).get();

    Eigen::VectorXd y = c(0)*x + c(1)*Eigen::VectorXd::Ones(x.size());

    outputs.resize(1);
    outputs.at(0) = y;
  };

  virtual void JacobianImpl(unsigned int outWrt,
                            unsigned int inWrt,
                            muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs) override
  {
    Eigen::VectorXd const& x = inputs.at(0).get();
    Eigen::VectorXd const& c = inputs.at(1).get();

    // Jacobian wrt x
    if(inWrt==0){
      jacobian = c(0)*Eigen::VectorXd::Identity(x.size(), x.size());

    // Jacobian wrt c
    }else{
      jacobian = Eigen::MatrixXd::Ones(outputSizes(0), inputSizes(inWrt));
      jacobian.col(0) = x;
    }
  }

  virtual void GradientImpl(unsigned int outWrt,
                            unsigned int inWrt,
                            muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs,
                            Eigen::VectorXd const& sens) override
  {
    Eigen::VectorXd const& x = inputs.at(0).get();
    Eigen::VectorXd const& c = inputs.at(1).get();

    // Gradient wrt x
    if(inWrt==0){
      gradient = c(0) * sens;

    // Gradient wrt c
    }else{
      gradient.resize(2);
      gradient(0) = x.dot(sens);
      gradient(1) = sens.sum();
    }
  }

  virtual void ApplyJacobianImpl(unsigned int outWrt,
                                 unsigned int inWrt,
                                 muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs,
                                 Eigen::VectorXd const& vec) override
  {
    Eigen::VectorXd const& x = inputs.at(0).get();
    Eigen::VectorXd const& c = inputs.at(1).get();

    // Jacobian wrt x
    if(inWrt==0){
      jacobianAction = c(0)*vec;

    // Jacobian wrt c
    }else{
      jacobianAction = vec(0)*x + vec(1)*Eigen::VectorXd::Ones(x.size());
    }
  }

  virtual void ApplyHessianImpl(unsigned int outWrt,
                                 unsigned int inWrt1,
                                 unsigned int inWrt2,
                                 muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs,
                                 Eigen::VectorXd const& sens,
                                 Eigen::VectorXd const& vec) override
  {
    Eigen::VectorXd const& x = inputs.at(0).get();
    Eigen::VectorXd const& c = inputs.at(1).get();

    // Apply d^2 / dxdc
    if((inWrt1==0)&&(inWrt2==1)){
      hessAction = vec(0) * sens;

    // Apply d^2 / dcdx
    }else if((inWrt2==0)&&(inWrt1==1)){
      hessAction.resize(2);
      hessAction(0) = sens.dot(vec);
      hessAction(1) = 0;

    // Apply d^2 / dxds
    }else if((inWrt1==0)&&(inWrt2==2)){
      hessAction = c(0) * vec;

    // Apply d^2 / dcds
    }else if((inWrt1==1)&&(inWrt2==2)){

      hessAction.resize(2);
      hessAction(0) = x.dot(vec);
      hessAction(1) = vec.sum();

    // Apply d^2/dx^2  or  d^2/dc^2  or  d^2/ds^2 or d^2 / dsdx or  d^2 / dsdc
    }else{
      hessAction = Eigen::VectorXd::Zero(inputSizes(inWrt1));
    }
  }
}; // end of class SimpleModel
@endcodeblock
@codeblock{python,Python}
class SimpleModel(mm.PyModPiece):

    def __init__(self, numPts):
        super(SimpleModel,self).__init__([numPts,2],[numPts])

    def EvaluateImpl(self, inputs):
        x,c  = inputs

        y = c[0]*x + c[1]

        self.outputs = [y]

    def JacobianImpl(self, outWrt, inWrt, inputs):
        x,c = inputs

        # Jacobian wrt x
        if(inWrt==0):
            self.jacobian = c[0]*np.eye(x.shape[0])

        # Jacobian wrt c
        else:
            self.jacobian =np.ones((self.outputSizes[0], self.inputSizes[inWrt]))
            self.jacobian[:,0] = x

    def GradientImpl(self, outWrt, inWrt, inputs, sens):
        x, c = inputs

        # Gradient wrt x
        if(inWrt==0):
            self.gradient = c[0] * sens

        # Gradient wrt c
        else:
            self.gradient = np.array([x.dot(sens),sens.sum()])

    def ApplyJacobianImpl(outWrt, inWrt, inputs, vec):
        x,c = inputs

        # Jacobian wrt x
        if(inWrt==0):
            self.jacobianAction = c[0]*vec

        # Jacobian wrt c
        else:
            self.jacobianAction = vec[0]*x + vec[1]*np.ones(x.shape[0])

    def ApplyHessianImpl(outWrt, inWrt1, inWrt2, inputs, sens, vec):
        x,c = inputs

        # Apply d^2 / dxdc
        if((inWrt1==0)&(inWrt2==1)):
            hessAction = vec[0] * sens

        # Apply d^2 / dcdx
        elif((inWrt2==0)&(inWrt1==1)):
            hessAction = np.array([sens.dot(vec),0])

        # Apply d^2 / dxds
        elif((inWrt1==0)&(inWrt2==2)):
            hessAction = c[0]*vec

        # Apply d^2 / dcds
        elif((inWrt1==1)&(inWrt2==2)):
            hessAction = np.array([x.dot(vec), vec.sum()])

        # Apply d^2/dx^2  or  d^2/dc^2  or  d^2/ds^2 or d^2 / dsdx or  d^2 / dsdc
        else:
            hessAction = np.zeros(self.inputSizes[inWrt1])
@endcodeblock

*/

/**
@defgroup modgraphs Combining Components: Model Graphs
@ingroup modeling

## Description
Individual ModPieces form the building blocks for larger more complicated models.  MUQ makes it possible to connect ModPieces together on a computational graph to define larger models and enable the reuse of existing ModPiece implementations.  The idea is to great graph of ModPieces where each node in the graph corresponds to a ModPiece and each edge represents a composition of one ModPiece with another.   For example, if functions \f$g(y)\f$ and \f$f(x)\f$ are implemented as ModPieces, we can define the composition \f$g(f(x))\f$ by creating a graph with two nodes and one edge from the output of \f$f\f$ to the input of \f$g\f$.  Visually, the resulting graph would look something like


![imgDef]

[imgDef]: SimpleWorkGraph.png "Simple WorkGraph"

In MUQ, model graphs are defined using the [WorkGraph](\ref muq::Modeling::WorkGraph) class.  The <code>AddNode</code> function in the <code>WorkGraph</code> class enables new model components to be added while the <code>AddEdge</code> allows us to make connections between the components.   Here's an example of implementing the simple $g(f(x))$ model in MUQ.  We use \f$f(x)=\sin(x)\f$ and \f$g(y)=\exp(y)\f$.
@codeblock{cpp,C++}
#include "MUQ/Modeling/WorkGraph.h"
#include "MUQ/Modeling/ModGraphPiece.h"
#include "MUQ/Modeling/CwiseOperators/CwiseUnaryOperator.h"

using namespace muq::Modeling;

int main(){
  unsigned int dim = 2;
  auto f = std::make_shared<SinOperator>(dim);
  auto g = std::make_shared<ExpOperator>(dim);

  auto graph = std::make_shared<WorkGraph>();

  graph->AddNode(f,"f");
  graph->AddNode(g,"g");
  graph->AddEdge("f",0,"g",0); // connect output 0 of f with input 0 of g

  auto gof = graph->CreateModPiece("f");

  return 0;
}
@endcodeblock
@codeblock{python,Python}
import muq.Modeling as mm

dim = 2
f = mm.SinOperator(dim)
g = mm.ExpOperator(dim)

graph = mm.WorkGraph()
graph.AddNode(f,"f")
graph.AddNode(g,"g")
graph.AddEdge("f",0,"g",0) # <- connect output 0 of f with input 0 of g

gof = graph.CreateModPiece("f")
@endcodeblock

The <code>AddEdge</code> function takes in the names of the nodes you wish to connect as well as integers specifying a particular output of <code>f</code> we want to connect to a particular input of <code>g</code>.   In this case, both the <code>SinOperator</code> and <code>ExpOperator</code> ModPieces have a single input and a single input.  Hence, we used "0" for both inputs and outputs.  The <code>CreateModPiece</code> function returns a chile of the <code>ModPiece</code> class that uses the graph to evaluate \f$f(g(x))\f$.  Becuase it's a ModPiece, all of the usual ModPiece derivative functions are available.  MUQ uses first and second order chain rules to analytically compute derivative information through the entire graph.

Consider another example where \f$x\in\mathbb{R}^4\f$ and we want to implement a model that computes \f$\sin(x_{1:2}) + \exp(x_{3:4})\f$.   First, we'll use the [SplitVector](\ref muq::Modeling::SplitVector) ModPiece to separate the vector \f$x\f$ into two vectors: \f$x_{1:2}\f$ and \f$x_{3:4}\f$.  After using the <code>SinOperator</code> and <code>ExpOperator</code>, we'll then sum the results using the <code>SumPiece</code> ModPiece.  The resulting graph should look like

![imgDef2]

[imgDef2]: SplitSumWorkGraph.png "Split-Sum WorkGraph"

The code to construct this computational graph in MUQ is given below. 
@codeblock{cpp,C++}
#include "MUQ/Modeling/WorkGraph.h"
#include "MUQ/Modeling/ModGraphPiece.h"
#include "MUQ/Modeling/SumPiece.h"
#include "MUQ/Modeling/SplitVector.h"
#include "MUQ/Modeling/CwiseOperators/CwiseUnaryOperator.h"

using namespace muq::Modeling;

int main(){
  auto f = std::make_shared<SinOperator>(2);
  auto g = std::make_shared<ExpOperator>(2);
  auto sum = std::make_shared<SumPiece>(2);

  // Will split x_{1:dim} into two equally sized vectors
  auto splitter = std::make_shared<SplitVector>(std::vector<int>{0,2}, // indices of output
                                                std::vector<int>{2,2}, // sizes of output
                                                4); // size of input

  auto graph = std::make_shared<WorkGraph>();

  graph->AddNode(splitter, "x12,x34");
  graph->AddNode(g,"g");
  graph->AddNode(f,"f");
  graph->AddEdge("x12,x34",0,"f",0); // connect output 0 of x12,x34 with input 0 of f
  graph->AddEdge("x12,x34",0,"g",0); // connect output 1 of x12,x34 with input 0 of g

  graph->AddNode(sum,"f+g");
  graph->AddEdge("f",0,"f+g",0); // connect output 0 of f with input 0 of f+g
  graph->AddEdge("g",0,"f+g",1); // connect output 0 of g with intpu 1 of f+g

  auto mod = graph->CreateModPiece("f+g");

  return 0;
}
@endcodeblock
@codeblock{python,Python}
import muq.Modeling as mm

f = mm.SinOperator(2)
g = mm.ExpOperator(2)
sum = mm.SumPiece(2)

# Will split x_{1:dim} into two equally sized vectors
splitter = mm.SplitVector([0,2], # indices of output
                          [2,2], # sizes of output
                          4)     # size of input

graph = mm.WorkGraph()

graph.AddNode(splitter, "x12,x34");
graph.AddNode(g,"g")
graph.AddNode(f,"f")
graph.AddEdge("x12,x34",0,"f",0) # connect output 0 of x12,x34 with input 0 of f
graph.AddEdge("x12,x34",0,"g",0) # connect output 1 of x12,x34 with input 0 of g

graph.AddNode(sum,"f+g");
graph.AddEdge("f",0,"f+g",0) # connect output 0 of f with input 0 of f+g
graph.AddEdge("g",0,"f+g",1) # connect output 0 of g with intpu 1 of f+g

mod = graph.CreateModPiece("f+g")
@endcodeblock


*/


#endif
