
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "demo_opt.hpp"

namespace {

  class simpleFn : public Demo_Opt::ObjectiveFn
  {
    public:

      simpleFn() { }

      double compute_obj(const std::vector<double> & params, bool verbose = false) override
      {
        double fn = 0.0;
        for( auto const x : params )
          fn += pow((x-1.0), 4);
        return fn;
      };
  };

} // anonymous namespace


int main(int argc, char *argv[])
{
  std::shared_ptr<Demo_Opt> demoOpt(new Demo_Opt());
  std::shared_ptr<Demo_Opt::ObjectiveFn> fn(new simpleFn());
  demoOpt->register_obj_fn(&*fn);

  std::vector<double> init  = { 0.0};
  std::vector<double> lower = {-5.0};
  std::vector<double> upper = { 5.0};
  demoOpt->set_problem_data(init, lower, upper);

  demoOpt->execute(true);
}
