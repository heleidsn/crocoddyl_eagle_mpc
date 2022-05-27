#include "crocoddyl/core/solvers/ipopt.hpp"

namespace crocoddyl {

SolverIpopt::SolverIpopt(boost::shared_ptr<crocoddyl::ShootingProblem> problem)
    : SolverAbstract(problem), ipopt_iface_(new IpoptInterface(problem)), ipopt_app_(IpoptApplicationFactory()) {
  ipopt_app_->Options()->SetNumericValue("tol", 3.82e-6);
  ipopt_app_->Options()->SetStringValue("mu_strategy", "adaptive");

  ipopt_status_ = ipopt_app_->Initialize();

  if (ipopt_status_ != Ipopt::Solve_Succeeded) {
    std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
  }
}

bool SolverIpopt::solve(const std::vector<Eigen::VectorXd>& init_xs, const std::vector<Eigen::VectorXd>& init_us,
                        const std::size_t maxiter, const bool is_feasible, const double regInit) {
  setCandidate(init_xs, init_us, is_feasible);
  ipopt_iface_->set_xs(xs_);
  ipopt_iface_->set_us(us_);

  ipopt_app_->Options()->SetIntegerValue("max_iter", maxiter);
  ipopt_status_ = ipopt_app_->OptimizeTNLP(ipopt_iface_);

  std::copy(ipopt_iface_->get_xs().begin(), ipopt_iface_->get_xs().end(), xs_.begin());
  std::copy(ipopt_iface_->get_us().begin(), ipopt_iface_->get_us().end(), us_.begin());

  return ipopt_status_ == Ipopt::Solve_Succeeded;
}

SolverIpopt::~SolverIpopt() {}

void SolverIpopt::computeDirection(const bool recalc) {}
double SolverIpopt::tryStep(const double steplength) { return 0.0; }
double SolverIpopt::stoppingCriteria() { return 0.0; }
const Eigen::Vector2d& SolverIpopt::expectedImprovement() { return Eigen::Vector2d::Zero(); }

void SolverIpopt::setStringIpoptOption(const std::string& tag, const std::string& value) {
  ipopt_app_->Options()->SetStringValue(tag, value);
}

void SolverIpopt::setNumericIpoptOption(const std::string& tag, Ipopt::Number value) {
  ipopt_app_->Options()->SetNumericValue(tag, value);
}

void SolverIpopt::setConsiderControlBounds(const bool& consider) {
  ipopt_iface_->set_consider_control_bounds(consider);
}

}  // namespace crocoddyl