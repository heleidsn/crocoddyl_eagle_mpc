///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/frame-axis-alignment.hpp"

#include "python/crocoddyl/multibody/multibody.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualFrameAxisAlignment() {
  bp::register_ptr_to_python<
      boost::shared_ptr<ResidualModelFrameAxisAlignment>>();

  bp::class_<ResidualModelFrameAxisAlignment, bp::bases<ResidualModelAbstract>>(
      "ResidualModelFrameAxisAlignment",
      "This residual function defines the tracking of theframe placement "
      "residual as r = p - pref, with p and pref "
      "as\n"
      "the current and reference frame placements, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, pinocchio::FrameIndex,
               Eigen::Vector3d, std::size_t, std::size_t>(
          bp::args("self", "state", "id", "axisred", "direction", "nu"),
          "Initialize the frame placement residual model.\n\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param axisref: reference axis\n"
          ":param direction: frame axis that has to coincide with reference "
          "axis\n"
          ":param nu: dimension of control vector"))
      .def<void (ResidualModelFrameAxisAlignment::*)(
          const boost::shared_ptr<ResidualDataAbstract> &,
          const Eigen::Ref<const Eigen::VectorXd> &,
          const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calc", &ResidualModelFrameAxisAlignment::calc,
          bp::args("self", "data", "x", "u"),
          "Compute the frame placement residual.\n\n"
          ":param data: residual data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (ResidualModelFrameAxisAlignment::*)(
          const boost::shared_ptr<ResidualDataAbstract> &,
          const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelFrameAxisAlignment::*)(
          const boost::shared_ptr<ResidualDataAbstract> &,
          const Eigen::Ref<const Eigen::VectorXd> &,
          const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calcDiff", &ResidualModelFrameAxisAlignment::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Compute the Jacobians of the frame placement residual.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (ResidualModelFrameAxisAlignment::*)(
          const boost::shared_ptr<ResidualDataAbstract> &,
          const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calcDiff", &ResidualModelAbstract::calcDiff,
          bp::args("self", "data", "x"))
      .def("createData", &ResidualModelFrameAxisAlignment::createData,
           bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the frame placement residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. "
           "This function\n"
           "returns the allocated data for the frame placement residual.\n"
           ":param data: shared data\n"
           ":return residual data.")
      .add_property("id", &ResidualModelFrameAxisAlignment::get_id,
                    &ResidualModelFrameAxisAlignment::set_id,
                    "reference frame id")
      .add_property(
          "reference",
          bp::make_function(&ResidualModelFrameAxisAlignment::get_reference,
                            bp::return_internal_reference<>()),
          &ResidualModelFrameAxisAlignment::set_reference,
          "reference frame placement");

  bp::register_ptr_to_python<
      boost::shared_ptr<ResidualDataFrameAxisAlignment>>();

  bp::class_<ResidualDataFrameAxisAlignment, bp::bases<ResidualDataAbstract>>(
      "ResidualDataFrameAxisAlignment",
      "Data for frame placement residual.\n\n",
      bp::init<ResidualModelFrameAxisAlignment *, DataCollectorAbstract *>(
          bp::args("self", "model", "data"),
          "Create frame placement residual data.\n\n"
          ":param model: frame placement residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3>>()])
      .add_property("pinocchio",
                    bp::make_getter(&ResidualDataFrameAxisAlignment::pinocchio,
                                    bp::return_internal_reference<>()),
                    "pinocchio data")
      .add_property("axis",
                    bp::make_getter(&ResidualDataFrameAxisAlignment::axis,
                                    bp::return_internal_reference<>()),
                    "Frame vector that has to be aligned")
      .add_property("dotJvec",
                    bp::make_getter(&ResidualDataFrameAxisAlignment::dotJvec,
                                    bp::return_internal_reference<>()),
                    "Jacoabian of the dot product w.r.t. input vector")
      .add_property("ractJrot",
                    bp::make_getter(&ResidualDataFrameAxisAlignment::ractJrot,
                                    bp::return_internal_reference<>()),
                    "Jacobian of the rotation action w.r.t. rotation")
      .add_property("fJf",
                    bp::make_getter(&ResidualDataFrameAxisAlignment::fJf,
                                    bp::return_internal_reference<>()),
                    "local Jacobian of the frame");
}

}  // namespace python
}  // namespace crocoddyl