/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/PGOAgent.h>
#include <DPGO/DPGO_solver.h>
#include <DPGO/QuadraticOptimizer.h>
#include <glog/logging.h>

#include <Eigen/CholmodSupport>

using std::unique_lock;
using std::mutex;

namespace DPGO {

PGOAgent::PGOAgent(unsigned ID, const PGOAgentParameters &params)
    : PGOAgentBase(ID, params) {
  // Derived class specific initialization if needed
}

bool PGOAgent::performOptimization(bool doOptimization, bool acceleration) {
  // Lock during local optimization
  unique_lock<mutex> tLock(mPosesMutex);
  unique_lock<mutex> mLock(mMeasurementsMutex);
  unique_lock<mutex> nLock(mNeighborPosesMutex);
  
  LOG_IF(INFO, mParams.verbose) << "Robot " << getID() << " optimizes at iteration " << iteration_number();
  if (acceleration) CHECK(mParams.acceleration);
  CHECK(mState == PGOAgentState::INITIALIZED);

  // Initialize pose graph for optimization
  if (acceleration) {
    mPoseGraph->setNeighborPoses(neighborAuxPoseDict);
  } else {
    mPoseGraph->setNeighborPoses(neighborPoseDict);
  }

  // Skip optimization if cannot construct data matrices for some reason
  if (!mPoseGraph->constructDataMatrices()) {
    LOG(WARNING) << "Robot " << getID() << " cannot construct data matrices... Skip optimization.";
    mLocalOptResult = ROPTResult(false);
    return false;
  }

  // Initialize optimizer
  QuadraticProblem problem(mPoseGraph);
  QuadraticOptimizer optimizer(&problem, mParams.localOptimizationParams);
  optimizer.setVerbose(mParams.verbose);

  // Starting solution
  Matrix X0;
  if (acceleration) {
    X0 = Y.getData();
  } else {
    X0 = X.getData();
  }
  CHECK(X0.rows() == relaxation_rank());
  CHECK(X0.cols() == (dimension() + 1) * num_poses());

  // Optimize!
  X.setData(optimizer.optimize(X0));

  // Print optimization statistics
  mLocalOptResult = optimizer.getOptResult();
  if (mParams.verbose) {
    printf("df: %f, init_gradnorm: %f, opt_gradnorm: %f. \n",
           mLocalOptResult.fInit - mLocalOptResult.fOpt,
           mLocalOptResult.gradNormInit,
           mLocalOptResult.gradNormOpt);
  }

  return true;
}

}  // namespace DPGO
