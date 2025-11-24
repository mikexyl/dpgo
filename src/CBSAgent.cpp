/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   CBSAgent.cpp
 * @brief  PGO agent using CBS (Collaborative Belief Space) optimizer
 * @author Xiangyu Liu
 */

#include "DPGO/CBSAgent.h"
#include "DPGO/GTSAM_utils.h"
#include <glog/logging.h>

namespace DPGO {

CBSAgent::CBSAgent(unsigned ID, const PGOAgentParameters &params)
    : PGOAgentBase(ID, params), max_iterations_(100), damping_(1.0),
      convergence_threshold_(1e-5) {
  LOG(INFO) << "Created CBS agent " << ID << " in " << params.d << "D";
}

CBSAgent::~CBSAgent() { LOG(INFO) << "Destroyed CBS agent " << mID; }

bool CBSAgent::performOptimization(bool doOpt, bool accel) {
  // TODO: Implement CBS optimization
  // 1. Convert pose graph to GTSAM factor graph using GTSAM_utils
  // 2. Create BPSAM solver
  // 3. Run optimization
  // 4. Update X with optimized values

  LOG(WARNING) << "CBSAgent::performOptimization() not yet implemented";
  return false;
}

Matrix CBSAgent::localPoseGraphOptimization() {
  // TODO: Implement local CBS optimization
  // 1. Convert pose graph measurements to GTSAM factor graph
  // 2. Create initial values
  // 3. Run BPSAM
  // 4. Return optimized poses as Matrix

  LOG(WARNING) << "CBSAgent::localPoseGraphOptimization() not yet implemented";

  // Return empty Matrix for now
  Matrix result;
  return result;
}

void CBSAgent::setCBSParams(size_t max_iterations, double damping,
                            double convergence_threshold) {
  max_iterations_ = max_iterations;
  damping_ = damping;
  convergence_threshold_ = convergence_threshold;

  LOG(INFO) << "CBS parameters set: max_iterations=" << max_iterations_
            << ", damping=" << damping_
            << ", convergence_threshold=" << convergence_threshold_;
}

void CBSAgent::updateFactorGraph() {
  // TODO: Update graph_ from current pose graph
  // Use GTSAMUtils::poseGraphToGTSAM2D() or poseGraphToGTSAM3D()

  LOG(WARNING) << "CBSAgent::updateFactorGraph() not yet implemented";
}

void CBSAgent::updatePosesFromGTSAM(const gtsam::Values &values) {
  // TODO: Update X from GTSAM values
  // Use GTSAMUtils::fromGTSAMValues2D() or fromGTSAMValues3D()

  LOG(WARNING) << "CBSAgent::updatePosesFromGTSAM() not yet implemented";
}

bool CBSAgent::getLiftingMatrix(Matrix &M) const {

}

} // namespace DPGO
