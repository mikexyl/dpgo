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
#include <cbs/gbp/gbp.h>
#include <glog/logging.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

namespace DPGO {

CBSAgent::CBSAgent(unsigned ID, const PGOAgentParameters &params)
    : PGOAgentBase(ID, params) {
  cbs::BPSAM::Params bpsam_params;
  bpsam_params.root_id = ID;
  bpsam_params.gbp_update_params.type = gbp::GaussianMergeType::Contract;
  bpsam_params.gbp_update_params.contract_alpha = 0.5;
  bpsam_params.gbp_update_params.d_reset = 0.5;
  ISAM2Params isam2_params;
  isam2_params.relinearizeSkip = 1;
  isam2_params.cacheLinearizedFactors = true; // Cache to avoid double free
  isam2_params.findUnusedFactorSlots = true;
  ISAM2GaussNewtonParams gn_params;
  isam2_params.optimizationParams = gn_params;
  isam2_params.setRelinearizeThreshold(0.01);
  bpsam_params.sam_params_ = isam2_params;
  bpsam_ = std::make_shared<cbs::BPSAM>(bpsam_params);

  LOG(INFO) << "Created CBS agent " << ID << " in " << params.d << "D";
}

CBSAgent::~CBSAgent() { LOG(INFO) << "Destroyed CBS agent " << mID; }

bool CBSAgent::performOptimization(bool doOpt, bool accel) {
  CHECK(!accel, "CBSAgent does not support acceleration. Set accel = false.");
  CHECK_NOTNULL(bpsam_);
  // TODO: Implement local CBS optimization
  // 1. Convert pose graph measurements to GTSAM factor graph
  // 2. Create initial values
  // 3. Run BPSAM
  // 4. Return optimized poses as Matrix

  // read beliefs from shared poses
  std::map<Key, std::vector<std::pair<cbs::AgentId, gbp::Gaussian>>> beliefs;
  for (auto kv : neighborPoseDict) {
    const PoseID &nID = kv.first;
    const auto &var = kv.second;
    auto labeled_key = cbs::toPoseKey(nID.robot_id, nID.frame_id);
    // if var's sigma is empty, skip
    if (var.Sigma_.rows() == 0) {
      continue;
    }
    gbp::Gaussian gauss(labeled_key,
                        gtsam::traits<gtsam::Pose3>::Logmap(
                            GTSAMUtils::toGTSAMPose3(var.pose())),
                        var.Sigma_, 1);
    beliefs[labeled_key].emplace_back(nID.robot_id, gauss);
  }

  auto gtsam_graph_all = GTSAMUtils::poseGraphToGTSAM3D(
      *mPoseGraph, cbs::kRobotLabel, cbs::kPoseLabel, true);

  // because cbs stores the belief changes internally, we can't reset bpsam
  // create a new graph with only new factors
  gtsam::NonlinearFactorGraph gtsam_graph;
  for (const auto &factor : gtsam_graph_all) {
    if (not factor)
      continue;
    bool factor_exists{false};
    const auto &key = factor->keys()[0];
    if (bpsam_->getVariableIndex().find(key) !=
        bpsam_->getVariableIndex().end()) {
      // check if the factor already exist
      for (auto factor_index : bpsam_->getVariableIndex().find(key)->second) {
        auto old_factor = bpsam_->getFactor(factor_index);
        // check if keys are the same
        if (old_factor->keys() == factor->keys()) {
          factor_exists = true;
          break;
        }
      }
    }

    if (not factor_exists) {
      gtsam_graph.add(factor);
    }
  }

  // get current estimates from X
  gtsam::Values initial_values;
  for (int i = 0; i < num_poses(); ++i) {
    auto key = cbs::toPoseKey(mID, i);
    Matrix Ri = X.getData().block(0, i * (d + 1), d, d);
    Matrix ti = X.getData().block(0, i * (d + 1) + d, d, 1);
    gtsam::Rot3 rot(Ri);
    gtsam::Point3 trans(ti);
    gtsam::Pose3 pose(rot, trans);
    initial_values.insert(key, pose);
  }

  // add any missing values to initial_values
  //! TODO: initial values are not initialized
  for (const auto &factor : gtsam_graph) {
    for (const auto &key : factor->keys()) {
      if (!initial_values.exists(key)) {
        initial_values.insert(key, gtsam::Pose3());
      }
    }
  }

  // if bpsam has no factors, which means it's the first optimization, add a
  // prior
  if (bpsam_->getFactorsUnsafe().size() == 0) {
    // anchor pose
    auto first_key = cbs::toPoseKey(mID, 0);
    gtsam::Pose3 prior_pose = initial_values.at<gtsam::Pose3>(first_key);
    auto prior_noise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-4);
    auto prior_factor = boost::make_shared<gtsam::PriorFactor<gtsam::Pose3>>(
        first_key, prior_pose, prior_noise);
    // print prior pose
    LOG(INFO) << "Adding prior factor at first pose: "
              << gtsam::MultiRobotKeyFormatter(first_key)
              << " pose: " << prior_pose.translation().transpose();
    gtsam_graph.add(prior_factor);
  }

  cbs::BPSAM::UpdateParams update_params;
  // update once to initialize the graph
  // bpsam_->addBeliefs<gtsam::Pose3>(beliefs);
  bpsam_->update(gtsam_graph, initial_values, update_params);

  // Return empty Matrix for now
  Matrix result_X(d, num_poses() * (d + 1));
  auto values = bpsam_->calculateEstimate();
  for (int i = 0; i < num_poses(); ++i) {
    auto key = cbs::toPoseKey(mID, i);
    if (values.exists(key)) {
      gtsam::Pose3 pose = values.at<gtsam::Pose3>(key);
      Matrix Ri = pose.rotation().matrix();
      Matrix ti(3, 1);
      ti(0) = pose.translation().x();
      ti(1) = pose.translation().y();
      ti(2) = pose.translation().z();
      result_X.block(0, i * (d + 1), d, d) = Ri;
      result_X.block(0, i * (d + 1) + d, d, 1) = ti;
    } else {
      LOG(FATAL)
          << "CBSAgent::localPoseGraphOptimization(): Missing value for key "
          << gtsam::MultiRobotKeyFormatter(key);
    }
  }

  bpsam_->setMarginalizationGraph(cbs::BPSAM::MarginalizationType::LOCAL);

  X = PoseArray(d, num_poses());
  X.setData(result_X);

  return true;
}

Matrix CBSAgent::getPoseMarginal(const PoseID &pose_id) const {
  auto key = cbs::toPoseKey(pose_id.robot_id, pose_id.frame_id);
  try {
    if (bpsam_->marginalizationFactors().size() == 0) {
      return Matrix();
    }
    auto marginal = bpsam_->marginalCovariance(key);
    return marginal;
  } catch (...) {
    // print marginal keys
    bpsam_->marginalizationFactors().print("marginalization factors:",
                                           MultiRobotKeyFormatter);
    LOG(FATAL) << "CBSAgent::getPoseMarginal(): Failed to get marginal for key "
               << gtsam::MultiRobotKeyFormatter(key);
    throw;
  }
}

} // namespace DPGO
