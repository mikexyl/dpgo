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
#include <chrono>
#include <glog/logging.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

namespace DPGO {

CBSAgent::CBSAgent(unsigned ID, const PGOAgentParameters &params,
                   OptWatcherFunc opt_watcher)
    : PGOAgentBase(ID, params), opt_watcher_(opt_watcher) {
  cbs::BPSAM::Params bpsam_params;
  bpsam_params.robot_id = ID + 'a';
  bpsam_params.gbp_update_params.type = gbp::GaussianMergeType::Contract;
  bpsam_params.gbp_update_params.contract_alpha = 0.2;
  bpsam_params.gbp_update_params.d_reset = 0.6;
  ISAM2Params isam2_params;
  isam2_params.relinearizeSkip = 1;
  isam2_params.cacheLinearizedFactors = true; // Cache to avoid double free
  isam2_params.findUnusedFactorSlots = true;
  ISAM2GaussNewtonParams gn_params;
  isam2_params.optimizationParams = gn_params;
  isam2_params.setRelinearizeThreshold(0.01);
  bpsam_params.sam_params_ = isam2_params;
  bpsam_params.enable_gkcm = false;
  bpsam_ = std::make_shared<cbs::BPSAM>(bpsam_params);

  LOG(INFO) << "Created CBS agent " << ID << " in " << params.d << "D";
}

CBSAgent::~CBSAgent() { LOG(INFO) << "Destroyed CBS agent " << mID; }

bool CBSAgent::performOptimization(bool doOpt, bool accel) {
  static auto last_update_time = std::chrono::high_resolution_clock::now();
  auto start_time = std::chrono::high_resolution_clock::now();
  auto elapsed_since_last =
      std::chrono::duration_cast<std::chrono::milliseconds>(start_time -
                                                            last_update_time)
          .count();
  VLOG(1) << "CBSAgent::performOptimization(): Robot " << mID
          << " time since last update: " << elapsed_since_last << " ms";
  last_update_time = start_time;

  CHECK(!accel, "CBSAgent does not support acceleration. Set accel = false.");
  CHECK_NOTNULL(bpsam_);
  // TODO: Implement local CBS optimization
  // 1. Convert pose graph measurements to GTSAM factor graph
  // 2. Create initial values
  // 3. Run BPSAM
  // 4. Return optimized poses as Matrix

  // read beliefs from shared poses
  std::map<Key, std::vector<std::pair<cbs::AgentId, gbp::Gaussian>>> beliefs;
  int n_beliefs{0}, n_no_sigma{0};
  std::stringstream belief_key_ss;
  for (auto kv : neighborPoseDict) {
    const PoseID &nID = kv.first;
    const auto &var = kv.second;
    auto labeled_key = cbs::toPoseKey(nID.robot_id + 'a', nID.frame_id);
    // if var's sigma is empty or zero, skip
    if (var.Sigma_.rows() == 0 or var.Sigma_.cols() == 0 or
        var.Sigma_.norm() < 1e-6) {
      n_no_sigma++;
      continue;
    }
    gtsam::Pose3 T_w_p = GTSAMUtils::toGTSAMPose3(var.pose());
    CHECK(mNeighborLocalOrigins.find(nID.robot_id) !=
          mNeighborLocalOrigins.end());
    gtsam::Pose3 T_w_r = GTSAMUtils::toGTSAMPose3(
        mNeighborLocalOrigins.at(nID.robot_id).matrix());
    auto T_r_p = T_w_r.inverse() * T_w_p;
    n_beliefs++;
    gbp::Gaussian gauss(labeled_key, gtsam::traits<gtsam::Pose3>::Logmap(T_r_p),
                        var.Sigma_, 1);
    beliefs[labeled_key].emplace_back(nID.robot_id + 'a', gauss);
    belief_key_ss << gtsam::MultiRobotKeyFormatter(labeled_key) << " ";
  }
  VLOG(1) << "CBSAgent::performOptimization(): Robot " << mID << " has "
          << n_beliefs << " beliefs, " << n_no_sigma
          << " beliefs with no sigma.";
  VLOG(1) << "belief keys: " << belief_key_ss.str();

  // print pose graph size for debug
  VLOG(1) << "CBSAgent::performOptimization(): Robot " << mID
          << " pose graph has " << mPoseGraph->numOdometry()
          << " odometry factors, " << mPoseGraph->numPrivateLoopClosures()
          << " private loop closures, " << mPoseGraph->numSharedLoopClosures()
          << " shared loop closures.";

  auto gtsam_graph_all = GTSAMUtils::poseGraphToGTSAM3D(
      getID(), *mPoseGraph, cbs::kRobotLabel, cbs::kPoseLabel, true);

  std::stringstream factor_key_ss;
  for (const auto &key : gtsam_graph_all.keys()) {
    factor_key_ss << gtsam::MultiRobotKeyFormatter(key) << " ";
  }
  VLOG(1) << "CBSAgent::performOptimization(): Robot " << mID << " has "
          << gtsam_graph_all.size()
          << " factors. keys: " << factor_key_ss.str();

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
  for (unsigned int i = 0; i < num_poses(); ++i) {
    auto key = cbs::toPoseKey(mID + 'a', i);
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
  auto first_key = cbs::toPoseKey(mID + 'a', 0);
  gtsam::Pose3 prior_pose = initial_values.at<gtsam::Pose3>(first_key);
  T_w_o_ = prior_pose;
  if (bpsam_->getFactorsUnsafe().size() == 0) {
    auto prior_noise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-4);
    auto prior_factor = boost::make_shared<gtsam::PriorFactor<gtsam::Pose3>>(
        first_key, Pose3::Identity(), prior_noise);
    // print prior pose
    VLOG(1) << "Adding prior factor at first pose: "
            << gtsam::MultiRobotKeyFormatter(first_key)
            << " pose: " << prior_pose.translation().transpose();
    gtsam_graph.add(prior_factor);
  }

  cbs::BPSAM::UpdateParams update_params;
  // update once to initialize the graph
  bpsam_->addBeliefs(beliefs);
  VLOG(1) << "CBSAgent::performOptimization(): Robot " << mID
          << " updating with " << gtsam_graph.size() << " new factors";

  // Compute initial error for mLocalOptResult
  double fInit = 0.0;
  if (bpsam_->getFactorsUnsafe().size() > 0) {
    fInit = bpsam_->getFactorsUnsafe().error(bpsam_->calculateEstimate());
  }

  try {
    bpsam_->update(gtsam_graph, initial_values, update_params);
  } catch (gtsam::IndeterminantLinearSystemException &e) {
    // save graph as .dot file
    bpsam_->getFactorsUnsafe().saveGraph("/tmp/cbs_agent_" +
                                             std::to_string(mID) + ".dot",
                                         gtsam::MultiRobotKeyFormatter);

    LOG(FATAL) << "CBSAgent::performOptimization(): BPSAM update failed: "
               << e.what()
               << "key: " << MultiRobotKeyFormatter(e.nearbyVariable());
  }

  bpsam_->saveLocalGraphG2o(mParams.logDirectory + "/bpsam_robot_" +
                                std::to_string(mID) + ".g2o",
                            true);

  // Compute final error and populate mLocalOptResult
  auto end_time = std::chrono::high_resolution_clock::now();
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();
  double fOpt = bpsam_->getFactorsUnsafe().error(bpsam_->calculateEstimate());
  double error_decrease = fInit - fOpt;
  mLocalOptResult = ROPTResult(true, fInit, 0.0, fOpt, 0.0, elapsed_ms);

  // log iteration number and results stats for debug
  VLOG(1) << "CBSAgent::Robot " << mID << " iteration " << mIterationNumber
          << " finished BPSAM update in " << elapsed_ms << " ms"
          << " initial error: " << fInit << " final error: " << fOpt
          << " error decrease: " << error_decrease
          << " num factors: " << bpsam_->getFactorsUnsafe().size()
          << " num variables: " << bpsam_->getVariableIndex().size();

  if (opt_watcher_) {
    opt_watcher_(bpsam_->getFactorsUnsafe(), bpsam_->calculateEstimate());
  }

  // Return empty Matrix for now
  Matrix result_X(d, num_poses() * (d + 1));
  auto values = bpsam_->calculateEstimate();
  for (unsigned int i = 0; i < num_poses(); ++i) {
    auto key = cbs::toPoseKey(mID + 'a', i);
    if (values.exists(key)) {
      gtsam::Pose3 T_o_p = values.at<gtsam::Pose3>(key);
      gtsam::Pose3 pose = T_w_o_.value() * T_o_p;
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

  {
    std::lock_guard<std::mutex> lock(marginals_mutex_);
    cached_marginals_.clear();
  }

  X = PoseArray(d, num_poses());
  X.setData(result_X);

  return true;
}

Matrix CBSAgent::getPoseMarginal(const PoseID &pose_id) const {
  auto key = cbs::toPoseKey(pose_id.robot_id + 'a', pose_id.frame_id);
  if (bpsam_->marginalizationFactors().size() == 0) {
    return Matrix();
  }
  try {
    std::lock_guard<std::mutex> lock(marginals_mutex_);
    Matrix marginal;
    // if (cached_marginals_.find(key) != cached_marginals_.end()) {
    //   marginal = cached_marginals_.at(key);
    //   return marginal;
    // }

    marginal = bpsam_->marginalCovariance(key);
    return marginal;
  } catch (...) {
    LOG(ERROR) << "CBSAgent::getPoseMarginal(): Failed to get marginal for key "
               << gtsam::MultiRobotKeyFormatter(key);
    return Matrix();
  }
}

} // namespace DPGO
