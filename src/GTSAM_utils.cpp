/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/GTSAM_utils.h>
#include <cbs/key.h>
#include <glog/logging.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/LabeledSymbol.h>

namespace DPGO {
namespace GTSAMUtils {

gtsam::Pose3 toGTSAMPose3(const Matrix &pose) {
  // Extract rotation matrix and translation vector
  Eigen::Matrix3d R = pose.block<3, 3>(0, 0);
  Eigen::Vector3d t = pose.block<3, 1>(0, 3);

  return gtsam::Pose3(gtsam::Rot3(R), gtsam::Point3(t));
}

Matrix fromGTSAMPose3(const gtsam::Pose3 &pose) {
  Matrix T = Matrix::Identity(4, 4);

  T.block<3, 3>(0, 0) = pose.rotation().matrix();
  T.block<3, 1>(0, 3) = pose.translation();

  return T;
}

gtsam::BetweenFactor<gtsam::Pose3>::shared_ptr
toGTSAMBetweenFactor3D(size_t robot_id,
                       const RelativeSEMeasurement &measurement,
                       char robotSymbol, char poseSymbol) {
  // bool is_inter_robot = (measurement.r1 != measurement.r2);
  // if it's inter robot, then only add the outgoing factor, to avoid duplicate
  // in CBS
  // if (is_inter_robot && measurement.r2 == robot_id) {
  //   return nullptr;
  // }

  // Create keys
  gtsam::Key key1 = cbs::toPoseKey(measurement.r1 + 'a', measurement.p1);
  gtsam::Key key2 = cbs::toPoseKey(measurement.r2 + 'a', measurement.p2);

  // Convert relative pose measurement
  Eigen::Matrix3d R = measurement.R;
  Eigen::Vector3d t = measurement.t;
  gtsam::Rot3 rotation(R);
  gtsam::Point3 translation(t);
  gtsam::Pose3 relativePose(rotation, translation);

  // Create noise model
  // DPGO uses precision (inverse covariance), GTSAM uses covariance
  Eigen::Matrix<double, 6, 1> sigmas;
  sigmas << 1.0 / std::sqrt(measurement.kappa), // roll
      1.0 / std::sqrt(measurement.kappa),       // pitch
      1.0 / std::sqrt(measurement.kappa),       // yaw
      1.0 / std::sqrt(measurement.tau),         // x
      1.0 / std::sqrt(measurement.tau),         // y
      1.0 / std::sqrt(measurement.tau);         // z

  gtsam::SharedNoiseModel noise = gtsam::noiseModel::Diagonal::Sigmas(sigmas);

  // auto robust_noise = gtsam::noiseModel::Robust::Create(
  // gtsam::noiseModel::mEstimator::DCS::Create(3), noise);

  return boost::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
      key1, key2, relativePose, noise);
}

gtsam::NonlinearFactorGraph
toGTSAMFactorGraph3D(size_t robot_id,
                     const std::vector<RelativeSEMeasurement> &measurements,
                     char robotSymbol, char poseSymbol) {

  gtsam::NonlinearFactorGraph graph;

  int n_rejected_loops{0};
  for (const auto &measurement : measurements) {
    if (auto factor = toGTSAMBetweenFactor3D(robot_id, measurement, robotSymbol,
                                             poseSymbol)) {
      graph.add(factor);
    } else {
      n_rejected_loops++;
    }
  }

  LOG(INFO) << "GTSAMUtils::toGTSAMFactorGraph3D(): Robot " << robot_id
            << " converted " << measurements.size() - n_rejected_loops << " / "
            << measurements.size()
            << " measurements to GTSAM factors. Rejected " << n_rejected_loops
            << " inter-robot incoming loops.";

  return graph;
}

gtsam::NonlinearFactorGraph
poseGraphToGTSAM3D(size_t robot_id, const PoseGraph &poseGraph,
                   char robotSymbol, char poseSymbol, bool includeInactive) {
  // Get all measurements from the pose graph
  return toGTSAMFactorGraph3D(robot_id, poseGraph.measurements(), robotSymbol,
                              poseSymbol);
}

gtsam::Values toGTSAMValues3D(const PoseArray &poses, unsigned robotID,
                              char robotSymbol, char poseSymbol) {

  gtsam::Values values;

  for (unsigned i = 0; i < poses.n(); ++i) {
    Matrix T = poses.pose(i);
    gtsam::Pose3 pose = toGTSAMPose3(T);
    gtsam::Key key = cbs::toPoseKey(robotID + 'a', i);
    values.insert(key, pose);
  }

  return values;
}

} // namespace GTSAMUtils
} // namespace DPGO
