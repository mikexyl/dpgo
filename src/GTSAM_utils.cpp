/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/GTSAM_utils.h>
#include <glog/logging.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/LabeledSymbol.h>

namespace DPGO {
namespace GTSAMUtils {

gtsam::Key makeKey(unsigned robotID, unsigned poseID, char robotSymbol,
                   char poseSymbol) {
  return gtsam::LabeledSymbol(poseSymbol, robotID, poseID).key();
}

gtsam::Pose3 toGTSAMPose3(const Matrix &pose) {
  CHECK_EQ(pose.rows(), 3) << "matrix shape: " << pose.rows() << "x"
                           << pose.cols();
  CHECK_EQ(pose.cols(), 4) << "matrix shape: " << pose.rows() << "x"
                           << pose.cols();

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
toGTSAMBetweenFactor3D(const RelativeSEMeasurement &measurement,
                       char robotSymbol, char poseSymbol) {

  // Create keys
  gtsam::Key key1 =
      makeKey(measurement.r1, measurement.p1, robotSymbol, poseSymbol);
  gtsam::Key key2 =
      makeKey(measurement.r2, measurement.p2, robotSymbol, poseSymbol);

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

  // Scale by weight for robust optimization
  if (measurement.weight < 1.0) {
    noise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(measurement.weight),
        noise);
  }

  return boost::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
      key1, key2, relativePose, noise);
}

gtsam::NonlinearFactorGraph
toGTSAMFactorGraph3D(const std::vector<RelativeSEMeasurement> &measurements,
                     char robotSymbol, char poseSymbol) {

  gtsam::NonlinearFactorGraph graph;

  for (const auto &measurement : measurements) {
    graph.push_back(
        toGTSAMBetweenFactor3D(measurement, robotSymbol, poseSymbol));
  }

  return graph;
}

gtsam::NonlinearFactorGraph poseGraphToGTSAM3D(const PoseGraph &poseGraph,
                                               char robotSymbol,
                                               char poseSymbol,
                                               bool includeInactive) {

  // Get all measurements from the pose graph
  std::vector<RelativeSEMeasurement> measurements =
      poseGraph.localMeasurements();

  return toGTSAMFactorGraph3D(measurements, robotSymbol, poseSymbol);
}

gtsam::Values toGTSAMValues3D(const PoseArray &poses, unsigned robotID,
                              char robotSymbol, char poseSymbol) {

  gtsam::Values values;

  CHECK_EQ(poses.d(), 3) << "PoseArray dimension must be 3 for SE(3)";

  for (unsigned i = 0; i < poses.n(); ++i) {
    Matrix T = poses.pose(i);
    gtsam::Pose3 pose = toGTSAMPose3(T);
    gtsam::Key key = makeKey(robotID, i, robotSymbol, poseSymbol);
    values.insert(key, pose);
  }

  return values;
}

} // namespace GTSAMUtils
} // namespace DPGO
