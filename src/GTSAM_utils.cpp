/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/GTSAM_utils.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/LabeledSymbol.h>
#include <glog/logging.h>

namespace DPGO {
namespace GTSAMUtils {

gtsam::Key makeKey(unsigned robotID, unsigned poseID, char robotSymbol, char poseSymbol, bool useMultiRobotKeys) {
  if (useMultiRobotKeys) {
    // Use LabeledSymbol: robot label encoded in key
    return gtsam::LabeledSymbol(poseSymbol, robotID, poseID).key();
  } else {
    // Simple symbol (single robot case)
    return gtsam::Symbol(poseSymbol, poseID).key();
  }
}

gtsam::Pose2 toGTSAMPose2(const Matrix& pose) {
  CHECK_EQ(pose.rows(), 3);
  CHECK_EQ(pose.cols(), 3);
  
  // Extract rotation and translation
  double theta = std::atan2(pose(1, 0), pose(0, 0));
  double x = pose(0, 2);
  double y = pose(1, 2);
  
  return gtsam::Pose2(x, y, theta);
}

gtsam::Pose3 toGTSAMPose3(const Matrix& pose) {
  CHECK_EQ(pose.rows(), 4);
  CHECK_EQ(pose.cols(), 4);
  
  // Extract rotation matrix and translation vector
  Eigen::Matrix3d R = pose.block<3, 3>(0, 0);
  Eigen::Vector3d t = pose.block<3, 1>(0, 3);
  
  return gtsam::Pose3(gtsam::Rot3(R), gtsam::Point3(t));
}

Matrix fromGTSAMPose2(const gtsam::Pose2& pose) {
  Matrix T = Matrix::Identity(3, 3);
  
  double theta = pose.theta();
  double c = std::cos(theta);
  double s = std::sin(theta);
  
  T(0, 0) = c;
  T(0, 1) = -s;
  T(1, 0) = s;
  T(1, 1) = c;
  T(0, 2) = pose.x();
  T(1, 2) = pose.y();
  
  return T;
}

Matrix fromGTSAMPose3(const gtsam::Pose3& pose) {
  Matrix T = Matrix::Identity(4, 4);
  
  T.block<3, 3>(0, 0) = pose.rotation().matrix();
  T.block<3, 1>(0, 3) = pose.translation();
  
  return T;
}

gtsam::BetweenFactor<gtsam::Pose2>::shared_ptr toGTSAMBetweenFactor2D(
    const RelativeSEMeasurement& measurement,
    char robotSymbol,
    char poseSymbol) {
  
  // Create keys
  bool multiRobot = (measurement.r1 != measurement.r2);
  gtsam::Key key1 = makeKey(measurement.r1, measurement.p1, robotSymbol, poseSymbol, multiRobot);
  gtsam::Key key2 = makeKey(measurement.r2, measurement.p2, robotSymbol, poseSymbol, multiRobot);
  
  // Convert relative pose measurement
  gtsam::Pose2 relativePose = toGTSAMPose2(measurement.R);
  relativePose = gtsam::Pose2(measurement.t(0), measurement.t(1), relativePose.theta());
  
  // Create noise model
  // DPGO uses precision (inverse covariance), GTSAM uses covariance
  Eigen::Vector3d sigmas;
  sigmas << 1.0 / std::sqrt(measurement.tau),  // x
            1.0 / std::sqrt(measurement.tau),  // y
            1.0 / std::sqrt(measurement.kappa); // theta
  
  gtsam::SharedNoiseModel noise = gtsam::noiseModel::Diagonal::Sigmas(sigmas);
  
  // Scale by weight for robust optimization
  if (measurement.weight < 1.0) {
    noise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(measurement.weight),
        noise);
  }
  
  return boost::make_shared<gtsam::BetweenFactor<gtsam::Pose2>>(
      key1, key2, relativePose, noise);
}

gtsam::BetweenFactor<gtsam::Pose3>::shared_ptr toGTSAMBetweenFactor3D(
    const RelativeSEMeasurement& measurement,
    char robotSymbol,
    char poseSymbol) {
  
  // Create keys
  bool multiRobot = (measurement.r1 != measurement.r2);
  gtsam::Key key1 = makeKey(measurement.r1, measurement.p1, robotSymbol, poseSymbol, multiRobot);
  gtsam::Key key2 = makeKey(measurement.r2, measurement.p2, robotSymbol, poseSymbol, multiRobot);
  
  // Convert relative pose measurement
  Eigen::Matrix3d R = measurement.R;
  Eigen::Vector3d t = measurement.t;
  gtsam::Rot3 rotation(R);
  gtsam::Point3 translation(t);
  gtsam::Pose3 relativePose(rotation, translation);
  
  // Create noise model
  // DPGO uses precision (inverse covariance), GTSAM uses covariance
  Eigen::Matrix<double, 6, 1> sigmas;
  sigmas << 1.0 / std::sqrt(measurement.kappa),  // roll
            1.0 / std::sqrt(measurement.kappa),  // pitch
            1.0 / std::sqrt(measurement.kappa),  // yaw
            1.0 / std::sqrt(measurement.tau),    // x
            1.0 / std::sqrt(measurement.tau),    // y
            1.0 / std::sqrt(measurement.tau);    // z
  
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

gtsam::NonlinearFactorGraph toGTSAMFactorGraph2D(
    const std::vector<RelativeSEMeasurement>& measurements,
    char robotSymbol,
    char poseSymbol) {
  
  gtsam::NonlinearFactorGraph graph;
  
  for (const auto& measurement : measurements) {
    graph.push_back(toGTSAMBetweenFactor2D(measurement, robotSymbol, poseSymbol));
  }
  
  return graph;
}

gtsam::NonlinearFactorGraph toGTSAMFactorGraph3D(
    const std::vector<RelativeSEMeasurement>& measurements,
    char robotSymbol,
    char poseSymbol) {
  
  gtsam::NonlinearFactorGraph graph;
  
  for (const auto& measurement : measurements) {
    graph.push_back(toGTSAMBetweenFactor3D(measurement, robotSymbol, poseSymbol));
  }
  
  return graph;
}

gtsam::NonlinearFactorGraph poseGraphToGTSAM2D(
    const PoseGraph& poseGraph,
    char robotSymbol,
    char poseSymbol,
    bool includeInactive) {
  
  // Get all measurements from the pose graph
  std::vector<RelativeSEMeasurement> measurements = poseGraph.localMeasurements();
  
  return toGTSAMFactorGraph2D(measurements, robotSymbol, poseSymbol);
}

gtsam::NonlinearFactorGraph poseGraphToGTSAM3D(
    const PoseGraph& poseGraph,
    char robotSymbol,
    char poseSymbol,
    bool includeInactive) {
  
  // Get all measurements from the pose graph
  std::vector<RelativeSEMeasurement> measurements = poseGraph.localMeasurements();
  
  return toGTSAMFactorGraph3D(measurements, robotSymbol, poseSymbol);
}

gtsam::Values toGTSAMValues2D(
    const PoseArray& poses,
    unsigned robotID,
    char robotSymbol,
    char poseSymbol) {
  
  gtsam::Values values;
  
  CHECK_EQ(poses.d(), 2) << "PoseArray dimension must be 2 for SE(2)";
  
  for (unsigned i = 0; i < poses.n(); ++i) {
    Matrix T = poses.pose(i);
    gtsam::Pose2 pose = toGTSAMPose2(T);
    gtsam::Key key = makeKey(robotID, i, robotSymbol, poseSymbol, false);
    values.insert(key, pose);
  }
  
  return values;
}

gtsam::Values toGTSAMValues3D(
    const PoseArray& poses,
    unsigned robotID,
    char robotSymbol,
    char poseSymbol) {
  
  gtsam::Values values;
  
  CHECK_EQ(poses.d(), 3) << "PoseArray dimension must be 3 for SE(3)";
  
  for (unsigned i = 0; i < poses.n(); ++i) {
    Matrix T = poses.pose(i);
    gtsam::Pose3 pose = toGTSAMPose3(T);
    gtsam::Key key = makeKey(robotID, i, robotSymbol, poseSymbol, false);
    values.insert(key, pose);
  }
  
  return values;
}

PoseArray fromGTSAMValues2D(
    const gtsam::Values& values,
    unsigned robotID,
    char poseSymbol) {
  
  // Count poses for this robot
  unsigned numPoses = 0;
  for (const auto& key_value : values) {
    gtsam::Symbol symbol(key_value.key);
    if (symbol.chr() == poseSymbol) {
      numPoses++;
    }
  }
  
  CHECK_GT(numPoses, 0) << "No poses found in GTSAM Values";
  
  // Build matrix data for PoseArray
  Matrix allPoses(3, 3 * numPoses);
  
  for (unsigned i = 0; i < numPoses; ++i) {
    gtsam::Key key = gtsam::Symbol(poseSymbol, i).key();
    if (values.exists(key)) {
      gtsam::Pose2 pose = values.at<gtsam::Pose2>(key);
      Matrix T = fromGTSAMPose2(pose);
      allPoses.block<3, 3>(0, 3 * i) = T;
    } else {
      LOG(WARNING) << "Pose " << i << " not found in GTSAM Values";
      allPoses.block<3, 3>(0, 3 * i) = Matrix::Identity(3, 3);
    }
  }
  
  PoseArray poses(2, numPoses);
  poses.setData(allPoses);
  return poses;
}

PoseArray fromGTSAMValues3D(
    const gtsam::Values& values,
    unsigned robotID,
    char poseSymbol) {
  
  // Count poses for this robot
  unsigned numPoses = 0;
  for (const auto& key_value : values) {
    gtsam::Symbol symbol(key_value.key);
    if (symbol.chr() == poseSymbol) {
      numPoses++;
    }
  }
  
  CHECK_GT(numPoses, 0) << "No poses found in GTSAM Values";
  
  // Build matrix data for PoseArray
  Matrix allPoses(4, 4 * numPoses);
  
  for (unsigned i = 0; i < numPoses; ++i) {
    gtsam::Key key = gtsam::Symbol(poseSymbol, i).key();
    if (values.exists(key)) {
      gtsam::Pose3 pose = values.at<gtsam::Pose3>(key);
      Matrix T = fromGTSAMPose3(pose);
      allPoses.block<4, 4>(0, 4 * i) = T;
    } else {
      LOG(WARNING) << "Pose " << i << " not found in GTSAM Values";
      allPoses.block<4, 4>(0, 4 * i) = Matrix::Identity(4, 4);
    }
  }
  
  PoseArray poses(3, numPoses);
  poses.setData(allPoses);
  return poses;
}

}  // namespace GTSAMUtils
}  // namespace DPGO
