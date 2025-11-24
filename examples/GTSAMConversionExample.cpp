/* ----------------------------------------------------------------------------
 * Example demonstrating DPGO to GTSAM conversion utilities
 * -------------------------------------------------------------------------- */

#include <DPGO/GTSAM_utils.h>
#include <DPGO/RelativeSEMeasurement.h>
#include <iostream>

using namespace DPGO;

int main() {
  std::cout << "=== DPGO to GTSAM Conversion Example ===" << std::endl;

  // Create some sample DPGO measurements for SE(3)
  std::vector<RelativeSEMeasurement> measurements;
  
  // Odometry measurement between pose 0 and 1
  Eigen::Matrix3d R01 = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t01(1.0, 0.0, 0.0);
  measurements.emplace_back(0, 0, 0, 1, R01, t01, 100.0, 100.0);
  
  // Odometry measurement between pose 1 and 2
  Eigen::Matrix3d R12 = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t12(0.0, 1.0, 0.0);
  measurements.emplace_back(0, 0, 1, 2, R12, t12, 100.0, 100.0);
  
  // Loop closure measurement between pose 2 and 0
  Eigen::Matrix3d R20 = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t20(-1.0, -1.0, 0.0);
  measurements.emplace_back(0, 0, 2, 0, R20, t20, 50.0, 50.0);
  
  std::cout << "Created " << measurements.size() << " DPGO measurements" << std::endl;
  
  // Convert to GTSAM NonlinearFactorGraph
  gtsam::NonlinearFactorGraph graph = GTSAMUtils::toGTSAMFactorGraph3D(measurements);
  
  std::cout << "Converted to GTSAM factor graph with " << graph.size() << " factors" << std::endl;
  
  // Create initial values
  gtsam::Values initial;
  initial.insert(gtsam::Symbol('x', 0), gtsam::Pose3());
  initial.insert(gtsam::Symbol('x', 1), gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0)));
  initial.insert(gtsam::Symbol('x', 2), gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(1.0, 1.0, 0.0)));
  
  std::cout << "Created initial values with " << initial.size() << " poses" << std::endl;
  
  // Convert GTSAM values back to DPGO PoseArray
  PoseArray poses = GTSAMUtils::fromGTSAMValues3D(initial, 0, 'x');
  
  std::cout << "Converted back to DPGO PoseArray with " << poses.n() << " poses" << std::endl;
  std::cout << "Dimension: " << poses.d() << std::endl;
  
  std::cout << "\\nFirst pose:" << std::endl;
  std::cout << poses.pose(0) << std::endl;
  
  std::cout << "\\n=== Example completed successfully! ===" << std::endl;
  
  return 0;
}
