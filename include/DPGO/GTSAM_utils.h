/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef DPGO_GTSAM_UTILS_H
#define DPGO_GTSAM_UTILS_H

#include <DPGO/DPGO_types.h>
#include <DPGO/PoseGraph.h>
#include <DPGO/RelativeSEMeasurement.h>
#include <DPGO/manifold/Poses.h>

#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <Eigen/Dense>
#include <memory>

namespace DPGO {

/**
 * @brief Utility functions for converting between DPGO and GTSAM data
 * structures
 */
namespace GTSAMUtils {

/**
 * @brief Convert a DPGO RelativeSEMeasurement to a GTSAM BetweenFactor for
 * SE(2)
 * @param measurement The DPGO measurement
 * @param robotSymbol The GTSAM symbol character for robots (default 'r')
 * @param poseSymbol The GTSAM symbol character for poses (default 'x')
 * @return Shared pointer to GTSAM BetweenFactor<Pose2>
 */
gtsam::BetweenFactor<gtsam::Pose2>::shared_ptr
toGTSAMBetweenFactor2D(const RelativeSEMeasurement &measurement,
                       char robotSymbol = 'r', char poseSymbol = 'x');

/**
 * @brief Convert a DPGO RelativeSEMeasurement to a GTSAM BetweenFactor for
 * SE(3)
 * @param measurement The DPGO measurement
 * @param robotSymbol The GTSAM symbol character for robots (default 'r')
 * @param poseSymbol The GTSAM symbol character for poses (default 'x')
 * @return Shared pointer to GTSAM BetweenFactor<Pose3>
 */
gtsam::BetweenFactor<gtsam::Pose3>::shared_ptr
toGTSAMBetweenFactor3D(const RelativeSEMeasurement &measurement,
                       char robotSymbol = 'r', char poseSymbol = 'x');

/**
 * @brief Convert DPGO pose graph measurements to GTSAM NonlinearFactorGraph for
 * SE(2)
 * @param measurements Vector of DPGO measurements
 * @param robotSymbol The GTSAM symbol character for robots (default 'r')
 * @param poseSymbol The GTSAM symbol character for poses (default 'x')
 * @return GTSAM NonlinearFactorGraph
 */
gtsam::NonlinearFactorGraph
toGTSAMFactorGraph2D(const std::vector<RelativeSEMeasurement> &measurements,
                     char robotSymbol = 'r', char poseSymbol = 'x');

/**
 * @brief Convert DPGO pose graph measurements to GTSAM NonlinearFactorGraph for
 * SE(3)
 * @param measurements Vector of DPGO measurements
 * @param robotSymbol The GTSAM symbol character for robots (default 'r')
 * @param poseSymbol The GTSAM symbol character for poses (default 'x')
 * @return GTSAM NonlinearFactorGraph
 */
gtsam::NonlinearFactorGraph
toGTSAMFactorGraph3D(const std::vector<RelativeSEMeasurement> &measurements,
                     char robotSymbol = 'r', char poseSymbol = 'x');

/**
 * @brief Convert DPGO PoseGraph to GTSAM NonlinearFactorGraph for SE(3)
 * @param poseGraph The DPGO pose graph
 * @param robotSymbol The GTSAM symbol character for robots (default 'r')
 * @param poseSymbol The GTSAM symbol character for poses (default 'x')
 * @param includeInactive If true, include measurements with inactive neighbors
 * @return GTSAM NonlinearFactorGraph
 */
gtsam::NonlinearFactorGraph poseGraphToGTSAM3D(const PoseGraph &poseGraph,
                                               char robotSymbol = 'r',
                                               char poseSymbol = 'x',
                                               bool includeInactive = false);

/**
 * @brief Convert DPGO Pose (SE(3)) matrix to GTSAM Pose3
 * @param pose DPGO pose matrix (4x4 for SE(3))
 * @return GTSAM Pose3
 */
gtsam::Pose3 toGTSAMPose3(const Matrix &pose);

/**
 * @brief Convert DPGO PoseArray to GTSAM Values for SE(3)
 * @param poses DPGO pose array
 * @param robotID Robot ID
 * @param robotSymbol The GTSAM symbol character for robots (default 'r')
 * @param poseSymbol The GTSAM symbol character for poses (default 'x')
 * @return GTSAM Values containing all poses
 */
gtsam::Values toGTSAMValues3D(const PoseArray &poses, unsigned robotID = 0,
                              char robotSymbol = 'r', char poseSymbol = 'x');

} // namespace GTSAMUtils

} // namespace DPGO

#endif // DPGO_GTSAM_UTILS_H
