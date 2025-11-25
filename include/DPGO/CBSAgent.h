/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   CBSAgent.h
 * @brief  PGO agent using CBS (Collaborative Belief Space) optimizer
 * @author Xiangyu Liu
 */

#pragma once

#include "DPGO/PGOAgentBase.h"
#include <cbs/bpsam/bpsam.h>

// Forward declarations for CBS types
namespace cbs {
class BPSAM;
}

namespace gtsam {
class NonlinearFactorGraph;
class Values;
} // namespace gtsam

namespace DPGO {

/**
 * @brief PGO agent using CBS/BPSAM optimizer instead of Riemannian optimization
 *
 * This class implements distributed pose graph optimization using the
 * Collaborative Belief Space (CBS) framework with Belief Propagation SAM
 * (BPSAM). It inherits from PGOAgentBase and implements the optimizer-specific
 * methods using GTSAM factor graphs and CBS solvers.
 */
class CBSAgent : public PGOAgentBase {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  /**
   * @brief Construct a new CBS Agent
   *
   * @param ID Robot ID
   * @param params PGO agent parameters
   */
  CBSAgent(unsigned ID, const PGOAgentParameters &params);

  /**
   * @brief Destructor
   */
  virtual ~CBSAgent();

  /**
   * @brief Perform CBS optimization on the local pose graph
   *
   * @param doOpt Whether to perform optimization
   * @param accel Whether to use acceleration
   * @return true if optimization succeeded
   */
  bool performOptimization(bool doOpt, bool accel) override;

  Matrix getPoseMarginal(const PoseID &pose_id) const override;

protected:
  // GTSAM factor graph and values (using pointers to avoid including full
  // headers)
  std::shared_ptr<gtsam::NonlinearFactorGraph> graph_;
  std::shared_ptr<gtsam::Values> current_estimate_;
  std::shared_ptr<cbs::BPSAM> bpsam_;
};

} // namespace DPGO
