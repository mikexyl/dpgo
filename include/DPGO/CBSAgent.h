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

// Forward declarations for CBS types
namespace cbs {
  class BPSAM;
}

namespace gtsam {
  class NonlinearFactorGraph;
  class Values;
}

namespace DPGO {

/**
 * @brief PGO agent using CBS/BPSAM optimizer instead of Riemannian optimization
 * 
 * This class implements distributed pose graph optimization using the 
 * Collaborative Belief Space (CBS) framework with Belief Propagation SAM (BPSAM).
 * It inherits from PGOAgentBase and implements the optimizer-specific methods
 * using GTSAM factor graphs and CBS solvers.
 */
class CBSAgent : public PGOAgentBase {
public:
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

  /**
   * @brief Perform local pose graph optimization using CBS
   * @return trajectory estimate in matrix form T = [R1 t1 ... Rn tn] in an arbitrary frame
   */
  Matrix localPoseGraphOptimization() override;

  /**
   * @brief Set CBS-specific optimization parameters
   * 
   * @param max_iterations Maximum iterations for CBS
   * @param damping Damping factor for BPSAM
   * @param convergence_threshold Convergence threshold
   */
  void setCBSParams(size_t max_iterations, double damping, double convergence_threshold);

  /**
   * @brief override the base funciton, and populate the "lifting matrix" with poses and diagonal covariance
   */
  bool getLiftingMatrix(Matrix &M) const override;

protected:
  // CBS-specific parameters
  size_t max_iterations_;
  double damping_;
  double convergence_threshold_;

  // GTSAM factor graph and values (using pointers to avoid including full headers)
  std::shared_ptr<gtsam::NonlinearFactorGraph> graph_;
  std::shared_ptr<gtsam::Values> current_estimate_;

  /**
   * @brief Convert current pose graph to GTSAM factor graph
   */
  void updateFactorGraph();

  /**
   * @brief Update poses from GTSAM values
   */
  void updatePosesFromGTSAM(const gtsam::Values &values);
};

} // namespace DPGO
