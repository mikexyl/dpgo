/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef PGOAGENT_H
#define PGOAGENT_H

#include <DPGO/PGOAgentBase.h>
#include <DPGO/QuadraticProblem.h>

/*Define the namespace*/
namespace DPGO {

/**
 * @brief DC2PGO
 */
class PGOAgent : public PGOAgentBase {
 public:

  /**
   * @brief Constructor
   * @param ID
   * @param params
   */
  PGOAgent(unsigned ID, const PGOAgentParameters &params);

  /**
   * @brief Destructor
   */
  ~PGOAgent() override = default;

 protected:
  /**
   * @brief Perform the core optimization step using DPGO's Riemannian optimizer
   * @param doOptimization Whether this agent is selected to perform optimization
   * @param acceleration true to use acceleration
   * @return true if update is successful
   */
  bool performOptimization(bool doOptimization, bool acceleration) override;
};

}  // namespace DPGO

#endif
