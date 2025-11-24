/* ----------------------------------------------------------------------------
 * Example demonstrating CBSAgent usage
 * -------------------------------------------------------------------------- */

#include <DPGO/CBSAgent.h>
#include <iostream>

using namespace DPGO;

int main() {
  std::cout << "=== CBSAgent Example ===" << std::endl;

  // Create PGO agent parameters
  int dim = 3;  // 3D poses (SE(3))
  int rank = 3; // Relaxation rank
  PGOAgentParameters params(dim, rank);
  params.numRobots = 2;
  params.asynchronous = false;
  params.localInitializationMethod = InitializationMethod::Odometry;
  params.acceleration = true;
  params.verbose = true;
  params.logData = false;

  // Create CBS agent
  CBSAgent agent(0, params);

  std::cout << "Created CBSAgent with ID " << agent.getID() << std::endl;
  std::cout << "Dimension: " << agent.dimension() << std::endl;

  // TODO: The implementation of performOptimization() and
  // localPoseGraphOptimization() should be completed to:
  // 1. Convert DPGO pose graphs to GTSAM factor graphs using GTSAM_utils
  // 2. Use CBS/BPSAM solver for optimization
  // 3. Convert results back to DPGO format

  std::cout << "\n=== Example completed! ===" << std::endl;
  std::cout << "Note: Implement the TODO methods in CBSAgent.cpp to enable CBS "
               "optimization"
            << std::endl;

  return 0;
}
