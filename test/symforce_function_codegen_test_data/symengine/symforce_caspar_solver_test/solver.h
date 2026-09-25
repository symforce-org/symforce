#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "shared_indices.h"
#include "solver_params.h"

namespace caspar {

enum class ExitReason { MAX_ITERATIONS, CONVERGED_SCORE_THRESHOLD, CONVERGED_DIAG_EXIT };

struct IterationData {
  int solver_iter;
  int pcg_iter;
  double score_current;
  double score_best;
  double step_quality;
  double diag;
  double dt_inc;
  double dt_tot;
  bool step_accepted;
};

struct SolveResult {
  double initial_score;
  double final_score;
  int iteration_count;
  double runtime;
  ExitReason exit_reason;
  std::vector<IterationData> iterations;
};

class GraphSolver {
 public:
  /**
   * Base constructor.
   *
   * @param params: The params to use for the solver
   * @param Matrix21_num_max the maximum number of Matrix21s
   * @param rosenbrock_num_max the maximum number of rosenbrocks
   */
  GraphSolver(const SolverParams<double>& params, size_t Matrix21_num_max,
              size_t rosenbrock_num_max, int device_id = 0);

  // This class is managing cuda memory and cannot be copied.
  GraphSolver(const GraphSolver&) = delete;
  GraphSolver& operator=(const GraphSolver&) = delete;

  GraphSolver(GraphSolver&&) = default;
  GraphSolver& operator=(GraphSolver&&) = default;

  ~GraphSolver();

  /**
   * Set the solver parameters.
   */
  void set_params(const SolverParams<double>& params);

  /**
   * Run the solver.
   */
  SolveResult solve(bool print_progress = false, bool verbose_logging = false);

  /**
   * Finish the indices.
   *
   * This function has to be called after all indices are set and before the
   * solve function is called.
   */
  void finish_indices();

  /**
   * Get the number of allocated bytes.
   */
  size_t get_allocation_size();

  /**
   * Set the current value for the Matrix21 nodes from the stacked host data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetMatrix21NodesFromStackedHost(const double* const data, size_t offset, size_t num);

  /**
   * Set the current value for the Matrix21 nodes from the stacked device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetMatrix21NodesFromStackedDevice(const double* const data, size_t offset, size_t num);

  /**
   * Read the current value for the Matrix21 nodes into the stacked output host
   * data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetMatrix21NodesToStackedHost(double* const data, size_t offset, size_t num);

  /**
   * Read the current value for the Matrix21 nodes into the stacked output
   * device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetMatrix21NodesToStackedDevice(double* const data, size_t offset, size_t num);

  /**
   * Set the current number of active nodes of type Matrix21.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetMatrix21Num(size_t num);

  /**
   * Set the indices for the x argument for the Rosenbrock factor from host.
   */
  void SetRosenbrockXIndicesFromHost(const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the x argument for the Rosenbrock factor from device.
   */
  void SetRosenbrockXIndicesFromDevice(const unsigned int* const indices, size_t num);

  /**
   * Set the current number of Rosenbrock factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetRosenbrockNum(size_t num);

 private:
  SolverParams<double> params_;
  int device_id_;
  uint8_t* origin_ptr_;
  size_t scratch_inout_size_;
  size_t allocation_size_;

  int solver_iter_;
  int pcg_iter_;

  bool indices_valid_;

  double pcg_r_0_norm2_;
  double pcg_r_kp1_norm2_;

  size_t Matrix21_num_;
  size_t Matrix21_num_max_;
  size_t rosenbrock_num_;
  size_t rosenbrock_num_max_;

  size_t get_nbytes();
  double LinearizeFirst();
  void Linearize();
  double DoResJacFirst();
  void DoResJac();
  void DoNormalize();
  void DoJtjpDirect();
  void DoAlphaFirst();
  void DoAlpha();
  void DoUpdateStepFirst();
  void DoUpdateStep();
  void DoUpdateRFirst();
  void DoUpdateR();
  double DoRetractScore();
  void DoBeta();
  void DoUpdateP();
  void DoUpdateMp();
  double GetPredDecrease();

  double* marker__start_;
  double* nodes__Matrix21__storage_current_;
  double* nodes__Matrix21__storage_check_;
  double* nodes__Matrix21__storage_new_best_;
  SharedIndex* facs__rosenbrock__args__x__idx_shared_;
  double* marker__scratch_inout_;
  double* facs__rosenbrock__res_;
  double* facs__rosenbrock__args__x__jac_;
  double* nodes__Matrix21__z_;
  double* nodes__Matrix21__z_end__;
  double* nodes__Matrix21__p_;
  double* nodes__Matrix21__p_end__;
  double* nodes__Matrix21__step_;
  double* nodes__Matrix21__step_end__;
  double* marker__w_start_;
  double* nodes__Matrix21__w_;
  double* marker__w_end_;
  double* marker__r_0_start_;
  double* nodes__Matrix21__r_0_;
  double* marker__r_0_end_;
  double* marker__r_k_start_;
  double* nodes__Matrix21__r_k_;
  double* marker__r_k_end_;
  double* marker__Mp_start_;
  double* nodes__Matrix21__Mp_;
  double* marker__Mp_end_;
  double* marker__precond_start_;
  double* nodes__Matrix21__precond_diag_;
  double* nodes__Matrix21__precond_tril_;
  double* marker__precond_end_;
  double* marker__jp_start_;
  double* facs__rosenbrock__jp_;
  double* marker__jp_end_;
  double* solver__current_diag_;
  double* solver__alpha_numerator_;
  double* solver__alpha_denominator_;
  double* solver__alpha_;
  double* solver__neg_alpha_;
  double* solver__beta_numerator_;
  double* solver__beta_;
  double* solver__r_0_norm2_tot_;
  double* solver__r_kp1_norm2_tot_;
  double* solver__pred_decrease_tot_;
  double* solver__res_tot_;
};

}  // namespace caspar