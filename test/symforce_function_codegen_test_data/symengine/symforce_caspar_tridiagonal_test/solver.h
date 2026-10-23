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
   * @param ChainA_num_max the maximum number of ChainAs
   * @param ChainB_num_max the maximum number of ChainBs
   * @param chain_a_link_num_max the maximum number of chain_a_links
   * @param chain_a_prior_num_max the maximum number of chain_a_priors
   * @param chain_b_link_num_max the maximum number of chain_b_links
   * @param chain_b_prior_num_max the maximum number of chain_b_priors
   * @param chain_coupling_num_max the maximum number of chain_couplings
   */
  GraphSolver(const SolverParams<double>& params, size_t ChainA_num_max, size_t ChainB_num_max,
              size_t chain_a_link_num_max, size_t chain_a_prior_num_max,
              size_t chain_b_link_num_max, size_t chain_b_prior_num_max,
              size_t chain_coupling_num_max, int device_id = 0);

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
   * Set the current value for the ChainA nodes from the stacked host data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetChainANodesFromStackedHost(const double* const data, size_t offset, size_t num);

  /**
   * Set the current value for the ChainA nodes from the stacked device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetChainANodesFromStackedDevice(const double* const data, size_t offset, size_t num);

  /**
   * Read the current value for the ChainA nodes into the stacked output host
   * data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetChainANodesToStackedHost(double* const data, size_t offset, size_t num);

  /**
   * Read the current value for the ChainA nodes into the stacked output device
   * data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetChainANodesToStackedDevice(double* const data, size_t offset, size_t num);

  /**
   * Set the current number of active nodes of type ChainA.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetChainANum(size_t num);

  /**
   * Set the current value for the ChainB nodes from the stacked host data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetChainBNodesFromStackedHost(const double* const data, size_t offset, size_t num);

  /**
   * Set the current value for the ChainB nodes from the stacked device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetChainBNodesFromStackedDevice(const double* const data, size_t offset, size_t num);

  /**
   * Read the current value for the ChainB nodes into the stacked output host
   * data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetChainBNodesToStackedHost(double* const data, size_t offset, size_t num);

  /**
   * Read the current value for the ChainB nodes into the stacked output device
   * data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetChainBNodesToStackedDevice(double* const data, size_t offset, size_t num);

  /**
   * Set the current number of active nodes of type ChainB.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetChainBNum(size_t num);

  /**
   * Set the values for the weight consts ChainALink factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetChainALinkWeightDataFromStackedHost(const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight consts ChainALink factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetChainALinkWeightDataFromStackedDevice(const double* const data, size_t offset,
                                                size_t num);

  /**
   * Set the values for the offset consts ChainALink factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetChainALinkOffsetDataFromStackedHost(const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the offset consts ChainALink factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetChainALinkOffsetDataFromStackedDevice(const double* const data, size_t offset,
                                                size_t num);

  /**
   * Set the current number of ChainALink factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetChainALinkNum(size_t num);

  /**
   * Set the indices for the a argument for the ChainAPrior factor from host.
   */
  void SetChainAPriorAIndicesFromHost(const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the a argument for the ChainAPrior factor from device.
   */
  void SetChainAPriorAIndicesFromDevice(const unsigned int* const indices, size_t num);

  /**
   * Set the values for the target consts ChainAPrior factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetChainAPriorTargetDataFromStackedHost(const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the target consts ChainAPrior factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetChainAPriorTargetDataFromStackedDevice(const double* const data, size_t offset,
                                                 size_t num);

  /**
   * Set the current number of ChainAPrior factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetChainAPriorNum(size_t num);

  /**
   * Set the values for the weight consts ChainBLink factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetChainBLinkWeightDataFromStackedHost(const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight consts ChainBLink factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetChainBLinkWeightDataFromStackedDevice(const double* const data, size_t offset,
                                                size_t num);

  /**
   * Set the values for the offset consts ChainBLink factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetChainBLinkOffsetDataFromStackedHost(const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the offset consts ChainBLink factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetChainBLinkOffsetDataFromStackedDevice(const double* const data, size_t offset,
                                                size_t num);

  /**
   * Set the current number of ChainBLink factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetChainBLinkNum(size_t num);

  /**
   * Set the indices for the b argument for the ChainBPrior factor from host.
   */
  void SetChainBPriorBIndicesFromHost(const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the b argument for the ChainBPrior factor from device.
   */
  void SetChainBPriorBIndicesFromDevice(const unsigned int* const indices, size_t num);

  /**
   * Set the values for the target consts ChainBPrior factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetChainBPriorTargetDataFromStackedHost(const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the target consts ChainBPrior factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetChainBPriorTargetDataFromStackedDevice(const double* const data, size_t offset,
                                                 size_t num);

  /**
   * Set the current number of ChainBPrior factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetChainBPriorNum(size_t num);

  /**
   * Set the indices for the a argument for the ChainCoupling factor from host.
   */
  void SetChainCouplingAIndicesFromHost(const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the a argument for the ChainCoupling factor from
   * device.
   */
  void SetChainCouplingAIndicesFromDevice(const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the b argument for the ChainCoupling factor from host.
   */
  void SetChainCouplingBIndicesFromHost(const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the b argument for the ChainCoupling factor from
   * device.
   */
  void SetChainCouplingBIndicesFromDevice(const unsigned int* const indices, size_t num);

  /**
   * Set the values for the weight consts ChainCoupling factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetChainCouplingWeightDataFromStackedHost(const double* const data, size_t offset,
                                                 size_t num);

  /**
   * Set the values for the weight consts ChainCoupling factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetChainCouplingWeightDataFromStackedDevice(const double* const data, size_t offset,
                                                   size_t num);

  /**
   * Set the values for the offset consts ChainCoupling factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetChainCouplingOffsetDataFromStackedHost(const double* const data, size_t offset,
                                                 size_t num);

  /**
   * Set the values for the offset consts ChainCoupling factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetChainCouplingOffsetDataFromStackedDevice(const double* const data, size_t offset,
                                                   size_t num);

  /**
   * Set the current number of ChainCoupling factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetChainCouplingNum(size_t num);

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

  size_t ChainA_num_;
  size_t ChainA_num_max_;
  size_t ChainB_num_;
  size_t ChainB_num_max_;
  size_t chain_a_link_num_;
  size_t chain_a_link_num_max_;
  size_t chain_a_prior_num_;
  size_t chain_a_prior_num_max_;
  size_t chain_b_link_num_;
  size_t chain_b_link_num_max_;
  size_t chain_b_prior_num_;
  size_t chain_b_prior_num_max_;
  size_t chain_coupling_num_;
  size_t chain_coupling_num_max_;

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
  double* nodes__ChainA__storage_current_;
  double* nodes__ChainA__storage_check_;
  double* nodes__ChainA__storage_new_best_;
  double* nodes__ChainB__storage_current_;
  double* nodes__ChainB__storage_check_;
  double* nodes__ChainB__storage_new_best_;
  SharedIndex* facs__chain_a_link__args__a__idx_shared_;
  double* facs__chain_a_link__args__weight__data_;
  double* facs__chain_a_link__args__offset__data_;
  SharedIndex* facs__chain_a_prior__args__a__idx_shared_;
  double* facs__chain_a_prior__args__target__data_;
  SharedIndex* facs__chain_b_link__args__b__idx_shared_;
  double* facs__chain_b_link__args__weight__data_;
  double* facs__chain_b_link__args__offset__data_;
  SharedIndex* facs__chain_b_prior__args__b__idx_shared_;
  double* facs__chain_b_prior__args__target__data_;
  SharedIndex* facs__chain_coupling__args__a__idx_shared_;
  SharedIndex* facs__chain_coupling__args__b__idx_shared_;
  double* facs__chain_coupling__args__weight__data_;
  double* facs__chain_coupling__args__offset__data_;
  double* marker__scratch_inout_;
  double* facs__chain_a_link__res_;
  double* facs__chain_a_prior__res_;
  double* facs__chain_b_link__res_;
  double* facs__chain_b_prior__res_;
  double* facs__chain_coupling__res_;
  double* facs__chain_a_link__args__a__jac_first_;
  double* facs__chain_a_link__args__a__jac_second_;
  double* nodes__ChainA__tridig_U_;
  double* facs__chain_a_prior__args__a__jac_;
  double* facs__chain_b_link__args__b__jac_first_;
  double* facs__chain_b_link__args__b__jac_second_;
  double* nodes__ChainB__tridig_U_;
  double* facs__chain_b_prior__args__b__jac_;
  double* facs__chain_coupling__args__a__jac_;
  double* facs__chain_coupling__args__b__jac_;
  double* nodes__ChainA__z_;
  double* nodes__ChainA__z_end__;
  double* nodes__ChainB__z_;
  double* nodes__ChainB__z_end__;
  double* nodes__ChainA__p_;
  double* nodes__ChainA__p_end__;
  double* nodes__ChainB__p_;
  double* nodes__ChainB__p_end__;
  double* nodes__ChainA__step_;
  double* nodes__ChainA__step_end__;
  double* nodes__ChainB__step_;
  double* nodes__ChainB__step_end__;
  double* marker__w_start_;
  double* nodes__ChainA__w_;
  double* nodes__ChainB__w_;
  double* marker__w_end_;
  double* marker__r_0_start_;
  double* nodes__ChainA__r_0_;
  double* nodes__ChainB__r_0_;
  double* marker__r_0_end_;
  double* marker__r_k_start_;
  double* nodes__ChainA__r_k_;
  double* nodes__ChainB__r_k_;
  double* marker__r_k_end_;
  double* marker__Mp_start_;
  double* nodes__ChainA__Mp_;
  double* nodes__ChainB__Mp_;
  double* marker__Mp_end_;
  double* marker__precond_start_;
  double* nodes__ChainA__precond_diag_;
  double* nodes__ChainA__precond_tril_;
  double* nodes__ChainB__precond_diag_;
  double* nodes__ChainB__precond_tril_;
  double* marker__precond_end_;
  double* marker__jp_start_;
  double* facs__chain_a_link__jp_;
  double* facs__chain_a_prior__jp_;
  double* facs__chain_b_link__jp_;
  double* facs__chain_b_prior__jp_;
  double* facs__chain_coupling__jp_;
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
  double* nodes__ChainA__tridig_eq_;
  double* nodes__ChainB__tridig_eq_;
};

}  // namespace caspar