#include "solver.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

#include "caspar_mappings.h"
#include "kernel_ChainA_alpha_denominator_or_beta_numerator.h"
#include "kernel_ChainA_alpha_numerator_denominator.h"
#include "kernel_ChainA_cyclic_reduction_down.h"
#include "kernel_ChainA_cyclic_reduction_solve_two.h"
#include "kernel_ChainA_cyclic_reduction_up.h"
#include "kernel_ChainA_normalize.h"
#include "kernel_ChainA_pred_decrease_times_two.h"
#include "kernel_ChainA_retract.h"
#include "kernel_ChainA_start_w.h"
#include "kernel_ChainA_start_w_contribute.h"
#include "kernel_ChainA_tridig_make_eq.h"
#include "kernel_ChainA_update_Mp.h"
#include "kernel_ChainA_update_p.h"
#include "kernel_ChainA_update_r.h"
#include "kernel_ChainA_update_r_first.h"
#include "kernel_ChainA_update_step.h"
#include "kernel_ChainA_update_step_first.h"
#include "kernel_ChainB_alpha_denominator_or_beta_numerator.h"
#include "kernel_ChainB_alpha_numerator_denominator.h"
#include "kernel_ChainB_cyclic_reduction_down.h"
#include "kernel_ChainB_cyclic_reduction_solve_two.h"
#include "kernel_ChainB_cyclic_reduction_up.h"
#include "kernel_ChainB_normalize.h"
#include "kernel_ChainB_pred_decrease_times_two.h"
#include "kernel_ChainB_retract.h"
#include "kernel_ChainB_start_w.h"
#include "kernel_ChainB_start_w_contribute.h"
#include "kernel_ChainB_tridig_make_eq.h"
#include "kernel_ChainB_update_Mp.h"
#include "kernel_ChainB_update_p.h"
#include "kernel_ChainB_update_r.h"
#include "kernel_ChainB_update_r_first.h"
#include "kernel_ChainB_update_step.h"
#include "kernel_ChainB_update_step_first.h"
#include "kernel_chain_a_link_jtjnjtr_direct.h"
#include "kernel_chain_a_link_res_jac.h"
#include "kernel_chain_a_link_res_jac_first.h"
#include "kernel_chain_a_link_score.h"
#include "kernel_chain_a_prior_jtjnjtr_direct.h"
#include "kernel_chain_a_prior_res_jac.h"
#include "kernel_chain_a_prior_res_jac_first.h"
#include "kernel_chain_a_prior_score.h"
#include "kernel_chain_b_link_jtjnjtr_direct.h"
#include "kernel_chain_b_link_res_jac.h"
#include "kernel_chain_b_link_res_jac_first.h"
#include "kernel_chain_b_link_score.h"
#include "kernel_chain_b_prior_jtjnjtr_direct.h"
#include "kernel_chain_b_prior_res_jac.h"
#include "kernel_chain_b_prior_res_jac_first.h"
#include "kernel_chain_b_prior_score.h"
#include "kernel_chain_coupling_jtjnjtr_direct.h"
#include "kernel_chain_coupling_res_jac.h"
#include "kernel_chain_coupling_res_jac_first.h"
#include "kernel_chain_coupling_score.h"
#include "shared_indices.h"
#include "solver_tools.h"
#include "sort_indices.h"

namespace {

void make_aligned(size_t& offset, size_t alignment_bytes) {
  offset = ((offset + alignment_bytes - 1) / alignment_bytes) * alignment_bytes;
}

template <typename T>
void increment_offset(size_t& offset, size_t num_elements, size_t alignment_elements) {
  make_aligned(offset, alignment_elements * sizeof(T));
  offset += num_elements * sizeof(T);
}

template <typename T>
T* assign_and_increment(uint8_t* origin_ptr, size_t& offset, size_t num_elements,
                        size_t alignment_elements) {
  make_aligned(offset, alignment_elements * sizeof(T));
  size_t old_offset = offset;
  offset += num_elements * sizeof(T);
  return reinterpret_cast<T*>(origin_ptr + old_offset);
}

}  // namespace

namespace caspar {

GraphSolver::GraphSolver(const SolverParams<double>& params, size_t ChainA_num_max,
                         size_t ChainB_num_max, size_t chain_a_link_num_max,
                         size_t chain_a_prior_num_max, size_t chain_b_link_num_max,
                         size_t chain_b_prior_num_max, size_t chain_coupling_num_max, int device_id)
    : params_(params),
      device_id_(device_id),
      ChainA_num_(ChainA_num_max),
      ChainA_num_max_(ChainA_num_max),
      ChainB_num_(ChainB_num_max),
      ChainB_num_max_(ChainB_num_max),
      chain_a_link_num_(chain_a_link_num_max),
      chain_a_link_num_max_(chain_a_link_num_max),
      chain_a_prior_num_(chain_a_prior_num_max),
      chain_a_prior_num_max_(chain_a_prior_num_max),
      chain_b_link_num_(chain_b_link_num_max),
      chain_b_link_num_max_(chain_b_link_num_max),
      chain_b_prior_num_(chain_b_prior_num_max),
      chain_b_prior_num_max_(chain_b_prior_num_max),
      chain_coupling_num_(chain_coupling_num_max),
      chain_coupling_num_max_(chain_coupling_num_max) {
  indices_valid_ = false;
  if (params.pcg_rel_error_exit <= 0.0f) {
    throw std::runtime_error("params.pcg_rel_error_exit must be positive");
  }
  if (params.diag_init < 0.0f) {
    throw std::runtime_error("params.diag_init must be positive");
  }
  if (params.pcg_iter_max < 1) {
    throw std::runtime_error("params.pcg_iter_max must be at least 1");
  }
  allocation_size_ = get_nbytes();

  if (device_id_ < 0) {
    throw std::runtime_error("Invalid CUDA device id: " + std::to_string(device_id_));
  }
  if (device_id_ != 0) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount <= device_id_) {
      throw std::runtime_error("CUDA detected " + std::to_string(deviceCount) +
                               " devices, but device " + std::to_string(device_id_) +
                               " was requested (0-indexed)");
    }
  }
  cudaSetDevice(device_id_);
  cudaMalloc(&origin_ptr_, allocation_size_);

  size_t offset = 0;
  marker__start_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__ChainA__storage_current_ =
      assign_and_increment<double>(origin_ptr_, offset, 2 * ChainA_num_, 4);
  nodes__ChainA__storage_check_ =
      assign_and_increment<double>(origin_ptr_, offset, 2 * ChainA_num_, 4);
  nodes__ChainA__storage_new_best_ =
      assign_and_increment<double>(origin_ptr_, offset, 2 * ChainA_num_, 4);
  nodes__ChainB__storage_current_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * ChainB_num_, 4);
  nodes__ChainB__storage_check_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * ChainB_num_, 4);
  nodes__ChainB__storage_new_best_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * ChainB_num_, 4);
  facs__chain_a_link__args__a__idx_shared_ =
      assign_and_increment<SharedIndex>(origin_ptr_, offset, 1 * chain_a_link_num_, 4);
  facs__chain_a_link__args__weight__data_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * chain_a_link_num_, 4);
  facs__chain_a_link__args__offset__data_ =
      assign_and_increment<double>(origin_ptr_, offset, 2 * chain_a_link_num_, 4);
  facs__chain_a_prior__args__a__idx_shared_ =
      assign_and_increment<SharedIndex>(origin_ptr_, offset, 1 * chain_a_prior_num_, 4);
  facs__chain_a_prior__args__target__data_ =
      assign_and_increment<double>(origin_ptr_, offset, 2 * chain_a_prior_num_, 4);
  facs__chain_b_link__args__b__idx_shared_ =
      assign_and_increment<SharedIndex>(origin_ptr_, offset, 1 * chain_b_link_num_, 4);
  facs__chain_b_link__args__weight__data_ =
      assign_and_increment<double>(origin_ptr_, offset, 9 * chain_b_link_num_, 4);
  facs__chain_b_link__args__offset__data_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * chain_b_link_num_, 4);
  facs__chain_b_prior__args__b__idx_shared_ =
      assign_and_increment<SharedIndex>(origin_ptr_, offset, 1 * chain_b_prior_num_, 4);
  facs__chain_b_prior__args__target__data_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * chain_b_prior_num_, 4);
  facs__chain_coupling__args__a__idx_shared_ =
      assign_and_increment<SharedIndex>(origin_ptr_, offset, 1 * chain_coupling_num_, 4);
  facs__chain_coupling__args__b__idx_shared_ =
      assign_and_increment<SharedIndex>(origin_ptr_, offset, 1 * chain_coupling_num_, 4);
  facs__chain_coupling__args__weight__data_ =
      assign_and_increment<double>(origin_ptr_, offset, 6 * chain_coupling_num_, 4);
  facs__chain_coupling__args__offset__data_ =
      assign_and_increment<double>(origin_ptr_, offset, 2 * chain_coupling_num_, 4);
  marker__scratch_inout_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  facs__chain_a_link__res_ =
      assign_and_increment<double>(origin_ptr_, offset, 2 * chain_a_link_num_, 4);
  facs__chain_a_prior__res_ =
      assign_and_increment<double>(origin_ptr_, offset, 2 * chain_a_prior_num_, 4);
  facs__chain_b_link__res_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * chain_b_link_num_, 4);
  facs__chain_b_prior__res_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * chain_b_prior_num_, 4);
  facs__chain_coupling__res_ =
      assign_and_increment<double>(origin_ptr_, offset, 2 * chain_coupling_num_, 4);
  facs__chain_a_link__args__a__jac_first_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * chain_a_link_num_, 4);
  facs__chain_a_link__args__a__jac_second_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * chain_a_link_num_, 4);
  nodes__ChainA__tridig_U_ = assign_and_increment<double>(origin_ptr_, offset, 4 * ChainA_num_, 4);
  facs__chain_a_prior__args__a__jac_ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * chain_a_prior_num_, 4);
  facs__chain_b_link__args__b__jac_first_ =
      assign_and_increment<double>(origin_ptr_, offset, 9 * chain_b_link_num_, 4);
  facs__chain_b_link__args__b__jac_second_ =
      assign_and_increment<double>(origin_ptr_, offset, 9 * chain_b_link_num_, 4);
  nodes__ChainB__tridig_U_ = assign_and_increment<double>(origin_ptr_, offset, 9 * ChainB_num_, 4);
  facs__chain_b_prior__args__b__jac_ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * chain_b_prior_num_, 4);
  facs__chain_coupling__args__a__jac_ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * chain_coupling_num_, 4);
  facs__chain_coupling__args__b__jac_ =
      assign_and_increment<double>(origin_ptr_, offset, 6 * chain_coupling_num_, 4);
  nodes__ChainA__z_ = assign_and_increment<double>(origin_ptr_, offset, 2 * ChainA_num_, 4);
  nodes__ChainA__z_end__ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__ChainB__z_ = assign_and_increment<double>(origin_ptr_, offset, 4 * ChainB_num_, 4);
  nodes__ChainB__z_end__ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__ChainA__p_ = assign_and_increment<double>(origin_ptr_, offset, 2 * ChainA_num_, 4);
  nodes__ChainA__p_end__ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__ChainB__p_ = assign_and_increment<double>(origin_ptr_, offset, 4 * ChainB_num_, 4);
  nodes__ChainB__p_end__ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__ChainA__step_ = assign_and_increment<double>(origin_ptr_, offset, 2 * ChainA_num_, 4);
  nodes__ChainA__step_end__ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__ChainB__step_ = assign_and_increment<double>(origin_ptr_, offset, 4 * ChainB_num_, 4);
  nodes__ChainB__step_end__ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  marker__w_start_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__ChainA__w_ = assign_and_increment<double>(origin_ptr_, offset, 2 * ChainA_num_, 4);
  nodes__ChainB__w_ = assign_and_increment<double>(origin_ptr_, offset, 4 * ChainB_num_, 4);
  marker__w_end_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 1);
  marker__r_0_start_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__ChainA__r_0_ = assign_and_increment<double>(origin_ptr_, offset, 2 * ChainA_num_, 4);
  nodes__ChainB__r_0_ = assign_and_increment<double>(origin_ptr_, offset, 4 * ChainB_num_, 4);
  marker__r_0_end_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  marker__r_k_start_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__ChainA__r_k_ = assign_and_increment<double>(origin_ptr_, offset, 2 * ChainA_num_, 4);
  nodes__ChainB__r_k_ = assign_and_increment<double>(origin_ptr_, offset, 4 * ChainB_num_, 4);
  marker__r_k_end_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  marker__Mp_start_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__ChainA__Mp_ = assign_and_increment<double>(origin_ptr_, offset, 2 * ChainA_num_, 4);
  nodes__ChainB__Mp_ = assign_and_increment<double>(origin_ptr_, offset, 4 * ChainB_num_, 4);
  marker__Mp_end_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  marker__precond_start_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__ChainA__precond_diag_ =
      assign_and_increment<double>(origin_ptr_, offset, 2 * ChainA_num_, 4);
  nodes__ChainA__precond_tril_ =
      assign_and_increment<double>(origin_ptr_, offset, 1 * ChainA_num_, 4);
  nodes__ChainB__precond_diag_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * ChainB_num_, 4);
  nodes__ChainB__precond_tril_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * ChainB_num_, 4);
  marker__precond_end_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 1);
  marker__jp_start_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  facs__chain_a_link__jp_ =
      assign_and_increment<double>(origin_ptr_, offset, 2 * chain_a_link_num_, 4);
  facs__chain_a_prior__jp_ =
      assign_and_increment<double>(origin_ptr_, offset, 2 * chain_a_prior_num_, 4);
  facs__chain_b_link__jp_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * chain_b_link_num_, 4);
  facs__chain_b_prior__jp_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * chain_b_prior_num_, 4);
  facs__chain_coupling__jp_ =
      assign_and_increment<double>(origin_ptr_, offset, 2 * chain_coupling_num_, 4);
  marker__jp_end_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 1);
  solver__current_diag_ = assign_and_increment<double>(origin_ptr_, offset, 1 * 1, 1);
  solver__alpha_numerator_ = assign_and_increment<double>(origin_ptr_, offset, 1 * 1, 1);
  solver__alpha_denominator_ = assign_and_increment<double>(origin_ptr_, offset, 1 * 1, 1);
  solver__alpha_ = assign_and_increment<double>(origin_ptr_, offset, 1 * 1, 1);
  solver__neg_alpha_ = assign_and_increment<double>(origin_ptr_, offset, 1 * 1, 1);
  solver__beta_numerator_ = assign_and_increment<double>(origin_ptr_, offset, 1 * 1, 1);
  solver__beta_ = assign_and_increment<double>(origin_ptr_, offset, 1 * 1, 1);
  solver__r_0_norm2_tot_ = assign_and_increment<double>(origin_ptr_, offset, 1 * 1, 1);
  solver__r_kp1_norm2_tot_ = assign_and_increment<double>(origin_ptr_, offset, 1 * 1, 1);
  solver__pred_decrease_tot_ = assign_and_increment<double>(origin_ptr_, offset, 1 * 1, 1);
  solver__res_tot_ = assign_and_increment<double>(origin_ptr_, offset, 1 * 1, 1);
  nodes__ChainA__tridig_eq_ =
      assign_and_increment<double>(origin_ptr_, offset, 20 * ChainA_num_, 4);
  nodes__ChainB__tridig_eq_ =
      assign_and_increment<double>(origin_ptr_, offset, 42 * ChainB_num_, 4);

  scratch_inout_size_ = offset;  // sorting, sum,
}

GraphSolver::~GraphSolver() {
  cudaSetDevice(device_id_);
  cudaFree(origin_ptr_);
}

void GraphSolver::set_params(const SolverParams<double>& params) {
  this->params_ = params;
}

size_t GraphSolver::get_allocation_size() {
  return allocation_size_;
}

SolveResult GraphSolver::solve(bool print_progress, bool verbose_logging) {
  cudaSetDevice(device_id_);
  SolveResult result;
  result.exit_reason = ExitReason::MAX_ITERATIONS;
  double score_best;
  double score_best_pcg;
  double diag = params_.diag_init;
  cudaMemcpy(solver__current_diag_, &diag, sizeof(double), cudaMemcpyHostToDevice);

  double up_scale = params_.diag_scaling_up;
  double quality;

  std::chrono::time_point<std::chrono::steady_clock> t0 = std::chrono::steady_clock::now();
  std::chrono::time_point<std::chrono::steady_clock> t_prev = t0;
  score_best = DoResJacFirst();
  result.initial_score = score_best;
  if (print_progress) {
    printf("                                 score_init: % .6e\n", score_best);
  }

  for (solver_iter_ = 0; solver_iter_ < params_.solver_iter_max; solver_iter_++) {
    if (solver_iter_ != 0) {
      DoResJac();
    }
    score_best_pcg = score_best;
    // The inner loop can break before storing a proposal in storage_new_best_.
    bool have_pcg_proposal = false;
    for (pcg_iter_ = 0; pcg_iter_ < params_.pcg_iter_max; pcg_iter_++) {
      DoNormalize();

      if (pcg_iter_ == 0) {
        Copy(marker__r_k_start_, marker__r_k_end_, marker__w_start_);
        DoJtjpDirect();
        DoAlphaFirst();
        DoUpdateStepFirst();
        DoUpdateRFirst();
      } else {
        DoBeta();
        DoUpdateP();
        DoUpdateMp();
        DoJtjpDirect();
        DoAlpha();
        DoUpdateStep();
        DoUpdateR();
      }
      if (params_.pcg_rel_decrease_min != -1.0f || params_.pcg_rel_score_exit != -1.0f) {
        double score_new_pcg = DoRetractScore();
        if (!(score_new_pcg <= score_best_pcg * params_.pcg_rel_decrease_min)) {
          break;
        }
        std::swap(nodes__ChainA__storage_check_, nodes__ChainA__storage_new_best_);
        std::swap(nodes__ChainB__storage_check_, nodes__ChainB__storage_new_best_);
        score_best_pcg = score_new_pcg;
        have_pcg_proposal = true;
        if (params_.pcg_rel_score_exit != -1.0f &&
            score_best_pcg < score_best * params_.pcg_rel_score_exit) {
          break;
        }
      }
      if (pcg_r_kp1_norm2_ < pcg_r_0_norm2_ * params_.pcg_rel_error_exit) {
        break;
      }
    }
    pcg_iter_ = std::min(pcg_iter_, params_.pcg_iter_max - 1);

    if (params_.pcg_rel_decrease_min == -1.0f && params_.pcg_rel_score_exit == -1.0f) {
      score_best_pcg = DoRetractScore();
      std::swap(nodes__ChainA__storage_check_, nodes__ChainA__storage_new_best_);
      std::swap(nodes__ChainB__storage_check_, nodes__ChainB__storage_new_best_);
      have_pcg_proposal = true;
    }

    const double diag_current = diag;
    bool step_accepted = false;
    if (have_pcg_proposal && score_best_pcg < score_best * params_.solver_rel_decrease_min) {
      quality = (score_best - score_best_pcg) / GetPredDecrease();
      const double quality_tmp = 2 * quality - 1;
      double scale =
          std::max(params_.diag_scaling_down, 1.0f - quality_tmp * quality_tmp * quality_tmp);
      diag = std::max(params_.diag_min, diag * scale);
      cudaMemcpy(solver__current_diag_, &diag, sizeof(double), cudaMemcpyHostToDevice);
      up_scale = params_.diag_scaling_up;
      score_best = score_best_pcg;
      step_accepted = true;
      std::swap(nodes__ChainA__storage_current_, nodes__ChainA__storage_new_best_);
      std::swap(nodes__ChainB__storage_current_, nodes__ChainB__storage_new_best_);

    } else {
      quality = 0.0f;
      diag = diag * up_scale;
      if (diag > params_.diag_exit_value) {
        result.exit_reason = ExitReason::CONVERGED_DIAG_EXIT;
        break;
      }
      cudaMemcpy(solver__current_diag_, &diag, sizeof(double), cudaMemcpyHostToDevice);
      up_scale *= 2;
    }
    const auto t_now = std::chrono::steady_clock::now();
    const double dt_inc = std::chrono::duration<double>(t_now - t_prev).count();
    const double dt_tot = std::chrono::duration<double>(t_now - t0).count();

    if (verbose_logging) {
      IterationData iter_data;
      iter_data.solver_iter = solver_iter_;
      iter_data.pcg_iter = pcg_iter_;
      iter_data.score_current = score_best_pcg;
      iter_data.score_best = score_best;
      iter_data.step_quality = quality;
      iter_data.diag = diag_current;
      iter_data.dt_inc = dt_inc;
      iter_data.dt_tot = dt_tot;
      iter_data.step_accepted = step_accepted;
      result.iterations.push_back(iter_data);
    }

    if (print_progress) {
      printf("solver_iter: % 3d  ", solver_iter_);
      printf("pcg_iter: % 3d  ", pcg_iter_);
      printf("score_current: % 13.6e  ", score_best_pcg);
      printf("score_best: % 13.6e  ", score_best);
      printf("step_quality: % 7.3f  ", quality);
      printf("diag: % 6.3e  ", diag_current);
      printf("dt_inc: % 10.6f  ", dt_inc);
      printf("dt_tot: % 10.6f  ", dt_tot);
      t_prev = t_now;
      printf("\n");
    }
    if (score_best <= params_.score_exit_value) {
      result.exit_reason = ExitReason::CONVERGED_SCORE_THRESHOLD;
      break;
    }
  }

  const auto t_final = std::chrono::steady_clock::now();
  result.final_score = score_best;
  result.iteration_count = solver_iter_;
  result.runtime = std::chrono::duration<double>(t_final - t0).count();
  return result;
}

double GraphSolver::DoResJacFirst() {
  Zero(solver__res_tot_, solver__res_tot_ + 1);
  Zero(marker__r_0_start_, marker__precond_end_);

  ChainALinkResJacFirst(
      nodes__ChainA__storage_current_, ChainA_num_max_, facs__chain_a_link__args__weight__data_,
      chain_a_link_num_max_, facs__chain_a_link__args__offset__data_, chain_a_link_num_max_,

      facs__chain_a_link__res_, chain_a_link_num_, solver__res_tot_, nodes__ChainA__tridig_U_,
      ChainA_num_ - 1, nodes__ChainA__r_k_, ChainA_num_, nodes__ChainA__precond_diag_, ChainA_num_,
      nodes__ChainA__precond_tril_, ChainA_num_, chain_a_link_num_);

  ChainAPriorResJacFirst(
      nodes__ChainA__storage_current_, ChainA_num_max_, facs__chain_a_prior__args__a__idx_shared_,
      facs__chain_a_prior__args__target__data_, chain_a_prior_num_max_,

      facs__chain_a_prior__res_, chain_a_prior_num_, solver__res_tot_, nodes__ChainA__r_k_,
      ChainA_num_, nodes__ChainA__precond_diag_, ChainA_num_, nodes__ChainA__precond_tril_,
      ChainA_num_, chain_a_prior_num_);

  ChainBLinkResJacFirst(
      nodes__ChainB__storage_current_, ChainB_num_max_, facs__chain_b_link__args__weight__data_,
      chain_b_link_num_max_, facs__chain_b_link__args__offset__data_, chain_b_link_num_max_,

      facs__chain_b_link__res_, chain_b_link_num_, solver__res_tot_, nodes__ChainB__tridig_U_,
      ChainB_num_ - 1, nodes__ChainB__r_k_, ChainB_num_, nodes__ChainB__precond_diag_, ChainB_num_,
      nodes__ChainB__precond_tril_, ChainB_num_, chain_b_link_num_);

  ChainBPriorResJacFirst(
      nodes__ChainB__storage_current_, ChainB_num_max_, facs__chain_b_prior__args__b__idx_shared_,
      facs__chain_b_prior__args__target__data_, chain_b_prior_num_max_,

      facs__chain_b_prior__res_, chain_b_prior_num_, solver__res_tot_, nodes__ChainB__r_k_,
      ChainB_num_, nodes__ChainB__precond_diag_, ChainB_num_, nodes__ChainB__precond_tril_,
      ChainB_num_, chain_b_prior_num_);

  ChainCouplingResJacFirst(
      nodes__ChainA__storage_current_, ChainA_num_max_, facs__chain_coupling__args__a__idx_shared_,
      nodes__ChainB__storage_current_, ChainB_num_max_, facs__chain_coupling__args__b__idx_shared_,
      facs__chain_coupling__args__weight__data_, chain_coupling_num_max_,
      facs__chain_coupling__args__offset__data_, chain_coupling_num_max_,

      facs__chain_coupling__res_, chain_coupling_num_, solver__res_tot_,
      facs__chain_coupling__args__a__jac_, chain_coupling_num_, nodes__ChainA__r_k_, ChainA_num_,
      nodes__ChainA__precond_diag_, ChainA_num_, nodes__ChainA__precond_tril_, ChainA_num_,
      facs__chain_coupling__args__b__jac_, chain_coupling_num_, nodes__ChainB__r_k_, ChainB_num_,
      nodes__ChainB__precond_diag_, ChainB_num_, nodes__ChainB__precond_tril_, ChainB_num_,
      chain_coupling_num_);
  Copy(marker__r_k_start_, marker__r_k_end_, marker__r_0_start_);
  Copy(marker__r_k_start_, marker__r_k_end_, marker__Mp_start_);
  return 0.5 * ReadCuMem(solver__res_tot_);
}
void GraphSolver::DoResJac() {
  Zero(solver__res_tot_, solver__res_tot_ + 1);
  Zero(marker__r_0_start_, marker__precond_end_);

  ChainALinkResJac(nodes__ChainA__storage_current_, ChainA_num_max_,
                   facs__chain_a_link__args__weight__data_, chain_a_link_num_max_,
                   facs__chain_a_link__args__offset__data_, chain_a_link_num_max_,

                   facs__chain_a_link__res_, chain_a_link_num_,

                   nodes__ChainA__tridig_U_, ChainA_num_ - 1, nodes__ChainA__r_k_, ChainA_num_,
                   nodes__ChainA__precond_diag_, ChainA_num_, nodes__ChainA__precond_tril_,
                   ChainA_num_, chain_a_link_num_);

  ChainAPriorResJac(nodes__ChainA__storage_current_, ChainA_num_max_,
                    facs__chain_a_prior__args__a__idx_shared_,
                    facs__chain_a_prior__args__target__data_, chain_a_prior_num_max_,

                    facs__chain_a_prior__res_, chain_a_prior_num_,

                    nodes__ChainA__r_k_, ChainA_num_, nodes__ChainA__precond_diag_, ChainA_num_,
                    nodes__ChainA__precond_tril_, ChainA_num_, chain_a_prior_num_);

  ChainBLinkResJac(nodes__ChainB__storage_current_, ChainB_num_max_,
                   facs__chain_b_link__args__weight__data_, chain_b_link_num_max_,
                   facs__chain_b_link__args__offset__data_, chain_b_link_num_max_,

                   facs__chain_b_link__res_, chain_b_link_num_,

                   nodes__ChainB__tridig_U_, ChainB_num_ - 1, nodes__ChainB__r_k_, ChainB_num_,
                   nodes__ChainB__precond_diag_, ChainB_num_, nodes__ChainB__precond_tril_,
                   ChainB_num_, chain_b_link_num_);

  ChainBPriorResJac(nodes__ChainB__storage_current_, ChainB_num_max_,
                    facs__chain_b_prior__args__b__idx_shared_,
                    facs__chain_b_prior__args__target__data_, chain_b_prior_num_max_,

                    facs__chain_b_prior__res_, chain_b_prior_num_,

                    nodes__ChainB__r_k_, ChainB_num_, nodes__ChainB__precond_diag_, ChainB_num_,
                    nodes__ChainB__precond_tril_, ChainB_num_, chain_b_prior_num_);

  ChainCouplingResJac(
      nodes__ChainA__storage_current_, ChainA_num_max_, facs__chain_coupling__args__a__idx_shared_,
      nodes__ChainB__storage_current_, ChainB_num_max_, facs__chain_coupling__args__b__idx_shared_,
      facs__chain_coupling__args__weight__data_, chain_coupling_num_max_,
      facs__chain_coupling__args__offset__data_, chain_coupling_num_max_,

      facs__chain_coupling__res_, chain_coupling_num_,

      facs__chain_coupling__args__a__jac_, chain_coupling_num_, nodes__ChainA__r_k_, ChainA_num_,
      nodes__ChainA__precond_diag_, ChainA_num_, nodes__ChainA__precond_tril_, ChainA_num_,
      facs__chain_coupling__args__b__jac_, chain_coupling_num_, nodes__ChainB__r_k_, ChainB_num_,
      nodes__ChainB__precond_diag_, ChainB_num_, nodes__ChainB__precond_tril_, ChainB_num_,
      chain_coupling_num_);
  Copy(marker__r_k_start_, marker__r_k_end_, marker__r_0_start_);
  Copy(marker__r_k_start_, marker__r_k_end_, marker__Mp_start_);
}

void GraphSolver::DoNormalize() {
  double* z;
  z = pcg_iter_ == 0 ? nodes__ChainA__p_ : nodes__ChainA__z_;
  if (ChainA_num_ < 2) {
    // Without a second node there are no off-diagonal blocks, and the cyclic
    // reduction would write a second node's solution.
    ChainANormalize(nodes__ChainA__precond_diag_, ChainA_num_, nodes__ChainA__precond_tril_,
                    ChainA_num_, nodes__ChainA__r_k_, ChainA_num_, solver__current_diag_, z,
                    ChainA_num_, ChainA_num_);
  } else {
    Zero(z, pcg_iter_ == 0 ? nodes__ChainA__p_end__ : nodes__ChainA__z_end__);
    ChainATridigMakeEq(nodes__ChainA__precond_diag_, ChainA_num_, nodes__ChainA__precond_tril_,
                       ChainA_num_, nodes__ChainA__tridig_U_, ChainA_num_ - 1, -1, 1,
                       nodes__ChainA__r_k_, ChainA_num_, solver__current_diag_,
                       nodes__ChainA__tridig_eq_, ChainA_num_, ChainA_num_);
    int i;
    int offset = 0;
    int remaining = ChainA_num_;
    for (i = 0; 2 << i < ChainA_num_; i++) {
      offset += (~remaining & 1) << i;
      remaining = (remaining + 1) >> 1;

      ChainACyclicReductionDown(nodes__ChainA__tridig_eq_, ChainA_num_, offset, (2 << i),
                                nodes__ChainA__tridig_eq_, ChainA_num_, offset - (1 << i), (2 << i),
                                nodes__ChainA__tridig_eq_, ChainA_num_, offset, (2 << i),
                                remaining);
    }
    i--;

    ChainACyclicReductionSolveTwo(nodes__ChainA__tridig_eq_, ChainA_num_, offset, (1 << (i + 1)), z,
                                  ChainA_num_, offset, 1, z, ChainA_num_, offset + (1 << (i + 1)),
                                  1, 1);

    for (; i >= 0; i--) {
      int is_odd = offset < 1 << i;
      int shift = (is_odd * 2 - 1) * (1 << i);

      ChainACyclicReductionUp(nodes__ChainA__tridig_eq_, ChainA_num_, offset + shift, (2 << i), z,
                              ChainA_num_, offset + shift - (1 << i), (2 << i), z, ChainA_num_,
                              offset + shift, (2 << i), remaining - is_odd);
      offset += shift * (!is_odd);
      remaining = remaining * 2 - is_odd;
    }
  }
  z = pcg_iter_ == 0 ? nodes__ChainB__p_ : nodes__ChainB__z_;
  if (ChainB_num_ < 2) {
    // Without a second node there are no off-diagonal blocks, and the cyclic
    // reduction would write a second node's solution.
    ChainBNormalize(nodes__ChainB__precond_diag_, ChainB_num_, nodes__ChainB__precond_tril_,
                    ChainB_num_, nodes__ChainB__r_k_, ChainB_num_, solver__current_diag_, z,
                    ChainB_num_, ChainB_num_);
  } else {
    Zero(z, pcg_iter_ == 0 ? nodes__ChainB__p_end__ : nodes__ChainB__z_end__);
    ChainBTridigMakeEq(nodes__ChainB__precond_diag_, ChainB_num_, nodes__ChainB__precond_tril_,
                       ChainB_num_, nodes__ChainB__tridig_U_, ChainB_num_ - 1, -1, 1,
                       nodes__ChainB__r_k_, ChainB_num_, solver__current_diag_,
                       nodes__ChainB__tridig_eq_, ChainB_num_, ChainB_num_);
    int i;
    int offset = 0;
    int remaining = ChainB_num_;
    for (i = 0; 2 << i < ChainB_num_; i++) {
      offset += (~remaining & 1) << i;
      remaining = (remaining + 1) >> 1;

      ChainBCyclicReductionDown(nodes__ChainB__tridig_eq_, ChainB_num_, offset, (2 << i),
                                nodes__ChainB__tridig_eq_, ChainB_num_, offset - (1 << i), (2 << i),
                                nodes__ChainB__tridig_eq_, ChainB_num_, offset, (2 << i),
                                remaining);
    }
    i--;

    ChainBCyclicReductionSolveTwo(nodes__ChainB__tridig_eq_, ChainB_num_, offset, (1 << (i + 1)), z,
                                  ChainB_num_, offset, 1, z, ChainB_num_, offset + (1 << (i + 1)),
                                  1, 1);

    for (; i >= 0; i--) {
      int is_odd = offset < 1 << i;
      int shift = (is_odd * 2 - 1) * (1 << i);

      ChainBCyclicReductionUp(nodes__ChainB__tridig_eq_, ChainB_num_, offset + shift, (2 << i), z,
                              ChainB_num_, offset + shift - (1 << i), (2 << i), z, ChainB_num_,
                              offset + shift, (2 << i), remaining - is_odd);
      offset += shift * (!is_odd);
      remaining = remaining * 2 - is_odd;
    }
  }
}

void GraphSolver::DoUpdateMp() {
  ChainAUpdateMp(nodes__ChainA__r_k_, ChainA_num_, nodes__ChainA__Mp_, ChainA_num_, solver__beta_,
                 nodes__ChainA__Mp_, ChainA_num_, nodes__ChainA__w_, ChainA_num_, ChainA_num_);
  ChainBUpdateMp(nodes__ChainB__r_k_, ChainB_num_, nodes__ChainB__Mp_, ChainB_num_, solver__beta_,
                 nodes__ChainB__Mp_, ChainB_num_, nodes__ChainB__w_, ChainB_num_, ChainB_num_);
}

void GraphSolver::DoJtjpDirect() {
  ChainCouplingJtjnjtrDirect(
      nodes__ChainA__p_, ChainA_num_, facs__chain_coupling__args__a__idx_shared_,
      facs__chain_coupling__args__a__jac_, chain_coupling_num_, nodes__ChainB__p_, ChainB_num_,
      facs__chain_coupling__args__b__idx_shared_, facs__chain_coupling__args__b__jac_,
      chain_coupling_num_, nodes__ChainA__w_, ChainA_num_, nodes__ChainB__w_, ChainB_num_,
      chain_coupling_num_);
}

void GraphSolver::DoAlphaFirst() {
  Zero(solver__alpha_numerator_, solver__alpha_denominator_ + 1);
  ChainAAlphaNumeratorDenominator(nodes__ChainA__p_, ChainA_num_, nodes__ChainA__r_k_, ChainA_num_,
                                  nodes__ChainA__w_, ChainA_num_, solver__alpha_numerator_,
                                  solver__alpha_denominator_, ChainA_num_);
  ChainBAlphaNumeratorDenominator(nodes__ChainB__p_, ChainB_num_, nodes__ChainB__r_k_, ChainB_num_,
                                  nodes__ChainB__w_, ChainB_num_, solver__alpha_numerator_,
                                  solver__alpha_denominator_, ChainB_num_);

  AlphaFromNumDenom(solver__alpha_numerator_, solver__alpha_denominator_, solver__alpha_,
                    solver__neg_alpha_);
}

void GraphSolver::DoAlpha() {
  Zero(solver__alpha_denominator_, solver__alpha_denominator_ + 1);
  ChainAAlphaDenominatorOrBetaNumerator(nodes__ChainA__p_, ChainA_num_, nodes__ChainA__w_,
                                        ChainA_num_, solver__alpha_denominator_, ChainA_num_);
  ChainBAlphaDenominatorOrBetaNumerator(nodes__ChainB__p_, ChainB_num_, nodes__ChainB__w_,
                                        ChainB_num_, solver__alpha_denominator_, ChainB_num_);

  AlphaFromNumDenom(solver__beta_numerator_, solver__alpha_denominator_, solver__alpha_,
                    solver__neg_alpha_);
}

void GraphSolver::DoUpdateStepFirst() {
  ChainAUpdateStepFirst(nodes__ChainA__p_, ChainA_num_, solver__alpha_, nodes__ChainA__step_,
                        ChainA_num_, ChainA_num_);
  ChainBUpdateStepFirst(nodes__ChainB__p_, ChainB_num_, solver__alpha_, nodes__ChainB__step_,
                        ChainB_num_, ChainB_num_);
}

void GraphSolver::DoUpdateStep() {
  ChainAUpdateStep(nodes__ChainA__step_, ChainA_num_, nodes__ChainA__p_, ChainA_num_,
                   solver__alpha_, nodes__ChainA__step_, ChainA_num_, ChainA_num_);
  ChainBUpdateStep(nodes__ChainB__step_, ChainB_num_, nodes__ChainB__p_, ChainB_num_,
                   solver__alpha_, nodes__ChainB__step_, ChainB_num_, ChainB_num_);
}

void GraphSolver::DoUpdateRFirst() {
  Zero(solver__r_0_norm2_tot_, solver__r_0_norm2_tot_ + 1);

  ChainAUpdateRFirst(nodes__ChainA__r_k_, ChainA_num_, nodes__ChainA__w_, ChainA_num_,
                     solver__neg_alpha_, nodes__ChainA__r_k_, ChainA_num_, solver__r_0_norm2_tot_,
                     solver__r_kp1_norm2_tot_, ChainA_num_);

  ChainBUpdateRFirst(nodes__ChainB__r_k_, ChainB_num_, nodes__ChainB__w_, ChainB_num_,
                     solver__neg_alpha_, nodes__ChainB__r_k_, ChainB_num_, solver__r_0_norm2_tot_,
                     solver__r_kp1_norm2_tot_, ChainB_num_);

  pcg_r_0_norm2_ = ReadCuMem(solver__r_0_norm2_tot_);
  pcg_r_kp1_norm2_ = ReadCuMem(solver__r_kp1_norm2_tot_);
}

void GraphSolver::DoUpdateR() {
  Zero(solver__r_kp1_norm2_tot_, solver__r_kp1_norm2_tot_ + 1);

  ChainAUpdateR(nodes__ChainA__r_k_, ChainA_num_, nodes__ChainA__w_, ChainA_num_,
                solver__neg_alpha_, nodes__ChainA__r_k_, ChainA_num_, solver__r_kp1_norm2_tot_,
                ChainA_num_);
  ChainBUpdateR(nodes__ChainB__r_k_, ChainB_num_, nodes__ChainB__w_, ChainB_num_,
                solver__neg_alpha_, nodes__ChainB__r_k_, ChainB_num_, solver__r_kp1_norm2_tot_,
                ChainB_num_);
  pcg_r_kp1_norm2_ = ReadCuMem(solver__r_kp1_norm2_tot_);
}

double GraphSolver::DoRetractScore() {
  ChainARetract(nodes__ChainA__storage_current_, ChainA_num_max_, nodes__ChainA__step_, ChainA_num_,
                nodes__ChainA__storage_check_, ChainA_num_max_, ChainA_num_);
  ChainBRetract(nodes__ChainB__storage_current_, ChainB_num_max_, nodes__ChainB__step_, ChainB_num_,
                nodes__ChainB__storage_check_, ChainB_num_max_, ChainB_num_);
  Zero(solver__res_tot_, solver__res_tot_ + 1);
  ChainALinkScore(nodes__ChainA__storage_check_, ChainA_num_max_,
                  facs__chain_a_link__args__weight__data_, chain_a_link_num_max_,
                  facs__chain_a_link__args__offset__data_, chain_a_link_num_max_, solver__res_tot_,
                  chain_a_link_num_);
  ChainAPriorScore(nodes__ChainA__storage_check_, ChainA_num_max_,
                   facs__chain_a_prior__args__a__idx_shared_,
                   facs__chain_a_prior__args__target__data_, chain_a_prior_num_max_,
                   solver__res_tot_, chain_a_prior_num_);
  ChainBLinkScore(nodes__ChainB__storage_check_, ChainB_num_max_,
                  facs__chain_b_link__args__weight__data_, chain_b_link_num_max_,
                  facs__chain_b_link__args__offset__data_, chain_b_link_num_max_, solver__res_tot_,
                  chain_b_link_num_);
  ChainBPriorScore(nodes__ChainB__storage_check_, ChainB_num_max_,
                   facs__chain_b_prior__args__b__idx_shared_,
                   facs__chain_b_prior__args__target__data_, chain_b_prior_num_max_,
                   solver__res_tot_, chain_b_prior_num_);
  ChainCouplingScore(nodes__ChainA__storage_check_, ChainA_num_max_,
                     facs__chain_coupling__args__a__idx_shared_, nodes__ChainB__storage_check_,
                     ChainB_num_max_, facs__chain_coupling__args__b__idx_shared_,
                     facs__chain_coupling__args__weight__data_, chain_coupling_num_max_,
                     facs__chain_coupling__args__offset__data_, chain_coupling_num_max_,
                     solver__res_tot_, chain_coupling_num_);
  return 0.5 * ReadCuMem(solver__res_tot_);
}

void GraphSolver::DoBeta() {
  Zero(solver__beta_numerator_, solver__beta_numerator_ + 1);

  ChainAAlphaDenominatorOrBetaNumerator(nodes__ChainA__r_k_, ChainA_num_, nodes__ChainA__z_,
                                        ChainA_num_, solver__beta_numerator_, ChainA_num_);

  ChainBAlphaDenominatorOrBetaNumerator(nodes__ChainB__r_k_, ChainB_num_, nodes__ChainB__z_,
                                        ChainB_num_, solver__beta_numerator_, ChainB_num_);
  BetaFromNumDenom(solver__beta_numerator_, solver__alpha_numerator_, solver__beta_);
}

void GraphSolver::DoUpdateP() {
  ChainAUpdateP(nodes__ChainA__z_, ChainA_num_, nodes__ChainA__p_, ChainA_num_, solver__beta_,
                nodes__ChainA__p_, ChainA_num_, ChainA_num_);
  ChainBUpdateP(nodes__ChainB__z_, ChainB_num_, nodes__ChainB__p_, ChainB_num_, solver__beta_,
                nodes__ChainB__p_, ChainB_num_, ChainB_num_);
}

double GraphSolver::GetPredDecrease() {
  Zero(solver__pred_decrease_tot_, solver__pred_decrease_tot_ + 1);
  ChainAPredDecreaseTimesTwo(nodes__ChainA__step_, ChainA_num_, nodes__ChainA__precond_diag_,
                             ChainA_num_, solver__current_diag_, nodes__ChainA__r_0_, ChainA_num_,
                             solver__pred_decrease_tot_, ChainA_num_);
  ChainBPredDecreaseTimesTwo(nodes__ChainB__step_, ChainB_num_, nodes__ChainB__precond_diag_,
                             ChainB_num_, solver__current_diag_, nodes__ChainB__r_0_, ChainB_num_,
                             solver__pred_decrease_tot_, ChainB_num_);
  return 0.5 * ReadCuMem(solver__pred_decrease_tot_);
}

void GraphSolver::finish_indices() {
  indices_valid_ = true;
}

void GraphSolver::SetChainANum(const size_t num) {
  cudaSetDevice(device_id_);
  if (num > ChainA_num_max_) {
    throw std::runtime_error(std::to_string(num) + " > ChainA_num_max_");
  }
  ChainA_num_ = num;
}

void GraphSolver::SetChainANodesFromStackedHost(const double* const data, const size_t offset,
                                                const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > ChainA_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > ChainA_num_");
  }
  cudaMemcpy(marker__scratch_inout_, data, 2 * num * sizeof(double), cudaMemcpyHostToDevice);
  ChainAStackedToCaspar(marker__scratch_inout_, nodes__ChainA__storage_current_, ChainA_num_max_,
                        offset, num);
}

void GraphSolver::SetChainANodesFromStackedDevice(const double* const data, const size_t offset,
                                                  const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > ChainA_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > ChainA_num_");
  }
  ChainAStackedToCaspar(data, nodes__ChainA__storage_current_, ChainA_num_max_, offset, num);
}

void GraphSolver::GetChainANodesToStackedHost(double* const data, const size_t offset,
                                              const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > ChainA_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > ChainA_num_");
  }
  ChainACasparToStacked(nodes__ChainA__storage_current_, marker__scratch_inout_, ChainA_num_max_,
                        offset, num);
  cudaMemcpy(data, marker__scratch_inout_, 2 * num * sizeof(double), cudaMemcpyDeviceToHost);
}

void GraphSolver::GetChainANodesToStackedDevice(double* const data, const size_t offset,
                                                const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > ChainA_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > ChainA_num_");
  }
  ChainACasparToStacked(nodes__ChainA__storage_current_, data, ChainA_num_max_, offset, num);
}

void GraphSolver::SetChainBNum(const size_t num) {
  cudaSetDevice(device_id_);
  if (num > ChainB_num_max_) {
    throw std::runtime_error(std::to_string(num) + " > ChainB_num_max_");
  }
  ChainB_num_ = num;
}

void GraphSolver::SetChainBNodesFromStackedHost(const double* const data, const size_t offset,
                                                const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > ChainB_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > ChainB_num_");
  }
  cudaMemcpy(marker__scratch_inout_, data, 3 * num * sizeof(double), cudaMemcpyHostToDevice);
  ChainBStackedToCaspar(marker__scratch_inout_, nodes__ChainB__storage_current_, ChainB_num_max_,
                        offset, num);
}

void GraphSolver::SetChainBNodesFromStackedDevice(const double* const data, const size_t offset,
                                                  const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > ChainB_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > ChainB_num_");
  }
  ChainBStackedToCaspar(data, nodes__ChainB__storage_current_, ChainB_num_max_, offset, num);
}

void GraphSolver::GetChainBNodesToStackedHost(double* const data, const size_t offset,
                                              const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > ChainB_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > ChainB_num_");
  }
  ChainBCasparToStacked(nodes__ChainB__storage_current_, marker__scratch_inout_, ChainB_num_max_,
                        offset, num);
  cudaMemcpy(data, marker__scratch_inout_, 3 * num * sizeof(double), cudaMemcpyDeviceToHost);
}

void GraphSolver::GetChainBNodesToStackedDevice(double* const data, const size_t offset,
                                                const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > ChainB_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > ChainB_num_");
  }
  ChainBCasparToStacked(nodes__ChainB__storage_current_, data, ChainB_num_max_, offset, num);
}

void GraphSolver::SetChainALinkNum(const size_t num) {
  if (num > chain_a_link_num_max_) {
    throw std::runtime_error(std::to_string(num) + " > chain_a_link_num_max_");
  }
  chain_a_link_num_ = num;
}
void GraphSolver::SetChainALinkWeightDataFromStackedHost(const double* const data, size_t offset,
                                                         size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > chain_a_link_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) + " > chain_a_link_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_, data, 4 * num * sizeof(double), cudaMemcpyHostToDevice);
  Matrix22StackedToCaspar(marker__scratch_inout_, facs__chain_a_link__args__weight__data_,
                          chain_a_link_num_max_, offset, num);
}

void GraphSolver::SetChainALinkWeightDataFromStackedDevice(const double* const data, size_t offset,
                                                           size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > chain_a_link_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) + " > chain_a_link_num_max_");
  }
  Matrix22StackedToCaspar(data, facs__chain_a_link__args__weight__data_, chain_a_link_num_max_,
                          offset, num);
}
void GraphSolver::SetChainALinkOffsetDataFromStackedHost(const double* const data, size_t offset,
                                                         size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > chain_a_link_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) + " > chain_a_link_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_, data, 2 * num * sizeof(double), cudaMemcpyHostToDevice);
  Matrix21StackedToCaspar(marker__scratch_inout_, facs__chain_a_link__args__offset__data_,
                          chain_a_link_num_max_, offset, num);
}

void GraphSolver::SetChainALinkOffsetDataFromStackedDevice(const double* const data, size_t offset,
                                                           size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > chain_a_link_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) + " > chain_a_link_num_max_");
  }
  Matrix21StackedToCaspar(data, facs__chain_a_link__args__offset__data_, chain_a_link_num_max_,
                          offset, num);
}
void GraphSolver::SetChainAPriorNum(const size_t num) {
  if (num > chain_a_prior_num_max_) {
    throw std::runtime_error(std::to_string(num) + " > chain_a_prior_num_max_");
  }
  chain_a_prior_num_ = num;
}
void GraphSolver::SetChainAPriorAIndicesFromHost(const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != chain_a_prior_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != chain_a_prior_num_. Use Setchain_a_priorNum "
                             "before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_, indices, num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetChainAPriorAIndicesFromDevice((unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetChainAPriorAIndicesFromDevice(const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != chain_a_prior_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != chain_a_prior_num_. Use Setchain_a_priorNum "
                             "before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error("Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
                             ", num: " + std::to_string(num) +
                             ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(indices, facs__chain_a_prior__args__a__idx_shared_, num);
}
void GraphSolver::SetChainAPriorTargetDataFromStackedHost(const double* const data, size_t offset,
                                                          size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > chain_a_prior_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) + " > chain_a_prior_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_, data, 2 * num * sizeof(double), cudaMemcpyHostToDevice);
  Matrix21StackedToCaspar(marker__scratch_inout_, facs__chain_a_prior__args__target__data_,
                          chain_a_prior_num_max_, offset, num);
}

void GraphSolver::SetChainAPriorTargetDataFromStackedDevice(const double* const data, size_t offset,
                                                            size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > chain_a_prior_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) + " > chain_a_prior_num_max_");
  }
  Matrix21StackedToCaspar(data, facs__chain_a_prior__args__target__data_, chain_a_prior_num_max_,
                          offset, num);
}
void GraphSolver::SetChainBLinkNum(const size_t num) {
  if (num > chain_b_link_num_max_) {
    throw std::runtime_error(std::to_string(num) + " > chain_b_link_num_max_");
  }
  chain_b_link_num_ = num;
}
void GraphSolver::SetChainBLinkWeightDataFromStackedHost(const double* const data, size_t offset,
                                                         size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > chain_b_link_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) + " > chain_b_link_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_, data, 9 * num * sizeof(double), cudaMemcpyHostToDevice);
  Matrix33StackedToCaspar(marker__scratch_inout_, facs__chain_b_link__args__weight__data_,
                          chain_b_link_num_max_, offset, num);
}

void GraphSolver::SetChainBLinkWeightDataFromStackedDevice(const double* const data, size_t offset,
                                                           size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > chain_b_link_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) + " > chain_b_link_num_max_");
  }
  Matrix33StackedToCaspar(data, facs__chain_b_link__args__weight__data_, chain_b_link_num_max_,
                          offset, num);
}
void GraphSolver::SetChainBLinkOffsetDataFromStackedHost(const double* const data, size_t offset,
                                                         size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > chain_b_link_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) + " > chain_b_link_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_, data, 3 * num * sizeof(double), cudaMemcpyHostToDevice);
  Matrix31StackedToCaspar(marker__scratch_inout_, facs__chain_b_link__args__offset__data_,
                          chain_b_link_num_max_, offset, num);
}

void GraphSolver::SetChainBLinkOffsetDataFromStackedDevice(const double* const data, size_t offset,
                                                           size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > chain_b_link_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) + " > chain_b_link_num_max_");
  }
  Matrix31StackedToCaspar(data, facs__chain_b_link__args__offset__data_, chain_b_link_num_max_,
                          offset, num);
}
void GraphSolver::SetChainBPriorNum(const size_t num) {
  if (num > chain_b_prior_num_max_) {
    throw std::runtime_error(std::to_string(num) + " > chain_b_prior_num_max_");
  }
  chain_b_prior_num_ = num;
}
void GraphSolver::SetChainBPriorBIndicesFromHost(const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != chain_b_prior_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != chain_b_prior_num_. Use Setchain_b_priorNum "
                             "before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_, indices, num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetChainBPriorBIndicesFromDevice((unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetChainBPriorBIndicesFromDevice(const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != chain_b_prior_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != chain_b_prior_num_. Use Setchain_b_priorNum "
                             "before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error("Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
                             ", num: " + std::to_string(num) +
                             ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(indices, facs__chain_b_prior__args__b__idx_shared_, num);
}
void GraphSolver::SetChainBPriorTargetDataFromStackedHost(const double* const data, size_t offset,
                                                          size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > chain_b_prior_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) + " > chain_b_prior_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_, data, 3 * num * sizeof(double), cudaMemcpyHostToDevice);
  Matrix31StackedToCaspar(marker__scratch_inout_, facs__chain_b_prior__args__target__data_,
                          chain_b_prior_num_max_, offset, num);
}

void GraphSolver::SetChainBPriorTargetDataFromStackedDevice(const double* const data, size_t offset,
                                                            size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > chain_b_prior_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) + " > chain_b_prior_num_max_");
  }
  Matrix31StackedToCaspar(data, facs__chain_b_prior__args__target__data_, chain_b_prior_num_max_,
                          offset, num);
}
void GraphSolver::SetChainCouplingNum(const size_t num) {
  if (num > chain_coupling_num_max_) {
    throw std::runtime_error(std::to_string(num) + " > chain_coupling_num_max_");
  }
  chain_coupling_num_ = num;
}
void GraphSolver::SetChainCouplingAIndicesFromHost(const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != chain_coupling_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != chain_coupling_num_. Use "
                             "Setchain_couplingNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_, indices, num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetChainCouplingAIndicesFromDevice((unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetChainCouplingAIndicesFromDevice(const unsigned int* const indices,
                                                     size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != chain_coupling_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != chain_coupling_num_. Use "
                             "Setchain_couplingNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error("Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
                             ", num: " + std::to_string(num) +
                             ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(indices, facs__chain_coupling__args__a__idx_shared_, num);
}
void GraphSolver::SetChainCouplingBIndicesFromHost(const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != chain_coupling_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != chain_coupling_num_. Use "
                             "Setchain_couplingNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_, indices, num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetChainCouplingBIndicesFromDevice((unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetChainCouplingBIndicesFromDevice(const unsigned int* const indices,
                                                     size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != chain_coupling_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != chain_coupling_num_. Use "
                             "Setchain_couplingNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error("Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
                             ", num: " + std::to_string(num) +
                             ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(indices, facs__chain_coupling__args__b__idx_shared_, num);
}
void GraphSolver::SetChainCouplingWeightDataFromStackedHost(const double* const data, size_t offset,
                                                            size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > chain_coupling_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) + " > chain_coupling_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_, data, 6 * num * sizeof(double), cudaMemcpyHostToDevice);
  Matrix23StackedToCaspar(marker__scratch_inout_, facs__chain_coupling__args__weight__data_,
                          chain_coupling_num_max_, offset, num);
}

void GraphSolver::SetChainCouplingWeightDataFromStackedDevice(const double* const data,
                                                              size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > chain_coupling_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) + " > chain_coupling_num_max_");
  }
  Matrix23StackedToCaspar(data, facs__chain_coupling__args__weight__data_, chain_coupling_num_max_,
                          offset, num);
}
void GraphSolver::SetChainCouplingOffsetDataFromStackedHost(const double* const data, size_t offset,
                                                            size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > chain_coupling_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) + " > chain_coupling_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_, data, 2 * num * sizeof(double), cudaMemcpyHostToDevice);
  Matrix21StackedToCaspar(marker__scratch_inout_, facs__chain_coupling__args__offset__data_,
                          chain_coupling_num_max_, offset, num);
}

void GraphSolver::SetChainCouplingOffsetDataFromStackedDevice(const double* const data,
                                                              size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > chain_coupling_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) + " > chain_coupling_num_max_");
  }
  Matrix21StackedToCaspar(data, facs__chain_coupling__args__offset__data_, chain_coupling_num_max_,
                          offset, num);
}

size_t GraphSolver::get_nbytes() {
  size_t offset = 0;
  size_t at_least = 0;
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * ChainA_num_, 4);
  increment_offset<double>(offset, 2 * ChainA_num_, 4);
  increment_offset<double>(offset, 2 * ChainA_num_, 4);
  increment_offset<double>(offset, 4 * ChainB_num_, 4);
  increment_offset<double>(offset, 4 * ChainB_num_, 4);
  increment_offset<double>(offset, 4 * ChainB_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * chain_a_link_num_, 4);
  increment_offset<double>(offset, 4 * chain_a_link_num_, 4);
  increment_offset<double>(offset, 2 * chain_a_link_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * chain_a_prior_num_, 4);
  increment_offset<double>(offset, 2 * chain_a_prior_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * chain_b_link_num_, 4);
  increment_offset<double>(offset, 9 * chain_b_link_num_, 4);
  increment_offset<double>(offset, 4 * chain_b_link_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * chain_b_prior_num_, 4);
  increment_offset<double>(offset, 4 * chain_b_prior_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * chain_coupling_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * chain_coupling_num_, 4);
  increment_offset<double>(offset, 6 * chain_coupling_num_, 4);
  increment_offset<double>(offset, 2 * chain_coupling_num_, 4);
  at_least = std::max(
      at_least, offset + std::max({2 * ChainA_num_max_, 3 * ChainB_num_max_}) * sizeof(double));
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * chain_a_link_num_, 4);
  increment_offset<double>(offset, 2 * chain_a_prior_num_, 4);
  increment_offset<double>(offset, 4 * chain_b_link_num_, 4);
  increment_offset<double>(offset, 4 * chain_b_prior_num_, 4);
  increment_offset<double>(offset, 2 * chain_coupling_num_, 4);
  increment_offset<double>(offset, 4 * chain_a_link_num_, 4);
  increment_offset<double>(offset, 4 * chain_a_link_num_, 4);
  increment_offset<double>(offset, 4 * ChainA_num_, 4);
  increment_offset<double>(offset, 0 * chain_a_prior_num_, 4);
  increment_offset<double>(offset, 9 * chain_b_link_num_, 4);
  increment_offset<double>(offset, 9 * chain_b_link_num_, 4);
  increment_offset<double>(offset, 9 * ChainB_num_, 4);
  increment_offset<double>(offset, 0 * chain_b_prior_num_, 4);
  increment_offset<double>(offset, 0 * chain_coupling_num_, 4);
  increment_offset<double>(offset, 6 * chain_coupling_num_, 4);
  increment_offset<double>(offset, 2 * ChainA_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 4 * ChainB_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * ChainA_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 4 * ChainB_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * ChainA_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 4 * ChainB_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * ChainA_num_, 4);
  increment_offset<double>(offset, 4 * ChainB_num_, 4);
  increment_offset<double>(offset, 0 * 0, 1);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * ChainA_num_, 4);
  increment_offset<double>(offset, 4 * ChainB_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * ChainA_num_, 4);
  increment_offset<double>(offset, 4 * ChainB_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * ChainA_num_, 4);
  increment_offset<double>(offset, 4 * ChainB_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * ChainA_num_, 4);
  increment_offset<double>(offset, 1 * ChainA_num_, 4);
  increment_offset<double>(offset, 4 * ChainB_num_, 4);
  increment_offset<double>(offset, 4 * ChainB_num_, 4);
  increment_offset<double>(offset, 0 * 0, 1);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * chain_a_link_num_, 4);
  increment_offset<double>(offset, 2 * chain_a_prior_num_, 4);
  increment_offset<double>(offset, 4 * chain_b_link_num_, 4);
  increment_offset<double>(offset, 4 * chain_b_prior_num_, 4);
  increment_offset<double>(offset, 2 * chain_coupling_num_, 4);
  increment_offset<double>(offset, 0 * 0, 1);
  increment_offset<double>(offset, 1 * 1, 1);
  increment_offset<double>(offset, 1 * 1, 1);
  increment_offset<double>(offset, 1 * 1, 1);
  increment_offset<double>(offset, 1 * 1, 1);
  increment_offset<double>(offset, 1 * 1, 1);
  increment_offset<double>(offset, 1 * 1, 1);
  increment_offset<double>(offset, 1 * 1, 1);
  increment_offset<double>(offset, 1 * 1, 1);
  increment_offset<double>(offset, 1 * 1, 1);
  increment_offset<double>(offset, 1 * 1, 1);
  increment_offset<double>(offset, 1 * 1, 1);
  increment_offset<double>(offset, 20 * ChainA_num_, 4);
  increment_offset<double>(offset, 42 * ChainB_num_, 4);

  return std::max(offset, at_least);
}

}  // namespace caspar