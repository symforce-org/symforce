#include "solver.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

#include "caspar_mappings.h"
#include "kernel_Matrix21_alpha_denominator_or_beta_numerator.h"
#include "kernel_Matrix21_alpha_numerator_denominator.h"
#include "kernel_Matrix21_normalize.h"
#include "kernel_Matrix21_pred_decrease_times_two.h"
#include "kernel_Matrix21_retract.h"
#include "kernel_Matrix21_start_w.h"
#include "kernel_Matrix21_start_w_contribute.h"
#include "kernel_Matrix21_update_Mp.h"
#include "kernel_Matrix21_update_p.h"
#include "kernel_Matrix21_update_r.h"
#include "kernel_Matrix21_update_r_first.h"
#include "kernel_Matrix21_update_step.h"
#include "kernel_Matrix21_update_step_first.h"
#include "kernel_rosenbrock_jtjnjtr_direct.h"
#include "kernel_rosenbrock_res_jac.h"
#include "kernel_rosenbrock_res_jac_first.h"
#include "kernel_rosenbrock_score.h"
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

GraphSolver::GraphSolver(const SolverParams<double>& params, size_t Matrix21_num_max,
                         size_t rosenbrock_num_max, int device_id)
    : params_(params),
      device_id_(device_id),
      Matrix21_num_(Matrix21_num_max),
      Matrix21_num_max_(Matrix21_num_max),
      rosenbrock_num_(rosenbrock_num_max),
      rosenbrock_num_max_(rosenbrock_num_max) {
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
  nodes__Matrix21__storage_current_ =
      assign_and_increment<double>(origin_ptr_, offset, 2 * Matrix21_num_, 4);
  nodes__Matrix21__storage_check_ =
      assign_and_increment<double>(origin_ptr_, offset, 2 * Matrix21_num_, 4);
  nodes__Matrix21__storage_new_best_ =
      assign_and_increment<double>(origin_ptr_, offset, 2 * Matrix21_num_, 4);
  facs__rosenbrock__args__x__idx_shared_ =
      assign_and_increment<SharedIndex>(origin_ptr_, offset, 1 * rosenbrock_num_, 4);
  marker__scratch_inout_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  facs__rosenbrock__res_ =
      assign_and_increment<double>(origin_ptr_, offset, 2 * rosenbrock_num_, 4);
  facs__rosenbrock__args__x__jac_ =
      assign_and_increment<double>(origin_ptr_, offset, 1 * rosenbrock_num_, 4);
  nodes__Matrix21__z_ = assign_and_increment<double>(origin_ptr_, offset, 2 * Matrix21_num_, 4);
  nodes__Matrix21__z_end__ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Matrix21__p_ = assign_and_increment<double>(origin_ptr_, offset, 2 * Matrix21_num_, 4);
  nodes__Matrix21__p_end__ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Matrix21__step_ = assign_and_increment<double>(origin_ptr_, offset, 2 * Matrix21_num_, 4);
  nodes__Matrix21__step_end__ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  marker__w_start_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Matrix21__w_ = assign_and_increment<double>(origin_ptr_, offset, 2 * Matrix21_num_, 4);
  marker__w_end_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 1);
  marker__r_0_start_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Matrix21__r_0_ = assign_and_increment<double>(origin_ptr_, offset, 2 * Matrix21_num_, 4);
  marker__r_0_end_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  marker__r_k_start_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Matrix21__r_k_ = assign_and_increment<double>(origin_ptr_, offset, 2 * Matrix21_num_, 4);
  marker__r_k_end_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  marker__Mp_start_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Matrix21__Mp_ = assign_and_increment<double>(origin_ptr_, offset, 2 * Matrix21_num_, 4);
  marker__Mp_end_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  marker__precond_start_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Matrix21__precond_diag_ =
      assign_and_increment<double>(origin_ptr_, offset, 2 * Matrix21_num_, 4);
  nodes__Matrix21__precond_tril_ =
      assign_and_increment<double>(origin_ptr_, offset, 1 * Matrix21_num_, 4);
  marker__precond_end_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 1);
  marker__jp_start_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  facs__rosenbrock__jp_ = assign_and_increment<double>(origin_ptr_, offset, 2 * rosenbrock_num_, 4);
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
        std::swap(nodes__Matrix21__storage_check_, nodes__Matrix21__storage_new_best_);
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
      std::swap(nodes__Matrix21__storage_check_, nodes__Matrix21__storage_new_best_);
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
      std::swap(nodes__Matrix21__storage_current_, nodes__Matrix21__storage_new_best_);

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

  RosenbrockResJacFirst(
      nodes__Matrix21__storage_current_, Matrix21_num_max_, facs__rosenbrock__args__x__idx_shared_,

      facs__rosenbrock__res_, rosenbrock_num_, solver__res_tot_, nodes__Matrix21__r_k_,
      Matrix21_num_, nodes__Matrix21__precond_diag_, Matrix21_num_, nodes__Matrix21__precond_tril_,
      Matrix21_num_, rosenbrock_num_);
  Copy(marker__r_k_start_, marker__r_k_end_, marker__r_0_start_);
  Copy(marker__r_k_start_, marker__r_k_end_, marker__Mp_start_);
  return 0.5 * ReadCuMem(solver__res_tot_);
}
void GraphSolver::DoResJac() {
  Zero(solver__res_tot_, solver__res_tot_ + 1);
  Zero(marker__r_0_start_, marker__precond_end_);

  RosenbrockResJac(nodes__Matrix21__storage_current_, Matrix21_num_max_,
                   facs__rosenbrock__args__x__idx_shared_,

                   facs__rosenbrock__res_, rosenbrock_num_,

                   nodes__Matrix21__r_k_, Matrix21_num_, nodes__Matrix21__precond_diag_,
                   Matrix21_num_, nodes__Matrix21__precond_tril_, Matrix21_num_, rosenbrock_num_);
  Copy(marker__r_k_start_, marker__r_k_end_, marker__r_0_start_);
  Copy(marker__r_k_start_, marker__r_k_end_, marker__Mp_start_);
}

void GraphSolver::DoNormalize() {
  double* z;
  z = pcg_iter_ == 0 ? nodes__Matrix21__p_ : nodes__Matrix21__z_;
  Matrix21Normalize(nodes__Matrix21__precond_diag_, Matrix21_num_, nodes__Matrix21__precond_tril_,
                    Matrix21_num_, nodes__Matrix21__r_k_, Matrix21_num_, solver__current_diag_, z,
                    Matrix21_num_, Matrix21_num_);
}

void GraphSolver::DoUpdateMp() {
  Matrix21UpdateMp(nodes__Matrix21__r_k_, Matrix21_num_, nodes__Matrix21__Mp_, Matrix21_num_,
                   solver__beta_, nodes__Matrix21__Mp_, Matrix21_num_, nodes__Matrix21__w_,
                   Matrix21_num_, Matrix21_num_);
}

void GraphSolver::DoJtjpDirect() {}

void GraphSolver::DoAlphaFirst() {
  Zero(solver__alpha_numerator_, solver__alpha_denominator_ + 1);
  Matrix21AlphaNumeratorDenominator(
      nodes__Matrix21__p_, Matrix21_num_, nodes__Matrix21__r_k_, Matrix21_num_, nodes__Matrix21__w_,
      Matrix21_num_, solver__alpha_numerator_, solver__alpha_denominator_, Matrix21_num_);

  AlphaFromNumDenom(solver__alpha_numerator_, solver__alpha_denominator_, solver__alpha_,
                    solver__neg_alpha_);
}

void GraphSolver::DoAlpha() {
  Zero(solver__alpha_denominator_, solver__alpha_denominator_ + 1);
  Matrix21AlphaDenominatorOrBetaNumerator(nodes__Matrix21__p_, Matrix21_num_, nodes__Matrix21__w_,
                                          Matrix21_num_, solver__alpha_denominator_, Matrix21_num_);

  AlphaFromNumDenom(solver__beta_numerator_, solver__alpha_denominator_, solver__alpha_,
                    solver__neg_alpha_);
}

void GraphSolver::DoUpdateStepFirst() {
  Matrix21UpdateStepFirst(nodes__Matrix21__p_, Matrix21_num_, solver__alpha_,
                          nodes__Matrix21__step_, Matrix21_num_, Matrix21_num_);
}

void GraphSolver::DoUpdateStep() {
  Matrix21UpdateStep(nodes__Matrix21__step_, Matrix21_num_, nodes__Matrix21__p_, Matrix21_num_,
                     solver__alpha_, nodes__Matrix21__step_, Matrix21_num_, Matrix21_num_);
}

void GraphSolver::DoUpdateRFirst() {
  Zero(solver__r_0_norm2_tot_, solver__r_0_norm2_tot_ + 1);

  Matrix21UpdateRFirst(nodes__Matrix21__r_k_, Matrix21_num_, nodes__Matrix21__w_, Matrix21_num_,
                       solver__neg_alpha_, nodes__Matrix21__r_k_, Matrix21_num_,
                       solver__r_0_norm2_tot_, solver__r_kp1_norm2_tot_, Matrix21_num_);

  pcg_r_0_norm2_ = ReadCuMem(solver__r_0_norm2_tot_);
  pcg_r_kp1_norm2_ = ReadCuMem(solver__r_kp1_norm2_tot_);
}

void GraphSolver::DoUpdateR() {
  Zero(solver__r_kp1_norm2_tot_, solver__r_kp1_norm2_tot_ + 1);

  Matrix21UpdateR(nodes__Matrix21__r_k_, Matrix21_num_, nodes__Matrix21__w_, Matrix21_num_,
                  solver__neg_alpha_, nodes__Matrix21__r_k_, Matrix21_num_,
                  solver__r_kp1_norm2_tot_, Matrix21_num_);
  pcg_r_kp1_norm2_ = ReadCuMem(solver__r_kp1_norm2_tot_);
}

double GraphSolver::DoRetractScore() {
  Matrix21Retract(nodes__Matrix21__storage_current_, Matrix21_num_max_, nodes__Matrix21__step_,
                  Matrix21_num_, nodes__Matrix21__storage_check_, Matrix21_num_max_, Matrix21_num_);
  Zero(solver__res_tot_, solver__res_tot_ + 1);
  RosenbrockScore(nodes__Matrix21__storage_check_, Matrix21_num_max_,
                  facs__rosenbrock__args__x__idx_shared_, solver__res_tot_, rosenbrock_num_);
  return 0.5 * ReadCuMem(solver__res_tot_);
}

void GraphSolver::DoBeta() {
  Zero(solver__beta_numerator_, solver__beta_numerator_ + 1);

  Matrix21AlphaDenominatorOrBetaNumerator(nodes__Matrix21__r_k_, Matrix21_num_, nodes__Matrix21__z_,
                                          Matrix21_num_, solver__beta_numerator_, Matrix21_num_);
  BetaFromNumDenom(solver__beta_numerator_, solver__alpha_numerator_, solver__beta_);
}

void GraphSolver::DoUpdateP() {
  Matrix21UpdateP(nodes__Matrix21__z_, Matrix21_num_, nodes__Matrix21__p_, Matrix21_num_,
                  solver__beta_, nodes__Matrix21__p_, Matrix21_num_, Matrix21_num_);
}

double GraphSolver::GetPredDecrease() {
  Zero(solver__pred_decrease_tot_, solver__pred_decrease_tot_ + 1);
  Matrix21PredDecreaseTimesTwo(nodes__Matrix21__step_, Matrix21_num_,
                               nodes__Matrix21__precond_diag_, Matrix21_num_, solver__current_diag_,
                               nodes__Matrix21__r_0_, Matrix21_num_, solver__pred_decrease_tot_,
                               Matrix21_num_);
  return 0.5 * ReadCuMem(solver__pred_decrease_tot_);
}

void GraphSolver::finish_indices() {
  indices_valid_ = true;
}

void GraphSolver::SetMatrix21Num(const size_t num) {
  cudaSetDevice(device_id_);
  if (num > Matrix21_num_max_) {
    throw std::runtime_error(std::to_string(num) + " > Matrix21_num_max_");
  }
  Matrix21_num_ = num;
}

void GraphSolver::SetMatrix21NodesFromStackedHost(const double* const data, const size_t offset,
                                                  const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > Matrix21_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > Matrix21_num_");
  }
  cudaMemcpy(marker__scratch_inout_, data, 2 * num * sizeof(double), cudaMemcpyHostToDevice);
  Matrix21StackedToCaspar(marker__scratch_inout_, nodes__Matrix21__storage_current_,
                          Matrix21_num_max_, offset, num);
}

void GraphSolver::SetMatrix21NodesFromStackedDevice(const double* const data, const size_t offset,
                                                    const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > Matrix21_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > Matrix21_num_");
  }
  Matrix21StackedToCaspar(data, nodes__Matrix21__storage_current_, Matrix21_num_max_, offset, num);
}

void GraphSolver::GetMatrix21NodesToStackedHost(double* const data, const size_t offset,
                                                const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > Matrix21_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > Matrix21_num_");
  }
  Matrix21CasparToStacked(nodes__Matrix21__storage_current_, marker__scratch_inout_,
                          Matrix21_num_max_, offset, num);
  cudaMemcpy(data, marker__scratch_inout_, 2 * num * sizeof(double), cudaMemcpyDeviceToHost);
}

void GraphSolver::GetMatrix21NodesToStackedDevice(double* const data, const size_t offset,
                                                  const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > Matrix21_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > Matrix21_num_");
  }
  Matrix21CasparToStacked(nodes__Matrix21__storage_current_, data, Matrix21_num_max_, offset, num);
}

void GraphSolver::SetRosenbrockNum(const size_t num) {
  if (num > rosenbrock_num_max_) {
    throw std::runtime_error(std::to_string(num) + " > rosenbrock_num_max_");
  }
  rosenbrock_num_ = num;
}
void GraphSolver::SetRosenbrockXIndicesFromHost(const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != rosenbrock_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != rosenbrock_num_. Use SetrosenbrockNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_, indices, num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetRosenbrockXIndicesFromDevice((unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetRosenbrockXIndicesFromDevice(const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != rosenbrock_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != rosenbrock_num_. Use SetrosenbrockNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error("Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
                             ", num: " + std::to_string(num) +
                             ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(indices, facs__rosenbrock__args__x__idx_shared_, num);
}

size_t GraphSolver::get_nbytes() {
  size_t offset = 0;
  size_t at_least = 0;
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * Matrix21_num_, 4);
  increment_offset<double>(offset, 2 * Matrix21_num_, 4);
  increment_offset<double>(offset, 2 * Matrix21_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * rosenbrock_num_, 4);
  at_least = std::max(at_least, offset + std::max({2 * Matrix21_num_max_}) * sizeof(double));
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * rosenbrock_num_, 4);
  increment_offset<double>(offset, 1 * rosenbrock_num_, 4);
  increment_offset<double>(offset, 2 * Matrix21_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * Matrix21_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * Matrix21_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * Matrix21_num_, 4);
  increment_offset<double>(offset, 0 * 0, 1);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * Matrix21_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * Matrix21_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * Matrix21_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * Matrix21_num_, 4);
  increment_offset<double>(offset, 1 * Matrix21_num_, 4);
  increment_offset<double>(offset, 0 * 0, 1);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * rosenbrock_num_, 4);
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

  return std::max(offset, at_least);
}

}  // namespace caspar