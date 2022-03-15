/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./tic_toc.h"

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <spdlog/spdlog.h>

namespace sym {
namespace internal {

namespace {

// These must be defined in this order, so that they are constructed in this order, and more
// importantly destroyed in reverse order
// https://stackoverflow.com/a/469613
static TicTocManager g_tic_toc{};
static thread_local ThreadContext g_thread_ctx{};

double ToSeconds(const Duration& duration) {
  return static_cast<double>(duration.count()) * Duration::period::num / Duration::period::den;
}

}  // namespace

TimePoint GetMonotonicTime() {
  return Clock::now();
}

// Accumulate a duration specified by the startime and end time with the named block
void TicTocUpdate(const std::string& name, const Duration& duration) {
  g_thread_ctx.Update(name, duration);
}

// --------------------------------------------------------------------------------------------
//                                          TicTocStats
// --------------------------------------------------------------------------------------------

void TicTocStats::Update(const Duration& duration) {
  num_tics_++;
  total_time_ += duration;
  min_time_ = std::min(min_time_, duration);
  max_time_ = std::max(max_time_, duration);
}

void TicTocStats::Merge(const TicTocStats& other) {
  num_tics_ += other.num_tics_;
  total_time_ += other.total_time_;
  min_time_ = std::min(min_time_, other.min_time_);
  max_time_ = std::max(max_time_, other.max_time_);
}

double TicTocStats::TotalTime() const {
  return ToSeconds(total_time_);
}

double TicTocStats::AverageTime() const {
  if (num_tics_ == 0) {
    return 0;
  }
  return TotalTime() / static_cast<double>(num_tics_);
}

double TicTocStats::MaxTime() const {
  return ToSeconds(max_time_);
}

double TicTocStats::MinTime() const {
  return ToSeconds(min_time_);
}

int64_t TicTocStats::Count() const {
  return num_tics_;
}

// --------------------------------------------------------------------------------------------
//                                    ThreadContext
// --------------------------------------------------------------------------------------------

ThreadContext::~ThreadContext() {
  g_tic_toc.Consume(block_map_);
}

void ThreadContext::Update(const std::string& name, const Duration& duration) {
  // This intentionally default-constructs the block if it doesn't exist
  block_map_[name].Update(duration);
}

// --------------------------------------------------------------------------------------------
//                                    TicTocManager
// --------------------------------------------------------------------------------------------

TicTocManager::TicTocManager() {
  // Allow env variable to disable print on destruction
  if (std::getenv("SYMFORCE_TIC_TOC_QUIET") != nullptr) {
    print_on_destruction_ = false;
  }
}

TicTocManager::~TicTocManager() {
  if (print_on_destruction_ && spdlog::should_log(spdlog::level::info)) {
    PrintTimingResults();
  }
}

void TicTocManager::PrintTimingResults(std::ostream& out) const {
  std::vector<std::pair<std::string, TicTocStats>> blocks;

  {
    std::lock_guard<std::mutex> lock(tictoc_blocks_mutex_);
    blocks.reserve(tictoc_blocks_.size());
    for (const auto& block_pair : tictoc_blocks_) {
      blocks.push_back(block_pair);
    }
  }

  if (blocks.empty()) {
    return;
  }

  // Sort blocks by total time
  std::sort(blocks.begin(), blocks.end(), [](const auto& a, const auto& b) {
    return a.second.TotalTime() > b.second.TotalTime();
  });

  int longest_name = 0;
  for (const auto& block : blocks) {
    longest_name = std::max<int>(block.first.size(), longest_name);
  }

  const std::string header_fmt =
      fmt::format("{{:<{}}}", longest_name) + " : {:^14} | {:^14} | {:^14} | {:^14} | {:^14}\n";
  const std::string output_fmt = fmt::format("{{:<{}}}", longest_name) +
                                 " : {:^14} | {:^14.5} | {:^14.5} | {:^14.5} | {:^14.5}\n";

  std::string separator(longest_name + 1, '-');
  for (int i = 0; i < 5; i++) {
    separator += std::string("+") + std::string(16, '-');
  }

  const std::string legend = fmt::format(header_fmt, "   Name", "Count", "Total Time (s)",
                                         "Mean Time (s)", "Max Time (s)", "Min Time (s)");

  fmt::print(out, "\nSymForce TicToc Results:\n");
  fmt::print(out, legend);
  fmt::print(out, separator + "\n");

  for (const auto& block_pair : blocks) {
    const auto& name = block_pair.first;
    const auto& block = block_pair.second;

    fmt::print(out, output_fmt, name, block.Count(), float(block.TotalTime()),
               float(block.AverageTime()), float(block.MaxTime()), float(block.MinTime()));
  }
}

void TicTocManager::Consume(const std::unordered_map<std::string, TicTocStats>& thread_map) {
  // Lock the consumer thread
  std::lock_guard<std::mutex> lock(tictoc_blocks_mutex_);
  for (const auto& pair : thread_map) {
    GetStatsWithoutLock(pair.first).Merge(pair.second);
  }
}

TicTocStats& TicTocManager::GetStatsWithoutLock(const std::string& name) {
  // This intentionally default-constructs the block if it doesn't exist
  return tictoc_blocks_[name];
}

}  // namespace internal
}  // namespace sym
