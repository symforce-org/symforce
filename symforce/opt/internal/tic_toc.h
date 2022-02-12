/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sym {
namespace internal {

using Clock = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;
using Duration = TimePoint::duration;

TimePoint GetMonotonicTime();
void TicTocUpdate(const std::string& name, const Duration& duration);

class ScopedTicToc {
 public:
  explicit ScopedTicToc(const std::string& name) : name_(name), start_(GetMonotonicTime()) {}

  ~ScopedTicToc() {
    TicTocUpdate(name_, GetMonotonicTime() - start_);
  }

 private:
  std::string name_;
  TimePoint start_;
};

// Stores accumulated statistics about time spent doing something.
class TicTocStats {
 public:
  void Update(const Duration& duration);
  void Merge(const TicTocStats& other);

  double TotalTime() const;
  double AverageTime() const;
  double MaxTime() const;
  double MinTime() const;
  int64_t Count() const;

 private:
  int64_t num_tics_{0};
  Duration total_time_{0};
  Duration min_time_{std::numeric_limits<Duration::rep>::max()};
  Duration max_time_{std::numeric_limits<Duration::rep>::min()};
};

// Each thread gets one of these
class ThreadContext {
 public:
  ThreadContext() = default;
  ~ThreadContext();

  // Add a sample of length Duration to the block for name
  void Update(const std::string& name, const Duration& duration);

 private:
  std::unordered_map<std::string, TicTocStats> block_map_;
};

class TicTocManager {
 public:
  TicTocManager();
  ~TicTocManager();

  // Create string with results of all tic tocs.
  void PrintTimingResults(std::ostream& out = std::cout) const;

  // Set whether or not the tic-toc manager prints on destruction. Default true.
  void SetPrintOnDestruction(const bool print_on_destruction) {
    print_on_destruction_ = print_on_destruction;
  }

  // Lock the global blockmap, then merge blocks from the thread blockmap into blocks from the
  // global blockmap. Called from the producer thread on termination and locks the consumer thread.
  void Consume(const std::unordered_map<std::string, TicTocStats>& thread_map);

 private:
  // Return the TicTocBlock for the given name, select out of tictoc_blocks_ and created if it does
  // not yet exist. This function does not lock tictoc_blocks_ structure.
  TicTocStats& GetStatsWithoutLock(const std::string& name);

  std::unordered_map<std::string, TicTocStats> tictoc_blocks_;
  mutable std::mutex tictoc_blocks_mutex_;

  bool print_on_destruction_{true};
};

}  // namespace internal
}  // namespace sym
