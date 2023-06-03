/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <algorithm>
#include <cctype>
#include <cstdlib>

#include <spdlog/spdlog.h>
//
#include <spdlog/fmt/bundled/ranges.h>

namespace sym {
namespace internal {

bool SetLogLevel(const std::string& log_level) {
  // Convert to lowercase
  // https://stackoverflow.com/a/313990
  const std::string log_level_lower = [&log_level]() {
    std::string log_level_lower = log_level;
    std::transform(log_level_lower.begin(), log_level_lower.end(), log_level_lower.begin(),
                   [](const char c) { return std::tolower(c); });
    return log_level_lower;
  }();

  // These should match the python logging levels from
  // https://docs.python.org/3/library/logging.html#logging-levels
  const std::unordered_map<std::string, spdlog::level::level_enum> level_for_string = {
      {"debug", spdlog::level::debug},       {"info", spdlog::level::info},
      {"warning", spdlog::level::warn},      {"error", spdlog::level::err},
      {"critical", spdlog::level::critical},
  };

  const auto maybe_log_level = level_for_string.find(log_level_lower);

  if (maybe_log_level == level_for_string.end()) {
    spdlog::error("Invalid log level: \"{}\", keeping current level \"{}\"", log_level_lower,
                  spdlog::level::to_string_view(spdlog::get_level()));

    std::vector<std::string> keys;
    std::transform(level_for_string.begin(), level_for_string.end(), std::back_inserter(keys),
                   [](const auto& p) { return p.first; });
    spdlog::error("Allowed values are: {}", keys);
    return false;
  }

  spdlog::set_level(maybe_log_level->second);
  spdlog::debug("Set log level to: {}", maybe_log_level->first);
  return true;
}

class SpdlogConfigurator {
 private:
  SpdlogConfigurator() {
    SetLogLevelFromEnvironment();
  }

  void SetLogLevelFromEnvironment() const {
    // Default log level to info
    spdlog::set_level(spdlog::level::info);

    // This is not an owning pointer - it points to the actual location of the string for this
    // environment variable in the process address space.  And this is thread safe, assuming no one
    // modifies SYMFORCE_LOGLEVEL at that location
    // https://stackoverflow.com/a/30476732
    const char* const log_level_cstr = std::getenv("SYMFORCE_LOGLEVEL");

    if (log_level_cstr == nullptr) {
      // getenv returns null if the variable does not exist
      return;
    }

    const std::string log_level_str = log_level_cstr;

    if (log_level_str.empty()) {
      return;
    }

    SetLogLevel(log_level_str);
  }

  // Construct a static SpdlogConfigurator, so the constructor runs on process startup
  static SpdlogConfigurator g_spdlog_configurator_;
};

SpdlogConfigurator SpdlogConfigurator::g_spdlog_configurator_{};

}  // namespace internal
}  // namespace sym
