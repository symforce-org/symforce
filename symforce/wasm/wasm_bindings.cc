#include <Eigen/Dense>
#include <emscripten/bind.h>

#include <sym/rot3.h>

using namespace emscripten;

// TODO(rachel): Use emscripten::val for template params. See:
// https://github.com/emscripten-core/emscripten/issues/4887#issuecomment-283285974
EMSCRIPTEN_BINDINGS(symforce)
{
    class_<sym::Rot3d>("Rot3")
        .constructor<>()
        .function("data", &sym::Rot3d::Data)
        .function("inverse", &sym::Rot3d::Inverse)
        .function("toRotationMatrix", &sym::Rot3d::ToRotationMatrix)
        .function("toYawPitchRoll", &sym::Rot3d::ToYawPitchRoll)
        .class_function("identity", &sym::Rot3d::Identity)
        // TODO(hayk): Figure this out.
        // .class_function("fromYawPitchRoll", static_cast<sym::Rot3d (sym::Rot3d::*)(const double, const double, const double)>(&sym::Rot3d::FromYawPitchRoll))
        // Added print for testing.
        .function("toString", optional_override([](const sym::Rot3d &self) {
                      std::stringstream buf;
                      buf << self.Data().transpose() << std::endl;
                      return buf.str();
                  }));

    // Need Eigen return values.

    // TODO(rachel): How to handle overloads? Errors:
    // JS: Cannot register public name 'Matrix' twice.
    // emcc: template argument for non-type template parameter must be an expression
    class_<Eigen::Matrix<double, 3, 3>>("Matrix33")
        .constructor<>()
        .function("toString", optional_override([](const Eigen::Matrix<double, 3, 3> &self) {
                      std::stringstream buf;
                      buf << self << std::endl;
                      return buf.str();
                  }));

    class_<Eigen::Matrix<double, 3, 1>>("Matrix31")
        .constructor<>()
        .function("toString", optional_override([](const Eigen::Matrix<double, 3, 1> &self) {
                      std::stringstream buf;
                      buf << self << std::endl;
                      return buf.str();
                  }));

    class_<Eigen::Matrix<double, 4, 1>>("Matrix41")
        .constructor<>()
        // Added print for testing.
        .function("toString", optional_override([](const Eigen::Matrix<double, 4, 1> &self) {
                      std::stringstream buf;
                      buf << self << std::endl;
                      return buf.str();
                  }));
}
