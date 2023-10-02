#pragma once

// FIXME: Currently, CPU and CUDA backend are mutually exclusive.
// This is a temporary workaround. We need a better way to support
// multi devices.
#ifndef USE_CUDA

#define AOTI_RUNTIME_DEVICE_CHECK(EXPR)                    \
    bool ok = EXPR;                                        \
    if (!ok) {                                             \
      throw std::runtime_error("CPU runtime error");       \
    }                                                      \

namespace torch {
namespace aot_inductor {

using DeviceStreamType = void*;

} // namespace aot_inductor
} // namespace torch

#endif // !USE_CUDA
