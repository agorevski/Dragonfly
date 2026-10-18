#include "strata/kernels/pdl.hpp"

namespace strata::kernels {

// The isolated engine does not capture Strata's verify-window PDL graph.
bool& pdl_scope() {
    static thread_local bool enabled = false;
    return enabled;
}

bool pdl_supported() { return false; }
bool pdl_launch_ok(const void*, cudaStream_t) { return false; }

}  // namespace strata::kernels
