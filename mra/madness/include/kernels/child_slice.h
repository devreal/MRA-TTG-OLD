#ifndef MRA_KERNELS_CHILD_SLICE_H
#define MRA_KERNELS_CHILD_SLICE_H

#include "key.h"
#include "types.h"

namespace mra {
  template<Dimension NDIM>
  SCOPE
  std::array<Slice, NDIM> get_child_slice(Key<NDIM> key, std::size_t K, int child) {
    std::array<Slice,NDIM> slices;
    for (size_t d = 0; d < NDIM; ++d) {
      int b = (child>>d) & 0x1;
      slices[d] = Slice(K*b, K*(b+1));
    }
    return slices;
  }
} // namespace mra

#endif // MRA_KERNELS_CHILD_SLICE_H
