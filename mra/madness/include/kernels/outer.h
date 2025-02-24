#ifndef MRA_OUTER_H_INCL
#define MRA_OUTER_H_INCL

#include "platform.h"
#include "types.h"
#include "tensorview.h"
#include "tensor.h"

namespace mra{

  namespace detail{

    template <typename T, Dimension NDIM>
    SCOPE void outer(
      const TensorView<T, NDIM>& left,
      const TensorView<T, NDIM>& right,
      TensorView<T, NDIM>& result)
    {
      static_assert(result.ndim() <= NDIM, "too many dimensions in the result", NDIM);
      static_assert(left.ndim() + right.ndim() == result.ndim(),
                     "inconsistent dimension in outer result", result.ndim());

      if (threadIdx.z == 0) {
        T* ptr = result.ptr();
        auto iter = right.unary_iterator(IterLevel::Vector, false, 0);
        for (auto iter0=left.unary_iterator(IterLevel::Vector, false, 0);
            iter0.ptr() != nullptr;
            ++iter0) {
          T val = *iter0;
          for (iter += thread_id();
               iter.ptr() != nullptr;
               iter += blockDim.x*blockDim.y) {
            T dimj = right.dim(0); // dimj = iter.dimj ??
            T* _p0  = iter.ptr();
            T Tstride = iter.s0();
            for (size_type i = 0; i<dimj; ++i, _p0 += Tstride) {
              *ptr++ = val * (*_p0);
            }
          }
        }
      }
    }
  }
}

#endif // MRA_OUTER_H_INCL
