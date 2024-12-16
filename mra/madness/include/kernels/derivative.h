#ifndef MRA_KERNELS_DERIVATIVE_H
#define MRA_KERNELS_DERIVATIVE_H

#include "platform.h"
#include "types.h"
#include "tensorview.h"
#include "key.h"

namespace mra {
  namespace detail {
    template <typename T, Dimension NDIM>
    DEVSCOPE void derivative_inner()
    {
      // diff2i()
    }

    template <typename T, Dimension NDIM>
    DEVSCOPE void derivative_boundary(
      const T* node_left,
      const T* node_center,
      const T* node_right,
      bool axis) // axis to determine left or right boundary
    {
      //diff2b()
    }

    template <typename T, Dimension NDIM>
    // need to implement diff2i() and diff2b()
    DEVSCOPE void derivative_kernel_impl(
      const T* node_left,
      const T* node_center,
      const T* node_right,
      bool is_bdy
    )
      { // if we reached here, all checks have passed, and we do the transform to compute the derivative
        // for a given axis by calling either derivative_inner() or derivative_boundary()
      }

    template <typename T, Dimension NDIM>
    GLOBALSCOPE void derivative_kernel()
    {
    }

  } // namespace detail

  void submit_derivative_kernel()
  {
    // given a key(level, translation), check if any of the input nodes have any children
    // if not, then we can compute the derivative by applying the
  }

} // namespace mra

#endif // MRA_KERNELS_DERIVATIVE_H
