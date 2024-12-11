#ifndef MRA_KERNELS_DERIVATIVE_H
#define MRA_KERNELS_DERIVATIVE_H

#include "platform.h"
#include "types.h"
#include "tensorview.h"
#include "key.h"

namespace mra {
  namespace detail {
    template <typename T, Dimension NDIM>
    DEVSCOPE void derivative_kernel_impl(
      T* r0,
      T* rminus,
      T* rplus,
      size_type K,
      mra::Key<NDIM> key)
      {
        const bool is_t0 = 0 == (threadIdx.x + threadIdx.y + threadIdx.z);
        SHARED TensorView<T, 2> r_center, r_right, r_left;
        if (is_t0) {
          r_center = TensorView<T, 2>(r0, K);
          r_right = TensorView<T, 2>(rplus, K);
          r_left = TensorView<T, 2>(rminus, K);
        }
        SYNCTHREADS();

        double iphase = 1.0;
        /* wrap the loops into foreach_idx() and obtain gammaij and Kij somehow */
        foreach_idx(r_center, [&](auto i, auto j) {
          double jphase = 1.0;
          double gammaij = std::sqrt(double((2*i+1)*(2*j+1)));
          double Kij;
          if (((i-j)>0) && (((i-j)%2)==1))
            Kij = 2.0;
          else
            Kij = 0.0;

          r_center(i,j) = T(0.5*(1.0 - iphase*jphase - 2.0*Kij)*gammaij);
          r_left(i,j) = T(0.5*jphase*gammaij);
          r_right(i,j) = T(-0.5*iphase*gammaij);
        });
      }

  } // namespace detail
} // namespace mra

#endif // MRA_KERNELS_DERIVATIVE_H
