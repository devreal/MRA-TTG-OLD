#ifndef MRA_KERNELS_NORM_H
#define MRA_KERNELS_NORM_H

#include "platform.h"
#include "types.h"
#include "tensorview.h"
#include "key.h"

namespace mra {
  namespace detail {
    template <typename T, Dimension NDIM>
    DEVSCOPE void norm_kernel_impl(
      const T* node, T* norm, size_type K)
    {
      const bool is_t0 = 0 == (threadIdx.x + threadIdx.y + threadIdx.z);
      SHARED TensorView<T, NDIM> n;
      if (is_t0) {
        n = TensorView<T, NDIM>(node, K);
      }
      SYNCTHREADS();

      /* compressed form */
      T sum = 0.0;
      foreach_idx(n, [&](auto... idx) {
        sum += n(idx...)*n(idx...);
      });

      sum = std::sqrt(sum);
      *norm = sum;
    }

    template <typename T, Dimension NDIM>
    GLOBALSCOPE void norm_kernel(
      const T* node, T* norm,
      size_type N, size_type K, const Key<NDIM>& key)
    {
      const size_type K2NDIM = std::pow(K, NDIM);
      size_type blockid = blockIdx.x;
      norm_kernel_impl<T, NDIM>(nullptr == node ? nullptr : &node[K2NDIM*blockid],
                                &norm[blockid],
                                K);
    }
  } // namespace detail


  template <typename T, Dimension NDIM>
  void submit_norm_kernel(
    const Key<NDIM>& key,
    size_type N,
    size_type K,
    const TensorView<T, NDIM>& in,
    const T norm,
    cudaStream_t stream)
  {
    Dim3 thread_dims = Dim3(K, K, 1);

    CALL_KERNEL(detail::norm_kernel, N, thread_dims, 0, stream,
        (in.data(), norm, N, K, key));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_KERNELS_NORM_H