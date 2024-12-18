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
      const T* node,
      T* result_norms,
      std::array<const T*, Key<NDIM>::num_children()>& child_norms,
      size_type blockid,
      size_type K)
    {
      const bool is_t0 = (0 == thread_id());
      SHARED TensorView<T, NDIM> n;
      if (is_t0) {
        n = TensorView<T, NDIM>(node, 2*K);
      }
      SYNCTHREADS();
      T norm = normf(n);
      if (is_t0) {
        /* thread 0 adds the child norms and publishes the result */
        for (int i = 0; i < Key<NDIM>::num_children(); ++i) {
          norm += (child_norms[i] != nullptr) ? child_norms[i][blockid] : T(0.0);
        }
        result_norms[blockid] = norm;
      }
    }

    template <typename T, Dimension NDIM>
    GLOBALSCOPE void norm_kernel(
      const T* node,
      T* result_norms,
      std::array<const T*, Key<NDIM>::num_children()>& child_norms,
      size_type N,
      size_type K,
      const Key<NDIM>& key)
    {
      const size_type TWOK2NDIM = std::pow(2*K, NDIM);
      for (size_type blockid = blockIdx.x; blockid < N; blockid += gridDim.x) {
        norm_kernel_impl<T, NDIM>(nullptr == node ? nullptr : &node[TWOK2NDIM*blockid],
                                  result_norms, child_norms, blockid, K);
      }
    }
  } // namespace detail


  template <typename T, Dimension NDIM>
  void submit_norm_kernel(
    const Key<NDIM>& key,
    size_type N,
    size_type K,
    const TensorView<T, NDIM+1>& in,
    TensorView<T, 1>& result_norms,
    std::array<const T*, Key<NDIM>::num_children()>& child_norms,
    cudaStream_t stream)
  {
    Dim3 thread_dims = Dim3(K, K, 1);
    size_type numthreads = thread_dims.x*thread_dims.y*thread_dims.z;

    CALL_KERNEL(detail::norm_kernel, N, thread_dims, numthreads*sizeof(T), stream,
        (in.data(), result_norms.data(), child_norms, N, K, key));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_KERNELS_NORM_H