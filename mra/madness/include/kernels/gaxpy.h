#ifndef MRA_KERNELS_GAXPY_H
#define MRA_KERNELS_GAXPY_H

#include "platform.h"
#include "types.h"
#include "tensorview.h"
#include "maxk.h"

namespace mra {
  namespace detail {
    template <typename T, Dimension NDIM>
    DEVSCOPE void gaxpy_kernel_impl(
      const TensorView<T, NDIM>& nodeA,
      const TensorView<T, NDIM>& nodeB,
      TensorView<T, NDIM>& nodeR,
      const T scalarA,
      const T scalarB,
      size_type K)
    {
      foreach_idx(nodeR, [&](size_type i) {
        nodeR[i] = scalarA*nodeA[i] + scalarB*nodeB[i];
      });
    }

    template <typename T, Dimension NDIM>
    GLOBALSCOPE void gaxpy_kernel(
      const TensorView<T, NDIM+1> nodeA_view,
      const TensorView<T, NDIM+1> nodeB_view,
      TensorView<T, NDIM+1> nodeR_view,
      const T scalarA,
      const T scalarB,
      size_type N,
      size_type K,
      const Key<NDIM>& key)
    {
      SHARED TensorView<T, NDIM> nodeA, nodeB, nodeR;
      const size_type TWOK2NDIM = std::pow(2*K, NDIM);
      for (size_type blockid = blockIdx.x; blockid < N; blockid += gridDim.x) {
        if (is_team_lead()) {
          nodeA = nodeA_view(blockid);
          nodeB = nodeB_view(blockid);
          nodeR = nodeR_view(blockid);
        }
        gaxpy_kernel_impl<T, NDIM>(nodeA, nodeB, nodeR, scalarA, scalarB, K);
      }
    }
  } // namespace detail


  template <typename T, Dimension NDIM>
  void submit_gaxpy_kernel(
    const Key<NDIM>& key,
    const TensorView<T, NDIM+1>& funcA,
    const TensorView<T, NDIM+1>& funcB,
    TensorView<T, NDIM+1>& funcR,
    const T scalarA,
    const T scalarB,
    size_type N,
    size_type K,
    cudaStream_t stream)
  {
    size_type max_threads = std::min(2*K, 2*MRA_MAX_K_SIZET);
    Dim3 thread_dims = Dim3(max_threads, max_threads, 1);

    CALL_KERNEL(detail::gaxpy_kernel, N, thread_dims, 0, stream,
      (funcA, funcB, funcR, scalarA, scalarB, N, K, key));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_KERNELS_GAXPY_H
