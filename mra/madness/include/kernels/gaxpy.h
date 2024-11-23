#ifndef MRA_KERNELS_GAXPY_H
#define MRA_KERNELS_GAXPY_H

#include "platform.h"
#include "types.h"
#include "tensorview.h"
#include "key.h"

namespace mra {
  namespace detail {
    template <typename T, Dimension NDIM>
    DEVSCOPE void gaxpy_kernel_impl(
      const T* nodeA, const T* nodeB, T* nodeR,
      const T scalarA, const T scalarB, size_type K)
    {
      const bool is_t0 = 0 == (threadIdx.x + threadIdx.y + threadIdx.z);
      SHARED TensorView<T, NDIM> nA, nB, nR;
      if (is_t0) {
        nA = TensorView<T, NDIM>(nodeA, K);
        nB = TensorView<T, NDIM>(nodeB, K);
        nR = TensorView<T, NDIM>(nodeR, K);
      }
      SYNCTHREADS();

      /* compressed form */
      foreach_idx(nA, [&](auto... idx) {
        nR(idx...) = scalarA*nA(idx...) + scalarB*nB(idx...);
      });

    }

    template <typename T, Dimension NDIM>
    GLOBALSCOPE void gaxpy_kernel(
      const T* nodeA, const T* nodeB, T* nodeR,
      const T scalarA, const T scalarB,
      size_type N, size_type K, const Key<NDIM>& key)
    {
      const size_type TWOK2NDIM = std::pow(2*K, NDIM);
      size_type blockid = blockIdx.x;
      gaxpy_kernel_impl<T, NDIM>(nullptr == nodeA ? nullptr : &nodeA[TWOK2NDIM*blockid],
                                 nullptr == nodeB ? nullptr : &nodeB[TWOK2NDIM*blockid],
                                 &nodeR[TWOK2NDIM*blockid],
                                 scalarA, scalarB, K);
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
    Dim3 thread_dims = Dim3(K, K, 1);

    CALL_KERNEL(detail::gaxpy_kernel, N, thread_dims, 0, stream,
      (funcA.data(), funcB.data(), funcR.data(), scalarA, scalarB, N, K, key));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_KERNELS_GAXPY_H
