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
      const T* nodeA, const T* nodeB, T* nodeR,
      const T scalarA, const T scalarB, size_type K)
    {
      const bool is_t0 = 0 == (threadIdx.x + threadIdx.y + threadIdx.z);
      SHARED TensorView<T, NDIM> nR;
      SHARED TensorView<const T, NDIM> nA, nB;
      if (is_t0) {
        nA = TensorView<const T, NDIM>(nodeA, 2*K);
        nB = TensorView<const T, NDIM>(nodeB, 2*K);
        nR = TensorView<T, NDIM>(nodeR, 2*K);
      }
      SYNCTHREADS();

      /* compressed form */
      foreach_idx(nA, [&](size_type i) {
        nR[i] = scalarA*nA[i] + scalarB*nB[i];
      });

    }

    template <typename T, Dimension NDIM>
    GLOBALSCOPE
    LAUNCH_BOUNDS(4*MRA_MAX_K*MRA_MAX_K)
    void gaxpy_kernel(
      const T* nodeA, const T* nodeB, T* nodeR,
      const T scalarA, const T scalarB,
      size_type N, size_type K, const Key<NDIM>& key)
    {
      const size_type TWOK2NDIM = std::pow(2*K, NDIM);
      for (size_type blockid = blockIdx.x; blockid < N; blockid += gridDim.x) {
        gaxpy_kernel_impl<T, NDIM>(nullptr == nodeA ? nullptr : &nodeA[TWOK2NDIM*blockid],
                                   nullptr == nodeB ? nullptr : &nodeB[TWOK2NDIM*blockid],
                                   &nodeR[TWOK2NDIM*blockid],
                                   scalarA, scalarB, K);
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
      (funcA.data(), funcB.data(), funcR.data(), scalarA, scalarB, N, K, key));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_KERNELS_GAXPY_H
