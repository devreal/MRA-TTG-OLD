#ifndef MRA_KERNELS_MULTIPLY_H
#define MRA_KERNELS_MULTIPLY_H

#include "platform.h"
#include "types.h"
#include "tensorview.h"
#include "maxk.h"


namespace mra {

  template<mra::Dimension NDIM>
  SCOPE size_type multiply_tmp_size(size_type K) {
    const size_type K2NDIM = std::pow(K,NDIM);
    return 4*K2NDIM; // workspace, r, r1, and r2
  }

  namespace detail {

    template <typename T, Dimension NDIM>
    DEVSCOPE void multiply_kernel_impl(
      const Domain<NDIM>& D,
      const T* nodeA,
      const T* nodeB,
      T* nodeR,
      T* tmp,
      const T* phiT_ptr,
      const T* phibar_ptr,
      Key<NDIM> key,
      size_type K)
    {
      const bool is_t0 = (0 == thread_id());
      const size_type K2NDIM = std::pow(K, NDIM);

      SHARED TensorView<T, NDIM> nA, nB, nR, r1, r2, r;
      SHARED TensorView<T, 2> phiT, phibar;
      SHARED T* workspace;

      if (is_t0) {
        nA        = TensorView<T, NDIM>(nodeA, K);
        nB        = TensorView<T, NDIM>(nodeB, K);
        nR        = TensorView<T, NDIM>(nodeR, K);
        workspace = &tmp[0];
        r         = TensorView<T, NDIM>(&tmp[1*K2NDIM], K);
        r1        = TensorView<T, NDIM>(&tmp[2*K2NDIM], K);
        r2        = TensorView<T, NDIM>(&tmp[3*K2NDIM], K);
        phiT      = TensorView<T, 2>(phiT_ptr, K, K);
        phibar    = TensorView<T, 2>(phibar_ptr, K, K);
      }
      SYNCTHREADS();

      // convert coeffs to function values
      transform(nA, phiT, r1, workspace);
      transform(nB, phiT, r2, workspace);
      const T scale = std::pow(T(2), T(0.5 * NDIM * key.level())) / std::sqrt(D.template get_volume<T>());

      foreach_idx(nA, [&](size_type i) {
          r[i] = scale * r1[i] * r2[i];
      });

      // convert back to coeffs
      transform(r, phibar, nR, workspace);
    }

    template <typename T, Dimension NDIM>
    GLOBALSCOPE void
    LAUNCH_BOUNDS(MRA_MAX_K*MRA_MAX_K)
    multiply_kernel(
      const Domain<NDIM>& D,
      const T* nodeA,
      const T* nodeB,
      T* nodeR,
      T* tmp,
      const T* phiT_ptr,
      const T* phibar_ptr,
      Key<NDIM> key,
      size_type N,
      size_type K)
    {
      const size_type K2NDIM = std::pow(K, NDIM);
      /* adjust pointers for the function of each block */
      size_type blockid = blockIdx.x;

      multiply_kernel_impl<T, NDIM>(D, nullptr == nodeA ? nullptr : &nodeA[K2NDIM*blockid],
                                    nullptr == nodeB ? nullptr : &nodeB[K2NDIM*blockid],
                                    &nodeR[K2NDIM*blockid], &tmp[(multiply_tmp_size<NDIM>(K)*blockid)],
                                    phiT_ptr, phibar_ptr, key, K);
    }
  } // namespace detail

  template <typename T, Dimension NDIM>
  void submit_multiply_kernel(
    const Domain<NDIM>& D,
    const TensorView<T, NDIM+1>& funcA,
    const TensorView<T, NDIM+1>& funcB,
    TensorView<T, NDIM+1>& funcR,
    const TensorView<T, 2>& phiT,
    const TensorView<T, 2>& phibar,
    size_type N,
    size_type K,
    const Key<NDIM>& key,
    T* tmp,
    ttg::device::Stream stream)
  {
    size_type max_threads = std::min(K, MRA_MAX_K_SIZET);
    Dim3 thread_dims = Dim3(max_threads, max_threads, 1);

    CALL_KERNEL(detail::multiply_kernel, N, thread_dims, 0, stream,
      (D, funcA.data(), funcB.data(), funcR.data(), tmp,
        phiT.data(), phibar.data(), key, N, K));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_KERNELS_MULTIPLY_H
