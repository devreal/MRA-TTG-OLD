#ifndef MRA_KERNELS_COMPRESS_H
#define MRA_KERNELS_COMPRESS_H

#include "platform.h"
#include "types.h"
#include "key.h"
#include "tensorview.h"
#include "kernels/child_slice.h"
#include "kernels/transform.h"
#include "maxk.h"

#include <array>

/**
 * Compress kernels
 */

namespace mra {

  template<mra::Dimension NDIM>
  SCOPE size_type compress_tmp_size(size_type K) {
    const size_type TWOK2NDIM = std::pow(2*K,NDIM);
    return (2*TWOK2NDIM); // s & workspace
  }

  namespace detail {

    template<typename T, Dimension NDIM>
    DEVSCOPE void compress_kernel_impl(
      Key<NDIM> key,
      size_type K,
      T* p_ptr,
      T* result_ptr,
      const T* hgT_ptr,
      T* tmp,
      T* d_sumsq,
      const std::array<const T*, Key<NDIM>::num_children()>& in_ptrs)
    {
      const bool is_t0 = (0 == thread_id());
      {   // Collect child coeffs and leaf info
        /* construct tensors */
        const size_type K2NDIM    = std::pow(  K,NDIM);
        const size_type TWOK2NDIM = std::pow(2*K,NDIM);
        SHARED TensorView<T,NDIM> s, d, p;
        SHARED TensorView<T,2> hgT;
        SHARED T* workspace;
        if (is_t0) {
          s = TensorView<T,NDIM>(&tmp[0], 2*K);
          workspace = &tmp[TWOK2NDIM];
          d = TensorView<T,NDIM>(result_ptr, 2*K);
          p = TensorView<T,NDIM>(p_ptr, K);
          hgT = TensorView<T,2>(hgT_ptr, 2*K);
        }
        SYNCTHREADS();
        d = 0.0;
        p = 0.0;

        for (int i = 0; i < Key<NDIM>::num_children(); ++i) {
          auto child_slice = get_child_slice<NDIM>(key, K, i);
          const TensorView<T, NDIM> in(in_ptrs[i], K);
          s(child_slice) = in;
        }
        //filter<T,K,NDIM>(s,d);  // Apply twoscale transformation
        transform<NDIM>(s, hgT, d, workspace);
        if (key.level() > 0) {
          auto child_slice = get_child_slice<NDIM>(key, K, 0);
          p = d(child_slice);
          d(child_slice) = 0.0;
        }
        sumabssq(d, d_sumsq);
      }
    }

    template<typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MRA_MAX_K*MRA_MAX_K)
    GLOBALSCOPE void compress_kernel(
      Key<NDIM> key,
      size_type N,
      size_type K,
      TensorView<T, NDIM+1>& p_ptr,
      TensorView<T, NDIM+1>& result_ptr,
      const TensorView<T, 2>& hgT_ptr,
      T* tmp,
      T* d_sumsq,
      const std::array<const T*, Key<NDIM>::num_children()> in_ptrs)
    {
      const bool is_t0 = (0 == thread_id());
      const size_type K2NDIM    = std::pow(  K,NDIM);
      const size_type TWOK2NDIM = std::pow(2*K,NDIM);
      SHARED std::array<const T*, Key<NDIM>::num_children()> block_in_ptrs;
      int blockId = blockIdx.x;

      if (is_t0) {
        for (int i = 0; i < Key<NDIM>::num_children(); ++i) {
          block_in_ptrs[i] = (nullptr != in_ptrs[i]) ? &in_ptrs[i][K2NDIM*blockId] : nullptr;
        }
      }

      SHARED TensorView<T, NDIM> p, result;
      for (size_type blockid; blockid < N; blockid += gridDim.x){
        if (is_team_lead()){
          p = p_ptr(blockid);
          result = result_ptr(blockid);
        }
      SYNCTHREADS();
        /* no need to sync threads here */
      compress_kernel_impl(key, K, p.data(), result.data(), hgT_ptr.data(), &tmp[compress_tmp_size<NDIM>(K)*blockId],
                           &d_sumsq[blockId], block_in_ptrs);
      }

    }
  } // namespace detail

  template<typename T, Dimension NDIM>
  void submit_compress_kernel(
    const Key<NDIM>& key,
    size_type N,
    size_type K,
    TensorView<T, NDIM+1>& p_view,
    TensorView<T, NDIM+1>& result_view,
    const TensorView<T, 2>& hgT_view,
    T* tmp,
    T* d_sumsq,
    const std::array<const T*, Key<NDIM>::num_children()>& in_ptrs,
    cudaStream_t stream)
  {
    size_type max_threads = std::min(K, MRA_MAX_K_SIZET);
    Dim3 thread_dims = Dim3(max_threads, max_threads, 1);
    size_type numthreads = thread_dims.x*thread_dims.y*thread_dims.z;

    CALL_KERNEL(detail::compress_kernel, N, thread_dims, numthreads*sizeof(T), stream,
      (key, N, K, p_view, result_view, hgT_view, tmp, d_sumsq, in_ptrs));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_KERNELS_COMPRESS_H
