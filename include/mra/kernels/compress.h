#ifndef MRA_KERNELS_COMPRESS_H
#define MRA_KERNELS_COMPRESS_H

#include "mra/platform.h"
#include "mra/types.h"
#include "mra/key.h"
#include "mra/maxk.h"
#include "mra/tensorview.h"
#include "mra/kernels/child_slice.h"
#include "mra/kernels.h"

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
      TensorView<T, NDIM>& p,
      TensorView<T, NDIM>& d,
      const TensorView<T, 2>& hgT,
      TensorView<T,NDIM>& s,
      T* workspace,
      T* d_sumsq,
      const std::array<TensorView<T, NDIM>, Key<NDIM>::num_children()>& in_views)
    {
      d = 0.0;
      p = 0.0;

      for (int i = 0; i < Key<NDIM>::num_children(); ++i) {
        auto child_slice = get_child_slice<NDIM>(key, K, i);
        const TensorView<T, NDIM>& in = in_views[i];
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

    template<typename T, Dimension NDIM>
    LAUNCH_BOUNDS(max_threads(2*MRA_MAX_K))
    GLOBALSCOPE void compress_kernel(
      Key<NDIM> key,
      size_type N,
      size_type K,
      TensorView<T, NDIM+1> p_in,
      TensorView<T, NDIM+1> result_in,
      const TensorView<T, 2> hgT,
      T* tmp,
      T* d_sumsq,
      const std::array<TensorView<T, NDIM+1>, Key<NDIM>::num_children()> in_views)
    {
      const bool is_t0 = (0 == thread_id());
      const size_type K2NDIM    = std::pow(  K,NDIM);
      const size_type TWOK2NDIM = std::pow(2*K,NDIM);
      SHARED std::array<TensorView<T, NDIM>, Key<NDIM>::num_children()> block_in_views;
      SHARED T* workspace;
      SHARED TensorView<T,NDIM> s, p, d;
      int blockId = blockIdx.x;
      T* block_tmp = &tmp[blockId*compress_tmp_size<NDIM>(K)];

      if (is_t0) {
        s = TensorView<T,NDIM>(&block_tmp[0], 2*K);
        workspace = &block_tmp[TWOK2NDIM];
      }

      for (size_type fnid = blockId; fnid < N; fnid += gridDim.x) {
        /* no need to sync threads here */
        if (is_t0) {
          for (int i = 0; i < Key<NDIM>::num_children(); ++i) {
            block_in_views[i] = in_views[i](fnid);
          }
          p = p_in(fnid);
          d = result_in(fnid);
        }
        SYNCTHREADS();

        compress_kernel_impl(key, K, p, d, hgT, s, workspace,
                             &d_sumsq[fnid], block_in_views);
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
    const std::array<TensorView<T, NDIM+1>, Key<NDIM>::num_children()>& in_views,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);
    size_type numthreads = thread_dims.x*thread_dims.y*thread_dims.z;

    CALL_KERNEL(detail::compress_kernel, N, thread_dims, numthreads*sizeof(T), stream,
      (key, N, K, p_view, result_view, hgT_view, tmp, d_sumsq, in_views));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_KERNELS_COMPRESS_H
