#ifndef MRA_KERNELS_RECONSTRUCT_H
#define MRA_KERNELS_RECONSTRUCT_H


#include "types.h"
#include "key.h"
#include "maxk.h"
#include "tensorview.h"
#include "platform.h"
#include "kernels/child_slice.h"
#include "kernels/transform.h"

namespace mra {

  template<mra::Dimension NDIM>
  SCOPE size_type reconstruct_tmp_size(size_type K) {
    const size_type TWOK2NDIM = std::pow(2*K,NDIM);
    return 3*TWOK2NDIM; // s, tmp_node & workspace
  }

  namespace detail {

    /**
     * kernel for reconstruct
     */

    template<typename T, Dimension NDIM>
    DEVSCOPE void reconstruct_kernel_impl(
      Key<NDIM> key,
      size_type K,
      const TensorView<T, NDIM>& node,
      T* tmp_ptr,
      const TensorView<T, 2>& hg,
      const TensorView<T, NDIM>& from_parent,
      std::array<T*, Key<NDIM>::num_children()>& r_arr)
    {
      const bool is_t0 = (0 == thread_id());
      const size_type TWOK2NDIM = std::pow(2*K,NDIM);

      SHARED TensorView<T, NDIM> s, tmp_node;
      SHARED T* workspace;
      if (is_t0) {
        s           = TensorView<T, NDIM>(&tmp_ptr[0], 2*K);
        tmp_node    = TensorView<T, NDIM>(&tmp_ptr[1*TWOK2NDIM], 2*K);
        s           = TensorView<T, NDIM>(&tmp_ptr[0], 2*K);
        workspace   = &tmp_ptr[TWOK2NDIM];
      }
      SYNCTHREADS();
      s = 0.0;

      tmp_node = node;
      auto child_slice = get_child_slice<NDIM>(key, K, 0);
      if (key.level() != 0) tmp_node(child_slice) = from_parent;

      //unfilter<T,K,NDIM>(node.get().coeffs, s);
      transform<NDIM>(tmp_node, hg, s, workspace);

      /* extract all r from s
      * NOTE: we could do this on 1<<NDIM blocks but the benefits would likely be small */
      for (size_type i = 0; i < key.num_children(); ++i) {
        auto child_slice = get_child_slice<NDIM>(key, K, i);
        /* tmp layout: 2K^NDIM for s, K^NDIM for workspace, [K^NDIM]* for r fields */
        auto r = TensorView<T, NDIM>(r_arr[i], K);
        r = s(child_slice);
      }
    }


    template<typename T, Dimension NDIM>
    GLOBALSCOPE void
    LAUNCH_BOUNDS(MRA_MAX_K*MRA_MAX_K)
    reconstruct_kernel(
      Key<NDIM> key,
      size_type N,
      size_type K,
      TensorView<T, NDIM+1>& node_view,
      T* tmp_ptr,
      const TensorView<T, 2>& hg,
      const TensorView<T, NDIM+1>& from_parent_view,
      std::array<T*, Key<NDIM>::num_children()> r_arr)
    {
      const bool is_t0 = (0 == thread_id());
      const size_type TWOK2NDIM = std::pow(2*K,NDIM);
      const size_type K2NDIM    = std::pow(  K,NDIM);

      /* pick the r's for this function */
      SHARED std::array<T*, Key<NDIM>::num_children()> block_r_arr;
      size_type blockId = blockIdx.x;
      if (is_t0) {
        for (size_type i = 0; i < Key<NDIM>::num_children(); ++i) {
          block_r_arr[i] = &r_arr[i][K2NDIM*blockId];
        }
      }

      SHARED TensorView<T, NDIM> node, from_parent;
      for (size_type blockid; blockid < N; blockid += gridDim.x){
        if(is_team_lead()){
          node = node_view(blockid);
          from_parent = from_parent_view(blockid);
        }
        SYNCTHREADS();
        reconstruct_kernel_impl(key, K, node, tmp_ptr + blockId*reconstruct_tmp_size<NDIM>(K),
                              hg, from_parent, block_r_arr);
      }
    }
  } // namespace detail

  template<typename T, Dimension NDIM>
  void submit_reconstruct_kernel(
    const Key<NDIM>& key,
    size_type N,
    size_type K,
    TensorView<T, NDIM+1>& node,
    const TensorView<T, 2>& hg,
    const TensorView<T, NDIM+1>& from_parent,
    const std::array<T*, mra::Key<NDIM>::num_children()>& r_arr,
    T* tmp,
    cudaStream_t stream)
  {
    size_type max_threads = std::min(K, MRA_MAX_K_SIZET);
    Dim3 thread_dims = Dim3(max_threads, max_threads, 1);
    CALL_KERNEL(detail::reconstruct_kernel, N, thread_dims, 0, stream,
      (key, N, K, node, tmp, hg, from_parent, r_arr));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_KERNELS_RECONSTRUCT_H
