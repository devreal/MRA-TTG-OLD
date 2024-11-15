#ifndef MRA_KERNELS_RECONSTRUCT_H
#define MRA_KERNELS_RECONSTRUCT_H


#include "types.h"
#include "key.h"
#include "tensorview.h"
#include "platform.h"
#include "kernels/child_slice.h"
#include "kernels/transform.h"

namespace mra {

  template<mra::Dimension NDIM>
  SCOPE size_type reconstruct_tmp_size(size_type K) {
    const size_type TWOK2NDIM = std::pow(2*K,NDIM); // s & workspace
    return 2*TWOK2NDIM;
  }

  namespace detail {

    /**
     * kernel for reconstruct
     */

    template<typename T, Dimension NDIM>
    DEVSCOPE void reconstruct_kernel_impl(
      Key<NDIM> key,
      size_type K,
      T* node_ptr,
      bool node_empty,
      T* tmp_ptr,
      const T* hg_ptr,
      const T* from_parent_ptr,
      std::array<T*, Key<NDIM>::num_children()>& r_arr)
    {
      const bool is_t0 = (0 == (threadIdx.x + threadIdx.y + threadIdx.z));
      const size_type TWOK2NDIM = std::pow(2*K,NDIM);
      SHARED TensorView<T, NDIM> node, s, workspace, from_parent;
      SHARED TensorView<T, 2> hg;
      if (is_t0) {
        node        = TensorView<T, NDIM>(node_ptr, 2*K);
        s           = TensorView<T, NDIM>(&tmp_ptr[0], 2*K);
        workspace   = TensorView<T, NDIM>(&tmp_ptr[TWOK2NDIM], 2*K);
        hg          = TensorView<T, 2>(hg_ptr, 2*K);
        from_parent = TensorView<T, NDIM>(from_parent_ptr, K);
      }
      SYNCTHREADS();
      s = 0.0;

      if (node_empty) {
        /* if the node was empty we reset it to zero */
        node = 0.0;
      }

      auto child_slice = get_child_slice<NDIM>(key, K, 0);
      if (key.level() != 0) node(child_slice) = from_parent;

      //unfilter<T,K,NDIM>(node.get().coeffs, s);
      transform<NDIM>(node, hg, s, workspace);

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
    GLOBALSCOPE void reconstruct_kernel(
      Key<NDIM> key,
      size_type N,
      size_type K,
      T* node_ptr,
      bool node_empty,
      T* tmp_ptr,
      const T* hg_ptr,
      const T* from_parent_ptr,
      std::array<T*, Key<NDIM>::num_children()> r_arr)
    {
      const bool is_t0 = (0 == (threadIdx.x + threadIdx.y + threadIdx.z));
      const size_type TWOK2NDIM = std::pow(2*K,NDIM);
      const size_type K2NDIM    = std::pow(  K,NDIM);

      /* pick the r's for this function */
      SHARED std::array<T*, Key<NDIM>::num_children()> block_r_arr;
      size_type blockid = blockIdx.x;
      if (is_t0) {
        for (size_type i = 0; i < Key<NDIM>::num_children(); ++i) {
          block_r_arr[i] = &r_arr[i][K2NDIM*blockid];
        }
      }
      /* no need to sync threads here, the impl will sync before the r_arr are used */
      reconstruct_kernel_impl(key, K, &node_ptr[TWOK2NDIM*blockid], node_empty,
                              tmp_ptr + blockid*reconstruct_tmp_size<NDIM>(K),
                              hg_ptr, &from_parent_ptr[K2NDIM*blockid],
                              block_r_arr);
    }
  } // namespace detail

  template<typename T, Dimension NDIM>
  void submit_reconstruct_kernel(
    const Key<NDIM>& key,
    size_type N,
    size_type K,
    TensorView<T, NDIM+1>& node,
    bool node_empty,
    const TensorView<T, 2>& hg,
    const TensorView<T, NDIM+1>& from_parent,
    const std::array<T*, mra::Key<NDIM>::num_children()>& r_arr,
    T* tmp,
    cudaStream_t stream)
  {
    /* runs on a single block */
    Dim3 thread_dims = Dim3(K, K, 1); // figure out how to consider register usage
    CALL_KERNEL(detail::reconstruct_kernel, N, thread_dims, 0, stream,
      (key, N, K, node.data(), node_empty, tmp, hg.data(), from_parent.data(), r_arr));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_KERNELS_RECONSTRUCT_H
