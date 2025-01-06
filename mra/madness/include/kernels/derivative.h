#ifndef MRA_KERNELS_DERIVATIVE_H
#define MRA_KERNELS_DERIVATIVE_H

#include "platform.h"
#include "types.h"
#include "tensorview.h"
#include "domain.h"
#include "key.h"
#include "maxk.h"

namespace mra {
  namespace detail {
    template <typename T, Dimension NDIM>
    DEVSCOPE void derivative_inner(
      const Domain<NDIM>& D,
      const Key<NDIM>& key,
      const TensorView<T, NDIM>& node_left,
      const TensorView<T, NDIM>& node_center,
      const TensorView<T, NDIM>& node_right,
      const TensorView<T, 2>& coeff_left,
      const TensorView<T, 2>& coeff_center,
      const TensorView<T, 2>& coeff_right,
      TensorView<T, NDIM>& deriv,
      size_type axis)
    {
      deriv = transform_dir(node_left, coeff_left, axis);
      deriv += transform_dir(node_center, coeff_center, axis);
      deriv += transform_dir(node_right, coeff_right, axis);

      T scale = std::sqrt(D.template get_reciprocal_width<T>(axis)*std::pow(T(2), T(key.level())));

      deriv *= scale;
    }

    template <typename T, Dimension NDIM>
    DEVSCOPE void derivative_boundary(
      const T* node_left,
      const T* node_center,
      const T* node_right,
      bool axis) // axis to determine left or right boundary
    {
      //diff2b()
    }

    template <typename T, Dimension NDIM>
    DEVSCOPE void derivative_kernel_impl(
      const Domain<NDIM>& D,
      const Key<NDIM>& key,
      const TensorView<T, NDIM>& node_left,
      const TensorView<T, NDIM>& node_center,
      const TensorView<T, NDIM>& node_right,
      const TensorView<T, 2>& coeff_left,
      const TensorView<T, 2>& coeff_center,
      const TensorView<T, 2>& coeff_right,
      TensorView<T, NDIM>& deriv,
      bool is_bdy)
      {
        // if we reached here, all checks have passed, and we do the transform to compute the derivative
        // for a given axis by calling either derivative_inner() or derivative_boundary()
        if (is_bdy){
          // derivative_boundary()
        }
        else{
          derivative_inner<T, NDIM>(D, key, node_left, node_center, node_right,
            coeff_left, coeff_center, coeff_right, deriv, axis);
        }
      }

    template <typename T, Dimension NDIM>
    GLOBALSCOPE void derivative_kernel(
      const Domain<NDIM>& D,
      const Key<NDIM>& key,
      const TensorView<T, NDIM+1>& node_left,
      const TensorView<T, NDIM+1>& node_center,
      const TensorView<T, NDIM+1>& node_right,
      const TensorView<T, 2>& coeff_left,
      const TensorView<T, 2>& coeff_center,
      const TensorView<T, 2>& coeff_right,
      TensorView<T, NDIM+1>& deriv,
      size_type N,
      const bool is_bdy)
    {
      SHARED TensorView<T, NDIM> node_left_view, node_center_view, node_right_view;
      for (size_type blockid = blockIdx.x; blockid < N; blockid += gridDim.x) {
        if (is_team_lead()) {
          node_left_view = node_left(blockid);
          node_center_view = node_center(blockid);
          node_right_view = node_right(blockid);
        }
        SYNCTHREADS();
        derivative_kernel_impl<T, NDIM>(D, node_left_view, node_center_view, node_right_view,
          coeff_left, coeff_center, coeff_right, deriv, is_bdy);
      }
    }

  } // namespace detail

  template <typename T, Dimension NDIM>
  void submit_derivative_kernel(
    const Domain<NDIM>& D,
    const Key<NDIM>& key,
    const TensorView<T, NDIM+1>& node_left,
    const TensorView<T, NDIM+1>& node_center,
    const TensorView<T, NDIM+1>& node_right,
    const TensorView<T, 2>& coeff_left,
    const TensorView<T, 2>& coeff_center,
    const TensorView<T, 2>& coeff_right,
    TensorView<T, NDIM+1>& deriv,
    size_type N,
    size_type K,
    const bool is_bdy,
    ttg::device::Stream stream)
  {
    size_type max_threads = std::min(K, MRA_MAX_K_SIZET);
    Dim3 thread_dims = Dim3(max_threads, max_threads, 1);

    CALL_KERNEL(detail::derivative_kernel, N, thread_dims, 0, stream,
      (D, key, node_left, node_center, node_right, coeff_left, coeff_center, coeff_right,
        deriv, N, is_bdy));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_KERNELS_DERIVATIVE_H
