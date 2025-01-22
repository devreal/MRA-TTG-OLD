#ifndef MRA_KERNELS_DERIVATIVE_H
#define MRA_KERNELS_DERIVATIVE_H

#include "platform.h"
#include "types.h"
#include "tensorview.h"
#include "domain.h"
#include "key.h"
#include "maxk.h"
#include "transform.h"
#include "functiondata.h"

namespace mra {

  template<mra::Dimension NDIM>
  SCOPE size_type derivative_tmp_size(size_type K) {
    const size_type K2NDIM = std::pow(K,NDIM);
    return 4*K2NDIM; // TODO: fix this
  }
  namespace detail {

    template <Dimension NDIM>
    Key<NDIM> neighbor(
      const Key<NDIM>& key,
      size_type step,
      size_type axis)
    { // TODO: check for boundary conditions to return an invalid key
      std::array<Translation, NDIM> l = key.l;
      l[axis] += step;
      return Key<NDIM>(key.n, l);
    }

    template <typename T, Dimension NDIM>
    DEVSCOPE void derivative_inner(
      const Domain<NDIM>& D,
      const Key<NDIM>& key,
      const TensorView<T, NDIM>& node_left,
      const TensorView<T, NDIM>& node_center,
      const TensorView<T, NDIM>& node_right,
      const TensorView<T, 3>& operators,
      TensorView<T, NDIM>& deriv,
      size_type axis,
      size_type K,
      T* tmp)
    {
      TensorView<T, NDIM> tmp_result = TensorView<T, NDIM>(&tmp[0], K, K, K);
      deriv = 0;

      transform_dir(node_left, operators(FunctionData<T, NDIM>::DerivOp::RM), tmp_result, deriv, axis);
      transform_dir(node_center, operators(FunctionData<T, NDIM>::DerivOP::R0), tmp_result, deriv, axis);
      transform_dir(node_right, operators(FunctionData<T, NDIM>::DerivOp::RP), tmp_result, deriv, axis);

      T scale = std::sqrt(D.template get_reciprocal_width<T>(axis)*std::pow(T(2), T(key.level())));

      deriv *= scale;
    }

    template <typename T, Dimension NDIM, class G1, class G2>
    DEVSCOPE void derivative_boundary(
      const T* node_left,
      const T* node_center,
      const T* node_right,
      const G1& g1,
      const G2& g2,
      size_type axis) // axis to determine left or right boundary
    {
      //diff2b()
    }

    template <typename T, Dimension NDIM, class G1, class G2>
    DEVSCOPE void derivative_kernel_impl(
      const Domain<NDIM>& D,
      const Key<NDIM>& key,
      const TensorView<T, NDIM>& node_left,
      const TensorView<T, NDIM>& node_center,
      const TensorView<T, NDIM>& node_right,
      const TensorView<T, 3>& operators,
      TensorView<T, NDIM>& deriv,
      T* tmp,
      size_type K,
      const G1& g1,
      const G2& g2,
      size_type axis,
      bool is_bdy)
      {
        // if we reached here, all checks have passed, and we do the transform to compute the derivative
        // for a given axis by calling either derivative_inner() or derivative_boundary()
        if (is_bdy){
          // derivative_boundary()
        }
        else{
          derivative_inner<T, NDIM>(D, key, node_left, node_center, node_right,
            operators, deriv, axis, K, tmp);
        }
      }

    template <typename T, Dimension NDIM, class G1, class G2>
    GLOBALSCOPE void derivative_kernel(
      const Domain<NDIM>& D,
      const Key<NDIM>& key,
      const TensorView<T, NDIM+1> node_left,
      const TensorView<T, NDIM+1> node_center,
      const TensorView<T, NDIM+1> node_right,
      const TensorView<T, 3> operators,
      TensorView<T, NDIM+1> deriv,
      T* tmp,
      size_type N,
      size_type K,
      const G1 g1,
      const G2 g2,
      size_type axis,
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
          operators, deriv, tmp, K, g1, g2, axis, is_bdy);
      }
    }

  } // namespace detail

  template <typename T, Dimension NDIM, class G1, class G2>
  void submit_derivative_kernel(
    const Domain<NDIM>& D,
    const Key<NDIM>& key,
    const TensorView<T, NDIM+1>& node_left,
    const TensorView<T, NDIM+1>& node_center,
    const TensorView<T, NDIM+1>& node_right,
    const TensorView<T, 3>& operators,
    TensorView<T, NDIM+1>& deriv,
    T* tmp,
    size_type N,
    size_type K,
    const G1& g1,
    const G2& g2,
    size_type axis,
    const bool is_bdy,
    ttg::device::Stream stream)
  {
    size_type max_threads = std::min(K, MRA_MAX_K_SIZET);
    Dim3 thread_dims = Dim3(max_threads, max_threads, 1);

    CALL_KERNEL(detail::derivative_kernel, N, thread_dims, 0, stream,
      (D, key, node_left, node_center, node_right, operators,
        deriv, tmp, N, K, axis, is_bdy));
    checkSubmit();
  }

  template <typename T, Dimension NDIM>
  void parent_to_child(
    const Domain<NDIM>& D,
    const Key<NDIM>& parent,
    const Key<NDIM>& child,
    const TensorView<T, NDIM>& coeffs,
    TensorView<T, NDIM>& result,
    TensorView<T, NDIM>& result_tmp,
    const TensorView<T, 2>& phibar,
    TensorView<T, NDIM>& phi,
    const T* quad_x,
    const size_type K,
    T* tmp)
    {
      if (parent == child || parent.is_invalid() || child.is_invalid()) result = coeffs;

      fcube_for_mul(D, child, parent, coeffs, result_tmp, phibar, phi, quad_x, K, tmp);
      T scale = std::sqrt(D.template get_volume<T>()*std::pow(T(0.5), T(NDIM*child.level())));
      result_tmp *= scale;
      transform(result_tmp, phibar, result, tmp);
    }

} // namespace mra

#endif // MRA_KERNELS_DERIVATIVE_H
