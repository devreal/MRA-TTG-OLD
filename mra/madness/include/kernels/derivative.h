#ifndef MRA_KERNELS_DERIVATIVE_H
#define MRA_KERNELS_DERIVATIVE_H

#include <assert.h>
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
    return 3*K2NDIM + 4*K; // workspace, _result, and deriv_tmp, bv_left, bv_right, bdy_fn, bdy_t
  }
  namespace detail {

    template <typename T, Dimension NDIM>
    SCOPE bool enforce_bc(int bc_left, int bc_right, const Level& n, Translation& l) {
      Translation two2n = 1ul << n;
        if (l < 0){
          if(bc_left == FunctionData<T, NDIM>::BC_ZERO || bc_left == FunctionData<T, NDIM>::BC_FREE ||
              bc_left == FunctionData<T, NDIM>::BC_DIRICHLET || bc_left == FunctionData<T, NDIM>::BC_ZERONEUMANN ||
              bc_left == FunctionData<T, NDIM>::BC_NEUMANN){
            return false;
          }
          else if (bc_left == FunctionData<T, NDIM>::BC_PERIODIC){
            l += two2n;
            assert(bc_left == bc_right);
        }
          else {
            throw std::runtime_error("Invalid boundary condition");
          }
        }
        else if (l >= two2n){
          if(bc_right == FunctionData<T, NDIM>::BC_ZERO || bc_right == FunctionData<T, NDIM>::BC_FREE ||
              bc_right == FunctionData<T, NDIM>::BC_DIRICHLET || bc_right == FunctionData<T, NDIM>::BC_ZERONEUMANN ||
              bc_right == FunctionData<T, NDIM>::BC_NEUMANN){
            return false;
          }
          else if (bc_right == FunctionData<T, NDIM>::BC_PERIODIC){
            l -= two2n;
            assert(bc_left == bc_right);
        }
          else {
            throw std::runtime_error("Invalid boundary condition");
          }
        }
        return true;
    }

    template <Dimension NDIM>
    Key<NDIM> neighbor(
      const Key<NDIM>& key,
      size_type step,
      size_type axis,
      const int bc_left,
      const int bc_right)
    { // TODO: check for boundary conditions to return an invalid key
      std::array<Translation, NDIM> l = key.l;
      l[axis] += step;
      if (!enforce_bc(bc_left, bc_right, key.n, l[axis])){
        return Key<NDIM>::invalid();
      }
      else{
        return Key<NDIM>(key.n, l);
      }
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
      TensorView<T, NDIM>& tmp_result,
      TensorView<T, NDIM>& _result,
      const TensorView<T, 2>& phi,
      const TensorView<T, 2>& phibar,
      const TensorView<T, 2>& quad_x,
      const int bc_left,
      const int bc_right,
      size_type axis,
      size_type K,
      T* workspace)
    {
      tmp_result = T(0);

      parent_to_child(D, key, neighbor(key, -1, axis, bc_left, bc_right),
                       node_left, _result, tmp_result, phibar, phi, quad_x, K, workspace);
      parent_to_child(D, key, key, node_center, _result, tmp_result, phibar, phi, quad_x, K, workspace);
      parent_to_child(D, key, neighbor(key, 1, axis, bc_left, bc_right),
                        node_right, _result, tmp_result, phibar, phi, quad_x, K, workspace);
      deriv = 0;

      transform_dir(node_left, operators(FunctionData<T, NDIM>::DerivOp::RP), _result, deriv, axis);
      transform_dir(node_center, operators(FunctionData<T, NDIM>::DerivOP::R0), _result, deriv, axis);
      transform_dir(node_right, operators(FunctionData<T, NDIM>::DerivOp::RM), _result, deriv, axis);

      T scale = std::sqrt(D.template get_reciprocal_width<T>(axis)*std::pow(T(2), T(key.level())));
      T thresh = T(1e-12);
      deriv *= scale;
      deriv.reduce_rank(thresh);
    }

    template <typename T, Dimension NDIM>
    DEVSCOPE void derivative_boundary(
      const Domain<NDIM>& D,
      const Key<NDIM>& key,
      const T* node_left,
      const T* node_center,
      const T* node_right,
      const TensorView<T, 2+1>& operators,
      TensorView<T, NDIM>& deriv,
      TensorView<T, NDIM>& tmp_result,
      TensorView<T, NDIM>& _result,
      const TensorView<T, 2>& phi,
      const TensorView<T, 2>& phibar,
      const TensorView<T, 2>& quad_x,
      const T g1,
      const T g2,
      const int bdy,
      const int bc_left,
      const int bc_right,
      size_type axis,
      size_type K,
      T* workspace)
    {
      std::array<Translation, NDIM> l = key.l;
      if (l[axis] == 0){
        tmp_result = T(0);
        parent_to_child(D, key, neighbor(key, -1, axis), node_right, _result, tmp_result, phibar, phi, quad_x, K, workspace);
        parent_to_child(D, key, key, node_center, _result, tmp_result, phibar, phi, quad_x, K, workspace);

        deriv = 0;
        transform_dir(node_right, operators(FunctionData<T, NDIM>::DerivOp::RMT), _result, deriv, axis);
        transform_dir(node_center, operators(FunctionData<T, NDIM>::DerivOp::R0T), _result, deriv, axis);
      }
      else {
        tmp_result = T(0);
        parent_to_child(D, key, key, node_center, _result, tmp_result, phibar, phi, quad_x, K, workspace);
        parent_to_child(D, key, neighbor(key, 1, axis), node_left, _result, tmp_result, phibar, phi, quad_x, K, workspace);

        deriv = 0;
        transform_dir(node_center, operators(FunctionData<T, NDIM>::DerivOp::RIGHT_R0T), _result, deriv, axis);
        transform_dir(node_left, operators(FunctionData<T, NDIM>::DerivOp::RIGHT_RPT), _result, deriv, axis);
      }

      T scale = std::sqrt(D.template get_reciprocal_width<T>(axis)*std::pow(T(2), T(key.level())));
      T thresh = T(1e-12);
      deriv *= scale;
      deriv.reduce_rank(thresh);

    }

    template <typename T, Dimension NDIM>
    DEVSCOPE void derivative_kernel_impl(
      const Domain<NDIM>& D,
      const Key<NDIM>& key,
      const TensorView<T, NDIM>& node_left,
      const TensorView<T, NDIM>& node_center,
      const TensorView<T, NDIM>& node_right,
      const TensorView<T, 3>& operators,
      TensorView<T, NDIM>& deriv,
      const TensorView<T, 2>& phi,
      const TensorView<T, 2>& phibar,
      const TensorView<T, 2>& quad_x,
      T* tmp,
      size_type K,
      const T g1,
      const T g2,
      size_type axis,
      const bool is_bdy,
      const int bc_left,
      const int bc_right)
      {
        // if we reached here, all checks have passed, and we do the transform to compute the derivative
        // for a given axis by calling either derivative_inner() or derivative_boundary()
        SHARED TensorView<T, NDIM> deriv_tmp, _result;
        SHARED T* workspace;

        size_type blockId = blockIdx.x;
        T* block_tmp_ptr = &tmp[blockId*derivative_tmp_size<NDIM>(K)];
        const size_type K2NDIM = std::pow(K, NDIM);
        if(is_team_lead()){
          deriv_tmp = TensorView<T, NDIM>(&block_tmp_ptr[       0], K);
          _result   = TensorView<T, NDIM>(&block_tmp_ptr[1*K2NDIM], K);
          workspace = &block_tmp_ptr[2*K2NDIM];
        }
        SYNCTHREADS();

        if (is_bdy){
          derivative_boundary<T, NDIM>(D, key, node_left, node_center, node_right,
            operators, deriv, deriv_tmp, _result, phi, phibar, quad_x, g1, g2, bc_left, bc_right, axis, K, workspace);
        }
        else{
          derivative_inner<T, NDIM>(D, key, node_left, node_center, node_right,
            operators, deriv, deriv_tmp, _result, phi, phibar, quad_x, bc_left, bc_right, axis, K, workspace);
        }
      }

    template <typename T, Dimension NDIM>
    GLOBALSCOPE void derivative_kernel(
      const Domain<NDIM>& D,
      const Key<NDIM>& key,
      const TensorView<T, NDIM+1> node_left,
      const TensorView<T, NDIM+1> node_center,
      const TensorView<T, NDIM+1> node_right,
      const TensorView<T, 3> operators,
      TensorView<T, NDIM+1> deriv,
      const TensorView<T, 2> phi,
      const TensorView<T, 2> phibar,
      const TensorView<T, 2> quad_x,
      T* tmp,
      size_type N,
      size_type K,
      const T g1,
      const T g2,
      size_type axis,
      const bool is_bdy,
      const int bc_left,
      const int bc_right)
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
          operators, deriv, phi, phibar, quad_x, tmp, K, g1, g2, axis, is_bdy, bc_left, bc_right);
      }
    }

  } // namespace detail

  template <typename Fn, typename T, Dimension NDIM>
  void submit_derivative_kernel(
    const Domain<NDIM>& D,
    const Key<NDIM>& key,
    const TensorView<T, NDIM+1>& node_left,
    const TensorView<T, NDIM+1>& node_center,
    const TensorView<T, NDIM+1>& node_right,
    const TensorView<T, 3>& operators,
    TensorView<T, NDIM+1>& deriv,
    const TensorView<T, 2>& phi,
    const TensorView<T, 2>& phibar,
    const TensorView<T, 2>& quad_x,
    T* tmp,
    size_type N,
    size_type K,
    const T g1,
    const T g2,
    size_type axis,
    const bool is_bdy,
    const int bc_left,
    const int bc_right,
    ttg::device::Stream stream)
  {
    size_type max_threads = std::min(K, MRA_MAX_K_SIZET);
    Dim3 thread_dims = Dim3(max_threads, max_threads, 1);

    CALL_KERNEL(detail::derivative_kernel, N, thread_dims, 0, stream,
      (D, key, node_left, node_center, node_right, operators,
        deriv, phi, phibar, quad_x, tmp, N, K, g1, g2, axis, is_bdy, bc_left, bc_right));
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

    template <typename T, Dimension NDIM>
    void init_bv(
      size_type K,
      const int bc_left,
      const int bc_right,
      TensorView<T, 1> bv_left,
      TensorView<T, 1> bv_right)
      {
        T iphase = 1;
        foreach_idx(bv_left, [&](size_type i){
          iphase = -iphase;
          if (bc_left == FunctionData<T, NDIM>::BC_DIRICHLET){
            bv_left[i] = iphase*sqrt(T(2*i+1));
          }
          else if (bc_left == FunctionData<T, NDIM>::BC_NEUMANN){
            bv_left[i] = -iphase*sqrt(T(2*i+1))/pow(K, T(2));;
          }
          else {
            bv_left[i] = 0;
          }
          if (bc_right == FunctionData<T, NDIM>::BC_DIRICHLET){
            bv_right[i] = iphase*sqrt(T(2*i+1));
          }
          else if (bc_right == FunctionData<T, NDIM>::BC_NEUMANN){
            bv_right[i] = -iphase*sqrt(T(2*i+1))/pow(K, T(2));;
          }
          else {
            bv_right[i] = 0;
          }
        });
      }

} // namespace mra

#endif // MRA_KERNELS_DERIVATIVE_H
