#ifndef HAVE_KERNELS_H
#define HAVE_KERNELS_H

#include <cstddef>
#include "gaussian.h"
#include "tensorview.h"
#include "key.h"
#include "domain.h"
#include "types.h"
#include "ttg/device/device.h"

/* Returns the total size of temporary memory needed for
 * the project() kernel. */
template<mra::Dimension NDIM>
SCOPE std::size_t project_tmp_size(std::size_t K) {
  const size_t K2NDIM = std::pow(K,NDIM);
  const std::size_t TWOK2NDIM = std::pow(2*K, NDIM);
  return (3*TWOK2NDIM) // workspace, values and r
       + (NDIM*K2NDIM) // xvec in fcube
       + (NDIM*K)      // x in fcube
       + (3*K2NDIM);   // workspace in transform, child_values, r
}

/**
 * Fcoeffs / project
 */

/* Explicitly instantiated for 1, 2, 3 dimensional Gaussians */
template<typename Fn, typename T, mra::Dimension NDIM>
void submit_fcoeffs_kernel(
  const mra::Domain<NDIM>& D,
  const T* gldata,
  const Fn* fn,
  const mra::Key<NDIM>& key,
  std::size_t N,
  std::size_t K,
  mra::TensorView<T, NDIM+1>& coeffs_view,
  const mra::TensorView<T, 2>& phibar_view,
  const mra::TensorView<T, 2>& hgT_view,
  T* tmp,
  bool* is_leaf_scratch,
  T thresh,
  ttg::device::Stream stream);


/* overload for single FunctionNode */
template<typename Fn, typename T, mra::Dimension NDIM>
void submit_fcoeffs_kernel(
  const mra::Domain<NDIM>& D,
  const T* gldata,
  const Fn& fn,
  const mra::Key<NDIM>& key,
  std::size_t K,
  mra::TensorView<T, NDIM>& coeffs_view,
  const mra::TensorView<T, 2>& phibar_view,
  const mra::TensorView<T, 2>& hgT_view,
  T* tmp,
  bool* is_leaf_scratch,
  T thresh,
  ttg::device::Stream stream);

template<mra::Dimension NDIM>
SCOPE std::size_t compress_tmp_size(std::size_t K) {
  const size_t TWOK2NDIM = std::pow(2*K,NDIM);
  return (2*TWOK2NDIM); // s & workspace
}

/**
 * Compress
 */

/* Explicitly instantiated for 3D */
template<typename T, mra::Dimension NDIM>
void submit_compress_kernel(
  const mra::Key<NDIM>& key,
  std::size_t N,
  std::size_t K,
  mra::TensorView<T, NDIM+1>& p_view,
  mra::TensorView<T, NDIM+1>& result_view,
  const mra::TensorView<T, 2>& hgT_view,
  T* tmp,
  T* sumsqs,
  const std::array<const T*, mra::Key<NDIM>::num_children()>& in_ptrs,
  ttg::device::Stream stream);

/* overload for single FunctionNode */
template<typename T, mra::Dimension NDIM>
void submit_compress_kernel(
  const mra::Key<NDIM>& key,
  std::size_t K,
  mra::TensorView<T, NDIM>& p_view,
  mra::TensorView<T, NDIM>& result_view,
  const mra::TensorView<T, 2>& hgT_view,
  T* tmp,
  T* sumsqs,
  const std::array<const T*, mra::Key<NDIM>::num_children()>& in_ptrs,
  ttg::device::Stream stream);


/**
 * Reconstruct
 */

template<mra::Dimension NDIM>
SCOPE std::size_t reconstruct_tmp_size(std::size_t K) {
  const size_t TWOK2NDIM = std::pow(2*K,NDIM); // s & workspace
  return 2*TWOK2NDIM;
}

template<typename T, mra::Dimension NDIM>
void submit_reconstruct_kernel(
  const mra::Key<NDIM>& key,
  std::size_t N,
  std::size_t K,
  mra::TensorView<T, NDIM+1>& node,
  const mra::TensorView<T, 2>& hg,
  const mra::TensorView<T, NDIM+1>& from_parent,
  const std::array<T*, mra::Key<NDIM>::num_children()>& r_arr,
  T* tmp,
  ttg::device::Stream stream);

/* overload for single function nodes */
template<typename T, mra::Dimension NDIM>
void submit_reconstruct_kernel(
  const mra::Key<NDIM>& key,
  std::size_t K,
  mra::TensorView<T, NDIM>& node,
  const mra::TensorView<T, 2>& hg,
  const mra::TensorView<T, NDIM>& from_parent,
  const std::array<T*, mra::Key<NDIM>::num_children()>& r_arr,
  T* tmp,
  ttg::device::Stream stream);

/**
 * add kernel
 */
template <typename T, mra::Dimension NDIM>
void submit_add_kernel(
  const mra::TensorView<T, NDIM+1>& nodeA,
  const mra::TensorView<T, NDIM+1>& nodeB,
  mra::TensorView<T, NDIM+1>& nodeR,
  const int* idxs,
  const T scalarA,
  const T scalarB,
  std::size_t N,
  std::size_t K,
  ttg::device::Stream stream);

#endif // HAVE_KERNELS_H