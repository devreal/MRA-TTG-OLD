#ifndef MRA_KERNELS_TRANSFORM_H
#define MRA_KERNELS_TRANSFORM_H

#include <cstdlib>
#include "mra/ops/mxm.h"
#include "mra/ops/inner.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/tensor/cycledim.h"
#include "mra/tensor/tensorview.h"
namespace mra {

  template <Dimension NDIM, typename T>
  SCOPE void transform(
    const TensorView<T, NDIM>& t,
    const TensorView<T, 2>& c,
    TensorView<T, NDIM>& result,
    T* workspace)
  {
    const T* pc = c.data();
#ifndef MRA_ENABLE_HOST
    // copy c to shared memory
    SHARED T c_tmp[MAX_THREADS_PER_BLOCK];
    if (thread_id() < c.size()) c_tmp[thread_id()] = pc[thread_id()];
    SYNCTHREADS();
    pc = c_tmp;
#endif

    T *t0=workspace, *t1=result.data();
    if (t.ndim() & 0x1) std::swap(t0,t1);
    const size_type dimj = c.dim(1);
    size_type dimi = 1;
    for (size_type n=1; n<t.ndim(); ++n) dimi *= dimj;
    mTxmq(dimi, dimj, dimj, t0, t.data(), pc);
    for (size_type n=1; n<t.ndim(); ++n) {
      mTxmq(dimi, dimj, dimj, t1, t0, pc);
      std::swap(t0,t1);
    }
    /* no need to synchronize here, mTxmq synchronizes */
  }

  template <typename T, Dimension NDIM>
  SCOPE void transform_dir(
    const TensorView<T, NDIM>& node,
    const TensorView<T, 2>& op,
    TensorView<T, NDIM>& tmp_result,
    TensorView<T, NDIM>& result,
    size_type axis) {
      if (axis == 0){
        detail::inner(op, node, result, 0, axis);
      }
      else if (axis == node.ndim()-1){
        detail::inner(node, op, result, axis, 0);
      }
      else {
        detail::inner(node, op, tmp_result, axis, 0);
        detail::cycledim(tmp_result, result, 1, axis, -1); // copy to make contiguous
      }
    }

  template <typename T, Dimension NDIM, std::size_t ARRDIM = NDIM>
  SCOPE void general_transform(
    const TensorView<T, NDIM>& t,
    const std::array<TensorView<T, 2>, ARRDIM>& c,
    TensorView<T, NDIM>& result)
    {
      result = t;
      for (size_type i = 0; i < NDIM; ++i){
        detail::inner(t, c[i], result, 0, 0);
      }
    }

} // namespace mra

#endif // MRA_TRANSFORM_H
