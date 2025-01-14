#ifndef MRA_TRANSFORM_H
#define MRA_TRANSFORM_H

#include "tensorview.h"
#include "types.h"
#include "platform.h"

#include <cstdlib>
#include "mxm.h"
namespace mra {

template <Dimension NDIM, typename T>
SCOPE void transform(const TensorView<T, NDIM>& t,
               const TensorView<T, 2>& c,
               TensorView<T, NDIM>& result,
               T* workspace) {
  const T* pc = c.data();
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
SCOPE TensorView<T, NDIM> transform_dir(
  const TensorView<T, NDIM>& node,
  const TensorView<T, 2>& op,
  size_type axis,
  TensorView<T, NDIM>& result) {
    if (axis == 0){
      result = inner(op, node, 0, axis);
      return result;
    }
    else if (axis == node.ndim()-1){
      result = inner(node, op, axis, 0);
      return result;
    }
    else{
      result = copy(inner(node, op, axis, 0).cycledim(1, axis, -1)); // copy to make contiguous
      return result;
    }
  }

template <typename T, Dimension NDIM>
SCOPE void general_transform(
        const TensorView<T, NDIM>& t,
        const std::array<TensorView<T, 2>, NDIM>& c,
        TensorView<T, NDIM>& result)
        {
          result = t;
          for (size_type i = 0; i < NDIM; ++i){
            inner(result, c[i], 0, 0);
          }
        }

} // namespace mra

#endif // MRA_TRANSFORM_H
