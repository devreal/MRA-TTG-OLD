#ifndef MRA_TRANSFORM_H
#define MRA_TRANSFORM_H

#include "tensorview.h"
#include "types.h"
#include "platform.h"

#include <cstdlib>
#include "mxm.h"
namespace mra {

// template <typename aT, typename bT, typename cT>
// SCOPE
// void mTxmq(size_type dimi, size_type dimj, size_type dimk,
//            cT* __restrict__ c, const aT* a, const bT* b, std::ptrdiff_t ldb=-1) {
//   if (ldb == -1) ldb=dimj;
//   /* 3D implementation utilizing the tall-and-skinny shape of a(i,k) and c(i,j) (see transform() below).
//    * We distribute work along the i-dimension across the Y and Z dimensions of the thread-block.
//    * The X dimension of the thread-block computes along the j dimension. The k dimension is not parallelized
//    * as it would require reductions (could be added later for square matrices). */
//   for (size_type i = threadIdx.z*blockDim.y+threadIdx.y; i < dimi; i += blockDim.z*blockDim.y) {
//     cT* ci = c + i*dimj; // the row of C all threads in dim x work on
//     const aT *aik_ptr = a + i;
//     // beta = 0
//     for (size_type j = threadIdx.x; j < dimj; j += blockDim.x) {
//       ci[j] = 0.0;
//     }

//     for (long k=0; k<dimk; ++k,aik_ptr+=dimi) { /* not parallelized */
//       aT aki = *aik_ptr;
//       for (size_type j = threadIdx.x; j < dimj; j += blockDim.x) {
//         ci[j] += aki*b[k*ldb+j];
//       }
//     }
//   }
//   SYNCTHREADS();
// }

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
SCOPE void transform_dir(
  const TensorView<T, NDIM>& node,
  const TensorView<T, 2>& op,
  TensorView<T, NDIM>& tmp_result,
  TensorView<T, NDIM>& result,
  size_type axis) {
    if (axis == 0){
      result = inner(op, node, result, 0, axis);
    }
    else if (axis == node.ndim()-1){
      result = inner(node, op, result, axis, 0);
    }
    else{
      inner(node, op, tmp_result, axis, 0);
      cycledim(tmp_result, result, 1, axis, -1); // copy to make contiguous
    }
  }

template <typename T, Dimension NDIM>
SCOPE void general_transform(
        const TensorView<T, NDIM>& t,
        const std::array<TensorView<T, 3>, NDIM>& c,
        TensorView<T, NDIM>& result)
        {
          result = t;
          for (size_type i = 0; i < NDIM; ++i){
            inner(t, c[i], result, 0, 0);
          }
        }

} // namespace mra

#endif // MRA_TRANSFORM_H
