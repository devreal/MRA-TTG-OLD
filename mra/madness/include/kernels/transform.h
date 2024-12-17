#ifndef MRA_TRANSFORM_H
#define MRA_TRANSFORM_H

#include "tensorview.h"
#include "types.h"
#include "platform.h"

#include <cstdlib>

namespace mra {

/* reference implementation, adapted from madness */
template <typename aT, typename bT, typename cT>
SCOPE
void mTxmq_xyz(size_type dimi, size_type dimj, size_type dimk,
               cT* __restrict__ c, const aT* a, const bT* b, ssize_type ldb=-1) {
  if (ldb == -1) ldb=dimj;
  // assume the i dimension is the square of the j dimension
  assert(dimi == dimj*dimj);
  assert(dimj == blockDim.x);
  assert(dimk == blockDim.y);
  /* The i dimension is the slow dimension of AT and C.
   * We can thus split the i dimension into multiple GEMM of dimk*dimj
   * and distribute the i dimension across the the z dimension of the thread block.
   * Note: Access to B is coalesced (dimj*dimk) in the x and y block dimensions while access to A is strided
   * by dimi with a blocklength of dimj. */
  for (size_type i = threadIdx.z; i < dimi; i += blockDim.z*dimj) {
    const aT *ai = a + i;
    cT* ci = c + i;
    // beta = 0
    cT sum = 0.0;
    /* i and j are implicit, k is explicit */
    for (size_type k = 0; k < dimk; ++k) {
      const aT aval = ai[k*dimi + threadIdx.y];
      const bT bval = b[k*ldb + threadIdx.x];
      sum += aval * bval;
    }
    ci[threadIdx.y*dimj + threadIdx.x] = sum;
  }
  SYNCTHREADS();
}

/* reference implementation, adapted from madness */
template <typename aT, typename bT, typename cT>
SCOPE
void mTxmq(size_type dimi, size_type dimj, size_type dimk,
           cT* __restrict__ c, const aT* a, const bT* b, ssize_type ldb=-1) {
  if (dimi == dimj*dimj && dimj == blockDim.x && dimk == blockDim.y) {
    return mTxmq_xyz(dimi, dimj, dimk, c, a, b, ldb);
  }
  if (ldb == -1) ldb=dimj;
  /* trivial 2D implementation for devices */
  if (threadIdx.z == 0) {
    for (size_type i = threadIdx.y; i < dimi; i += blockDim.y) {
      cT* ci = c + i*dimj; // the row of C all threads in dim x work on
      const aT *aik_ptr = a + i;
      // beta = 0
      for (size_type j = threadIdx.x; j < dimj; j += blockDim.x) {
        ci[j] = 0.0;
      }

      for (long k=0; k<dimk; ++k,aik_ptr+=dimi) { /* not parallelized */
        aT aki = *aik_ptr;
        for (size_type j = threadIdx.x; j < dimj; j += blockDim.x) {
          ci[j] += aki*b[k*ldb+j];
        }
      }
    }
  }
  SYNCTHREADS();
}

template <Dimension NDIM, typename T>
SCOPE
void transform(const TensorView<T, NDIM>& t,
               const TensorView<T, 2>& c,
               TensorView<T, NDIM>& result,
               TensorView<T, NDIM>& workspace) {
  workspace = 0.0; // set to zero
  const T* pc = c.data();
  T *t0=workspace.data(), *t1=result.data();
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

} // namespace mra

#endif // MRA_TRANSFORM_H
