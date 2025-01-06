#ifndef MRA_MXM_H_INCL
#define MRA_MXM_H_INCL

#include "platform.h"
#include "types.h"

namespace mra{

  /* reference implementation, adapted from madness */
  template <typename aT, typename bT, typename cT>
  SCOPE void mTxmq(size_type dimi, size_type dimj, size_type dimk,
          cT* __restrict__ c, const aT* a, const bT* b, std::ptrdiff_t ldb=-1) {
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

  template <typename T, Dimension NDIM>
  SCOPE void mxm(){}

  template <typename T, Dimension NDIM>
  SCOPE void mTxm(){}

  template <typename T, Dimension NDIM>
  SCOPE void mxmT(){}

  template <typename T, Dimension NDIM>
  SCOPE void mTxmT(){}

}

#endif // MRA_MXM_H_INCL