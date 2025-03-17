#ifndef MRA_MXM_H
#define MRA_MXM_H

#include "mra/misc/types.h"
#include "mra/misc/platform.h"

namespace mra{


  namespace detail {
    /* mTxm that blocks on A (i.e., dimi is the largest dimension )*/
    template <typename aT, typename bT, typename cT, bool Q = false>
    SCOPE bool mTxm_block_a(size_type dimi, size_type dimj, size_type dimk,
                            cT* __restrict__ c, const aT* a, const bT* b) {

      if (dimk != blockDim.x || dimj != blockDim.x) {
        /* Required: dimj and dimk must match thread X dimension */
        return false;
      }
      SHARED aT block_a[MAX_THREADS_PER_BLOCK];
      size_type a_block_dimi;
      a_block_dimi = block_size() / dimk;

      auto tid = thread_id();
      /* A is transposed and we want to coalesce in dimi */
      auto a_trans_i = tid % a_block_dimi; // index in transposed block A(k, i)
      auto a_trans_k = tid / a_block_dimi;
      auto a_block_i = a_trans_k; // index in non-transposed block_a(i, k)
      auto a_block_k = a_trans_i;
      for (size_type i = 0; i < dimi; i += a_block_dimi) {

        if (i+a_trans_i < dimi) { /* in case dimi is not a multiple of Y*Z */
          /* transpose block of A into shared memory */
          block_a[a_block_i*dimk + a_block_k] = a[a_trans_k*dimi + i+a_block_i];
        }

        /* make sure the block is written */
        SYNCTHREADS();

        if (i+a_block_i < dimi) { /* in case dimi is not a multiple of Y*Z */
          /* C is not transposed so use block_i */
          size_type c_idx = (i+a_block_i)*dimj + threadIdx.x;
          cT sum = 0.0;
          /* k is not parallel */
          const aT* a_ik = &block_a[(threadIdx.z*blockDim.y+threadIdx.y)*dimk];
          for (size_type k = 0; k < dimk; ++k) {
            sum += a_ik[k] * b[k*dimj + threadIdx.x];
          }
          if constexpr (Q) {
            c[c_idx] += sum;
          } else {
            c[c_idx] = sum;
          }
        }

        /* synchronize before entering next iteration or completing */
        SYNCTHREADS();
      }
      return true;
    }
  } // namespace detail


  /**
   * reference implementation, adapted from madness
   * c(i,j) += sum(k) a(k,i)*b(k,j)
   */
  template <typename aT, typename bT, typename cT, bool Q = false>
  SCOPE void mTxm(size_type dimi, size_type dimj, size_type dimk,
          cT* __restrict__ c, const aT* a, const bT* b, std::ptrdiff_t ldb=-1) {
    if (ldb == -1) ldb=dimj;
    if (ldb == dimj && mTxm_block_a(dimi, dimj, dimk, c, a, b)) {
      return; // succesfully blocked on A
    }
    /* TODO: block on B */

    /* trivial 2D implementation for devices */
    if (threadIdx.z == 0) {
      for (size_type i = threadIdx.y; i < dimi; i += blockDim.y) {
        cT* ci = c + i*dimj; // the row of C all threads in dim x work on
        const aT *aik_ptr = a + i;
        if constexpr(Q) {
          for (size_type j = threadIdx.x; j < dimj; j += blockDim.x) {
            ci[j] = 0.0;
          }
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

  // /**
  //  * reference implementation, adapted from madness
  //  * c(i,j) = sum(k) a(k,i)*b(k,j)
  //  */
  // template <typename aT, typename bT, typename cT>
  // SCOPE void mTxmq(size_type dimi, size_type dimj, size_type dimk,
  //         cT* __restrict__ c, const aT* a, const bT* b, std::ptrdiff_t ldb=-1) {
  //   mTxm<aT, bT, cT, true>(dimi, dimj, dimk, c, a, b, ldb);
  // }
  template <typename aT, typename bT, typename cT>
  SCOPE
  void mTxmq(size_type dimi, size_type dimj, size_type dimk,
            cT* __restrict__ c, const aT* a, const bT* b, std::ptrdiff_t ldb=-1) {
    if (ldb == -1) ldb=dimj;
    /* 3D implementation utilizing the tall-and-skinny shape of a(i,k) and c(i,j) (see transform() below).
    * We distribute work along the i-dimension across the Y and Z dimensions of the thread-block.
    * The X dimension of the thread-block computes along the j dimension. The k dimension is not parallelized
    * as it would require reductions (could be added later for square matrices). */
    for (size_type i = threadIdx.z*blockDim.y+threadIdx.y; i < dimi; i += blockDim.z*blockDim.y) {
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
    SYNCTHREADS();
  }

  /**
   * reference implementation, adapted from madness
   *
   * c(i,j) += sum(k) a(i,k)*b(k,j)
   */
  template <typename aT, typename bT, typename cT, bool Q = false>
  SCOPE void mxm(size_type dimi, size_type dimj, size_type dimk,
                 cT* __restrict__ c, const aT* a, const bT* b, std::ptrdiff_t ldb=-1) {
    if (ldb == -1) ldb=dimj;
    /* trivial 2D implementation for devices */
    if (threadIdx.z == 0) {
      for (size_type i = threadIdx.y; i < dimi; i += blockDim.y) {
        cT* ci = c + i*dimj; // the row of C all threads in dim x work on
        const aT *ai_ptr = a + i*dimk;
        if constexpr(Q) {
          for (size_type j = threadIdx.x; j < dimj; j += blockDim.x) {
            ci[j] = 0.0;
          }
        }
        for (size_type j = threadIdx.x; j < dimj; j += blockDim.x) {
          for (long k=0; k<dimk; ++k) { /* not parallelized */
            ci[j] += ai_ptr[k]*b[k*ldb+j];
          }
        }
      }
    }
    SYNCTHREADS();
  }

  /**
   * reference implementation, adapted from madness
   *
   * c(i,j) = sum(k) a(i,k)*b(k,j)
   */
  template <typename aT, typename bT, typename cT, bool Q = false>
  SCOPE void mxmq(size_type dimi, size_type dimj, size_type dimk,
                  cT* __restrict__ c, const aT* a, const bT* b, std::ptrdiff_t ldb=-1) {
    mxm<aT, bT, cT, true>(dimi, dimj, dimk, c, a, b, ldb);
  }

  /**
   * reference implementation, adapted from madness
   *
   * c(i,j) += sum(k) a(i,k)*b(j,k)
   */
  template <typename aT, typename bT, typename cT, bool Q = false>
  SCOPE void mxmT(size_type dimi, size_type dimj, size_type dimk,
                 cT* __restrict__ c, const aT* a, const bT* b, std::ptrdiff_t ldb=-1) {
    if (ldb == -1) ldb=dimj;
    /* trivial 2D implementation for devices */
    if (threadIdx.z == 0) {
      for (size_type i = threadIdx.y; i < dimi; i += blockDim.y) {
        cT* ci = c + i*dimj; // the row of C all threads in dim x work on
        const aT *ai_ptr = a + i*dimk;
        for (size_type j = threadIdx.x; j < dimj; j += blockDim.x) {
          cT sum = 0.0;
          for (size_type k=0; k<dimk; ++k) { /* not parallelized */
            /**
             * TODO: this is not optimal, we should transpose b first into shared memory
             */
            sum += ai_ptr[k]*b[j*dimk+k];
          }
          if constexpr (Q) {
            ci[j] = sum;
          } else {
            ci[j] += sum;
          }

        }
      }
    }
    SYNCTHREADS();
  }

  /**
   * reference implementation, adapted from madness
   *
   * c(i,j) = sum(k) a(i,k)*b(j,k)
   */
  template <typename aT, typename bT, typename cT>
  SCOPE void mxmTq(size_type dimi, size_type dimj, size_type dimk,
                 cT* __restrict__ c, const aT* a, const bT* b, std::ptrdiff_t ldb=-1) {
    mxmT<aT, bT, cT, true>(dimi, dimj, dimk, c, a, b, ldb);
  }

  /**
   * reference implementation, adapted from madness
   *
   * c(i,j) += sum(k) a(k,i)*b(j,k)
   */
  template <typename aT, typename bT, typename cT, bool Q = false>
  SCOPE void mTxmT(size_type dimi, size_type dimj, size_type dimk,
                   cT* __restrict__ c, const aT* a, const bT* b, std::ptrdiff_t ldb=-1) {
    if (ldb == -1) ldb=dimj;
    /* trivial 2D implementation for devices */
    if (threadIdx.z == 0) {
      for (size_type i = threadIdx.y; i < dimi; i += blockDim.y) {
        cT* ci = c + i*dimj; // the row of C all threads in dim x work on
        if constexpr(Q) {
          for (size_type j = threadIdx.x; j < dimj; j += blockDim.x) {
            ci[j] = 0.0;
          }
        }
        for (size_type j = threadIdx.x; j < dimj; j += blockDim.x) {
          const aT *aik_ptr = a + i;
          for (long k=0; k<dimk; ++k,aik_ptr+=dimj) { /* not parallelized */
            /**
             * TODO: this is not optimal, we should transpose a and b first into shared memory
             */
            ci[j] += *aik_ptr * b[j*ldb+k];
          }
        }
      }
    }
    SYNCTHREADS();
  }

  /**
   * reference implementation, adapted from madness
   *
   * c(i,j) = sum(k) a(k,i)*b(j,k)
   */
  template <typename aT, typename bT, typename cT>
  SCOPE void mTxmTq(size_type dimi, size_type dimj, size_type dimk,
                   cT* __restrict__ c, const aT* a, const bT* b, std::ptrdiff_t ldb=-1) {
    mTxmT<aT, bT, cT, true>(dimi, dimj, dimk, c, a, b, ldb);
  }
}

#endif // MRA_MXM_H
