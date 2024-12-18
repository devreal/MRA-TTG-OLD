#ifndef HAVE_MRA_DEVICE_FUNCTIONS_H
#define HAVE_MRA_DEVICE_FUNCTIONS_H

#include "platform.h"
#include "types.h"
#include "key.h"
#include "tensorview.h"

namespace mra {


    /// In given box return the truncation tolerance for given threshold
    template <typename T, Dimension NDIM>
    SCOPE T truncate_tol(const Key<NDIM>& key, const T thresh) {
        return thresh; // nothing clever for now
    }

    /// Computes square of distance between two coordinates
    template <typename T>
    SCOPE T distancesq(const Coordinate<T,1>& p, const Coordinate<T,1>& q) {
        T x = p[0]-q[0];
        return x*x;
    }

    template <typename T>
    SCOPE T distancesq(const Coordinate<T,2>& p, const Coordinate<T,2>& q) {
        T x = p[0]-q[0], y = p[1]-q[1];
        return x*x + y*y;
    }

    template <typename T>
    SCOPE T distancesq(const Coordinate<T,3>& p, const Coordinate<T,3>& q) {
        T x = p[0]-q[0], y = p[1]-q[1], z=p[2]-q[2];
        return x*x + y*y + z*z;
    }

    template <typename T>
    SCOPE void distancesq(const Coordinate<T,1>& p, const TensorView<T,1>& q, T* rsq, size_type N) {
        const T x = p(0);
#ifdef __CUDA_ARCH__
        for (size_type i = thread_id(); i < N; i += block_size()) {
            T xx = q(0,i) - x;
            rsq[i] = xx*xx;
        }
        SYNCTHREADS();
#else  // __CUDA_ARCH__
        for (size_type i=0; i<N; i++) {
            T xx = q(0,i) - x;
            rsq[i] = xx*xx;
        }
#endif // __CUDA_ARCH__
    }

    template <typename T>
    SCOPE void distancesq(const Coordinate<T,2>& p, const TensorView<T,2>& q, T* rsq, size_type N) {
        const T x = p(0);
        const T y = p(1);
#ifdef __CUDA_ARCH__
        for (size_type i = thread_id(); i < N; i += block_size()) {
            T xx = q(0,i) - x;
            T yy = q(1,i) - y;
            rsq[i] = xx*xx + yy*yy;
        }
        SYNCTHREADS();
#else  // __CUDA_ARCH__
        for (size_type i=0; i<N; i++) {
            T xx = q(0,i) - x;
            T yy = q(1,i) - y;
            rsq[i] = xx*xx + yy*yy;
        }
#endif // __CUDA_ARCH__
    }

    template <typename T>
    SCOPE void distancesq(const Coordinate<T,3>& p, const TensorView<T,2>& q, T* rsq, size_type N) {
        const T x = p(0);
        const T y = p(1);
        const T z = p(2);
#ifdef __CUDA_ARCH__
        for (size_type i = thread_id(); i < N; i += block_size()) {
            T xx = q(0,i) - x;
            T yy = q(1,i) - y;
            T zz = q(2,i) - z;
            rsq[i] = xx*xx + yy*yy + zz*zz;
        }
        SYNCTHREADS();
#else  // __CUDA_ARCH__
        for (size_type i=0; i<N; i++) {
            T xx = q(0,i) - x;
            T yy = q(1,i) - y;
            T zz = q(2,i) - z;
            rsq[i] = xx*xx + yy*yy + zz*zz;
        }
#endif // __CUDA_ARCH__
    }


    namespace detail {
      /**
       * Reduce the contributions of each calling thread in a block into a single value.
       * On the host, we simply copy the result into the output value.
       * This requires block_size() shared memory.
       */
      template <typename T>
      SCOPE void reduce(const T input, T* output) {
#ifdef __CUDA_ARCH__
        extern __shared__ T sdata[];

        size_type tid = thread_id();
        sdata[tid] = input;
        __syncthreads();

        for (size_type s = block_size() / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            *output = sdata[0];
        }
#else  // __CUDA_ARCH__
        *output = input;
#endif // __CUDA_ARCH__
      }
    }

    template <typename T, Dimension NDIM, typename accumulatorT>
    SCOPE void sumabssq(const TensorView<T, NDIM>& a, accumulatorT* sum) {
      accumulatorT s = 0.0;
      /* every thread computes a partial sum */
      foreach_idx(a, [&](auto... idx) mutable {
        accumulatorT x = a(idx...);
        s += x*x;
      });
      detail::reduce(s, sum);
      SYNCTHREADS();
    }


    /// Compute Frobenius norm ... still needs specializing for complex
    template <typename T, Dimension NDIM, typename accumulatorT = std::decay_t<T>>
    SCOPE accumulatorT normf(const TensorView<T, NDIM>& a) {
#ifdef __CUDA_ARCH__
      __shared__ accumulatorT sum;
#else  // __CUDA_ARCH__
      accumulatorT sum;
#endif // __CUDA_ARCH__
      sumabssq(a, &sum);
#ifdef __CUDA_ARCH__
      /* wait for all threads to contribute */
      SYNCTHREADS();
#endif // __CUDA_ARCH__
      return std::sqrt(sum);
    }

    template<typename T>
    SCOPE void print(const T& t) {
      foreach_idxs(t, [&](auto... idx){ printf("[%lu %lu %lu] %f\n", idx..., t(idx...)); });
      SYNCTHREADS();
    }

    template<typename T>
    SCOPE void print(const T& t, const char* loc, const char *name) {
      if constexpr (T::ndim() == 3) {
        foreach_idxs(t, [&](auto... idx){ printf("%s: %s[%lu %lu %lu] %p %e\n", loc, name, idx..., &t(idx...), t(idx...)); });
      } else if constexpr (T::ndim() == 2) {
        foreach_idxs(t, [&](auto... idx){ printf("%s: %s[%lu %lu] %p %e\n", loc, name, idx..., &t(idx...), t(idx...)); });
      }
      SYNCTHREADS();
    }

} // namespace mra


#endif // HAVE_MRA_DEVICE_FUNCTIONS_H