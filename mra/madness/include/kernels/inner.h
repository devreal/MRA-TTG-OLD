#ifndef MRA_INNER_H_INCL
#define MRA_INNER_H_INCL

#include "platform.h"
#include "types.h"
#include "tensorview.h"
#include "tensor.h"

namespace mra{

  namespace detail{

    template <typename T, Dimension NDIM>
    SCOPE void inner(
      const TensorView<T, NDIM>& left,
      const TensorView<T, NDIM>& right,
      TensorView<T, NDIM>& result,
      long k0 = -1,
      long k1 = 0)
    {
      if (k0 < 0) k0 += left.ndim();
      if (k1 < 0) k1 += right.ndim();
      long nd = left.ndim() + right.ndim() - 2;

      static_assert(NDIM > 1, "inner: dimension must be > 1");
      static_assert(left.dim(k0) == right.dim(k1), "inner: common index must have the same length");

      if (k0 < 0) k0 += left.ndim();
      if (k1 < 0) k1 += right.ndim();

      if (k0==0 && k1==0) {
        // c[i,j] = a[k,i]*b[k,j] ... collapsing extra indices to i & j
        long dimk = left.dim(k0);
        long dimj = right.stride(0);
        long dimi = left.stride(0);
        mTxm(dimi,dimj,dimk,ptr,left.ptr(),right.ptr());
        return;
      }
      else if (k0==(left.ndim()-1) && k1==(right.ndim()-1)) {
        // c[i,j] = a[i,k]*b[j,k] ... collapsing extra indices to i & j
        long dimk = left.dim(k0);
        long dimi = left.size()/dimk;
        long dimj = right.size()/dimk;
        mxmT(dimi,dimj,dimk,ptr,left.ptr(),right.ptr());
        return;
      }
      else if (k0==0 && k1==(right.ndim()-1)) {
        // c[i,j] = a[k,i]*b[j,k] ... collapsing extra indices to i & j
        long dimk = left.dim(k0);
        long dimi = left.stride(0);
        long dimj = right.size()/dimk;
        mTxmT(dimi,dimj,dimk,ptr,left.ptr(),right.ptr());
        return;
      }
      else if (k0==(left.ndim()-1) && k1==0) {
        // c[i,j] = a[i,k]*b[k,j] ... collapsing extra indices to i & j
        long dimk = left.dim(k0);
        long dimi = left.size()/dimk;
        long dimj = right.stride(0);
        mxm(dimi,dimj,dimk,ptr,left.ptr(),right.ptr());
        return;
      }

      // TODO: use more than the first slice of threads in z here
      if (threadIdx.z == 0) {
        size_type dimj = left.dim(k0);
        auto iter1 = right.unary_iterator(1, false, k1);
        T* ptr = result.data();
        for (auto iter0 = left.unary_iterator(1, false, k0);
             iter0.ptr() != nullptr;
             ++iter0, ptr += iter1.size()) {
          T* __restrict__ xp0 = iter0.ptr();
          ssize_type s0 = iter0.s0();
          iter1.reset();
          for (iter1 += thread_id();
               iter1.ptr() != nullptr;
               iter1 += blockDim.x*blockDim.y, ptr += blockDim.x*blockDim.y) {
            T* __restrict__ p0 = xp0;
            T* __restrict__ p1 = iter1.ptr();
            ssize_type s1 = iter1.s0();
            T sum = 0;
            for (size_type j=0; j<dimj; ++j, p0+=s0, p1+=s1) {
              sum += (*p0) * (*p1);
            }
            ptr[thread_id()] += sum;
          }
        }
      }
    }
  }
}

#endif // MRA_INNER_H_INCL