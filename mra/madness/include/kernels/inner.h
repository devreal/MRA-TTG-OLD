#ifndef MRA_INNER_H_INCL
#define MRA_INNER_H_INCL

#include "platform.h"
#include "types.h"
#include "tensorview.h"
#include "tensor.h"

namespace mra{

  namespace detail{

    template <typename T, Dimension NDIM>
    SCOPE TensorView<T, NDIM> inner(
      const TensorView<T, NDIM>& left,
      const TensorView<T, NDIM>& right,
      long k0 = -1,
      long k1 = 0)
    {
      if (k0 < 0) k0 += left.ndim();
      if (k1 < 0) k1 += right.ndim();
      long nd = left.ndim() + right.ndim() - 2;

      static_assert(NDIM > 1, "inner: dimension must be > 1");
      static_assert(left.dim(k0) == right.dim(k1), "inner: common index must have the same length");
      static_assert(nd > 0 && nd <= long(3), "inner: invalid number of diemensions in the result");

      long d[NDIM];

      long base=0;
      for (long i=0; i<k0; ++i) d[i] = left.dim(i);
      for (long i=k0+1; i<left.ndim(); ++i) d[i-1] = left.dim(i);

      base = left.ndim()-1;
      for (long i=0; i<k1; ++i) d[i+base] = right.dim(i);
      base--;

      for (long i=k1+1; i<right.ndim(); ++i) d[i+base] = right.dim(i);

      Tensor<T, NDIM> result(d);
      inner_result(left, right, k0, k1, result.current_view());
      return result;
    }

    template <typename T, Dimension NDIM>
    SCOPE void inner_result(
      const TensorView<T, NDIM>& left,
      const TensorView<T, NDIM>& right,
      long k0,
      long k1,
      TensorView<T, NDIM>& result)
    {
      if (k0 < 0) k0 += left.ndim();
      if (k1 < 0) k1 += right.ndim();
      long nd = left.ndim() + right.ndim() - 2;

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

      // TODO
      // long dimj = left.dim(k0);
      // TensorIterator<Q> iter1=right.unary_iterator(1,false,false,k1);

      // for (TensorIterator<T> iter0=left.unary_iterator(1,false,false,k0);
      //         iter0._p0; ++iter0) {
      //   T* MADNESS_RESTRICT xp0 = iter0._p0;
      //   long s0 = iter0._s0;
      //   for (iter1.reset(); iter1._p0; ++iter1) {
      //     T* MADNESS_RESTRICT p0 = xp0;
      //     Q* MADNESS_RESTRICT p1 = iter1._p0;
      //     long s1 = iter1._s0;
      //     resultT sum = 0;
      //     for (long j=0; j<dimj; ++j,p0+=s0,p1+=s1) {
      //       sum += (*p0) * (*p1);
      //     }
      //     *ptr++ += sum;
      //   }
      // }
    }
  }
}

#endif // MRA_INNER_H_INCL