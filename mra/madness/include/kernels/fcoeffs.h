#ifndef MRA_FCOEFFS_H
#define MRA_FCOEFFS_H

#include "types.h"
#include "tensorview.h"
#include "domain.h"
#include "gl.h"
#include "platform.h"
#include "kernels/fcube.h"
#include "kernels/transform.h"

namespace mra {

  /* Returns the total size of temporary memory needed for
  * the project() kernel. */
  template<mra::Dimension NDIM>
  SCOPE std::size_t fcoeffs_tmp_size(std::size_t K) {
  const size_t K2NDIM = std::pow(K,NDIM);
  const std::size_t TWOK2NDIM = std::pow(2*K, NDIM);
  return (3*TWOK2NDIM) // workspace, values and r
       + (NDIM*K2NDIM) // xvec in fcube
       + (NDIM*K)      // x in fcube
       + (3*K2NDIM);   // workspace in transform, child_values, r
  }

  namespace detail {

    template<typename Fn, typename T, Dimension NDIM>
    DEVSCOPE void fcoeffs_kernel_impl(
      const Domain<NDIM>& D,
      const T* gldata,
      const Fn& f,
      Key<NDIM> key,
      std::size_t K,
      T* tmp,
      const T* phibar_ptr,
      T* coeffs_ptr,
      const T* hgT_ptr,
      bool *is_leaf,
      T thresh)
    {
      bool is_t0 = 0 == (threadIdx.x + threadIdx.y + threadIdx.z);
      const std::size_t K2NDIM = std::pow(K, NDIM);
      const std::size_t TWOK2NDIM = std::pow(2*K, NDIM);
      /* reconstruct tensor views from pointers
      * make sure we have the values at the same offset (0) as in kernel 1 */
      SHARED TensorView<T, NDIM> values, r, child_values, workspace, coeffs;
      SHARED TensorView<T, 2   > hgT, x_vec, x, phibar;
      if (is_t0) {
        values       = TensorView<T, NDIM>(&tmp[0       ], 2*K);
        r            = TensorView<T, NDIM>(&tmp[TWOK2NDIM+0*K2NDIM], K);
        child_values = TensorView<T, NDIM>(&tmp[TWOK2NDIM+1*K2NDIM], K);
        workspace    = TensorView<T, NDIM>(&tmp[TWOK2NDIM+2*K2NDIM], K);
        x_vec        = TensorView<T, 2   >(&tmp[TWOK2NDIM+3*K2NDIM], NDIM, K2NDIM);
        x            = TensorView<T, 2   >(&tmp[TWOK2NDIM+3*K2NDIM + (NDIM*K2NDIM)], NDIM, K);
        phibar       = TensorView<T, 2   >(phibar_ptr, K, K);
        coeffs       = TensorView<T, NDIM>(coeffs_ptr, K);
      }
      SYNCTHREADS();

      /* check for our function */
      if ((key.level() < initial_level(f))) {
        coeffs = T(1e7); // set to obviously bad value to detect incorrect use
        *is_leaf = false;
      }
      if (is_negligible<Fn,T,NDIM>(f, D.template bounding_box<T>(key), mra::truncate_tol(key,thresh))) {
        /* zero coeffs */
        coeffs = T(0.0);
        *is_leaf = true;
      } else {

        /* compute one child */
        for (int bid = 0; bid < key.num_children(); bid++) {
          Key<NDIM> child = key.child_at(bid);
          child_values = 0.0; // TODO: needed?
          fcube(D, gldata, f, child, thresh, child_values, K, x, x_vec);
          r = 0.0;
          transform(child_values, phibar, r, workspace);
          auto child_slice = get_child_slice<NDIM>(key, K, bid);
          values(child_slice) = r;
        }

        /* reallocate some of the tensorviews */
        if (is_t0) {
          r          = TensorView<T, NDIM>(&tmp[TWOK2NDIM], 2*K);
          workspace  = TensorView<T, NDIM>(&tmp[2*TWOK2NDIM], 2*K);
          hgT        = TensorView<T, 2>(hgT_ptr, 2*K, 2*K);
        }
        SYNCTHREADS();
        T fac = std::sqrt(D.template get_volume<T>()*std::pow(T(0.5),T(NDIM*(1+key.level()))));
        r = 0.0;

        values *= fac;
        // Inlined: filter<T,K,NDIM>(values,r);
        transform<NDIM>(values, hgT, r, workspace);

        auto child_slice = get_child_slice<NDIM>(key, K, 0);
        auto r_slice = r(child_slice);
        coeffs = r_slice; // extract sum coeffs
        r_slice = 0.0; // zero sum coeffs so can easily compute norm of difference coeffs
        /* TensorView assignment synchronizes */
        T norm = mra::normf(r);
        if (is_t0) {
          *is_leaf = (norm < truncate_tol(key,thresh)); // test norm of difference coeffs
          //if (!*is_leaf) {
          //  std::cout << "fcoeffs not leaf " << key << " norm " << norm << std::endl;
          //}
        }
      }
    }

    template<typename Fn, typename T, Dimension NDIM>
    GLOBALSCOPE void fcoeffs_kernel(
      const Domain<NDIM>& D,
      const T* gldata,
      const Fn* fns,
      Key<NDIM> key,
      std::size_t N,
      std::size_t K,
      T* tmp,
      const T* phibar_ptr,
      T* coeffs_ptr,
      const T* hgT_ptr,
      bool *is_leaf,
      T thresh)
    {
      const size_t K2NDIM = std::pow(K,NDIM);
      /* adjust pointers for the function of each block */
      int blockid = blockIdx.x;
      for (std::size_t i = blockid; i < N; i += blockDim.x) {
        fcoeffs_kernel_impl(D, gldata, fns[i], key, K,
                            &tmp[(fcoeffs_tmp_size<NDIM>(K)*i)],
                            phibar_ptr, coeffs_ptr+(i*K2NDIM),
                            hgT_ptr, &is_leaf[i], thresh);
      }
    }

  } // namespace detail

  /**
   * Fcoeffs used in project
   */
  template<typename Fn, typename T, mra::Dimension NDIM>
  void submit_fcoeffs_kernel(
      const mra::Domain<NDIM>& D,
      const T* gldata,
      const Fn* fns,
      const mra::Key<NDIM>& key,
      std::size_t N,
      std::size_t K,
      mra::TensorView<T, NDIM+1>& coeffs_view,
      const mra::TensorView<T, 2>& phibar_view,
      const mra::TensorView<T, 2>& hgT_view,
      T* tmp,
      bool* is_leaf_scratch,
      T thresh,
      ttg::device::Stream stream)
  {
    /**
     * Launch the kernel with KxK threads in each of the N blocks.
     * Computation on functions is embarassingly parallel and no
     * synchronization is required.
     */
    Dim3 thread_dims = Dim3(K, K, 1); // figure out how to consider register usage
    /* launch one block per child */
    CALL_KERNEL(detail::fcoeffs_kernel, N, thread_dims, 0, stream,
      (D, gldata, fns, key, N, K, tmp, phibar_view.data(),
      coeffs_view.data(), hgT_view.data(),
      is_leaf_scratch, thresh));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_FCOEFFS_H
