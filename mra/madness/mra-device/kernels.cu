#include <type_traits>
#include <cstddef>
#include "tensorview.h"
#include "gaussian.h"
#include "key.h"
#include "domain.h"
#include "gl.h"
#include "kernels.h"
#include "functions.h"
#include "util.h"
#include "functionfunctor.h"
#include "helper.h"

template<typename T>
struct type_printer;


using namespace mra;

/// Set X(d,mu) to be the mu'th quadrature point in dimension d for the box described by key
template<typename T, Dimension NDIM>
SCOPE void make_quadrature_pts(
  const Domain<NDIM>& D,
  const T* gldata,
  const Key<NDIM>& key,
  TensorView<T,2>& X, std::size_t K)
{
  assert(X.dim(0) == NDIM);
  assert(X.dim(1) == K);
  const Level n = key.level();
  const std::array<Translation,NDIM>& l = key.translation();
  const T h = std::pow(T(0.5),T(n));
  /* retrieve x[] from constant memory, use float */
  const T *x, *w;
  GLget(gldata, &x, &w, K);
  if (threadIdx.z == 0) {
    for (int d = threadIdx.y; d < X.dim(0); d += blockDim.y) {
      T lo, hi; std::tie(lo,hi) = D.get(d);
      T width = h*D.get_width(d);
      for (int i = threadIdx.x; i < X.dim(1); i += blockDim.x) {
        X(d,i) = lo + width*(l[d] + x[i]);
      }
    }
  }
  /* wait for all to complete */
  SYNCTHREADS();
}


#if 0
// TODO: REMOVE

//namespace detail {
  template <class functorT> using initial_level_t =
      decltype(std::declval<const functorT>().initial_level());
  template <class functorT> using supports_initial_level =
      ::mra::is_detected<initial_level_t,functorT>;

  template <class functorT, class pairT> using is_negligible_t =
      decltype(std::declval<const functorT>().is_negligible(std::declval<pairT>(),std::declval<double>()));
  template <class functorT, class pairT> using supports_is_negligible =
      ::mra::is_detected<is_negligible_t,functorT,pairT>;
//}


template <typename functionT, typename T, Dimension NDIM>
SCOPE bool is_negligible(
  const functionT& f,
  const std::pair<Coordinate<T,NDIM>, Coordinate<T,NDIM>>& box,
  T thresh)
{
    using pairT = std::pair<Coordinate<T,NDIM>,Coordinate<T,NDIM>>;
    if constexpr (/*detail::*/supports_is_negligible<functionT,pairT>()) return f.is_negligible(box, thresh);
    else return false;
}
#endif // 0


template <typename functorT, typename T, Dimension NDIM>
SCOPE
void fcube(const Domain<NDIM>& D,
           const T* gldata,
           const functorT& f,
           const Key<NDIM>& key,
           const T thresh,
           // output
           TensorView<T,3>& values,
           std::size_t K,
           // temporaries
           TensorView<T, 2>& x,
           TensorView<T, 2>& xvec) {
  if (is_negligible(f, D.template bounding_box<T>(key), truncate_tol(key,thresh))) {
      values = 0.0;
      /* TensorView assigment synchronizes */
  }
  else {
    const size_t K = values.dim(0);
    const size_t K2NDIM = std::pow(K,NDIM);
    // sanity checks
    assert(x.dim(0) == NDIM);
    assert(x.dim(1) == K   );
    assert(xvec.dim(0) ==   NDIM);
    assert(xvec.dim(1) == K2NDIM);
    make_quadrature_pts(D, gldata, key, x, K);

    constexpr bool call_coord = std::is_invocable_r<T, decltype(f), Coordinate<T,NDIM>>(); // f(coord)
    constexpr bool call_1d = (NDIM==1) && std::is_invocable_r<T, decltype(f), T>(); // f(x)
    constexpr bool call_2d = (NDIM==2) && std::is_invocable_r<T, decltype(f), T, T>(); // f(x,y)
    constexpr bool call_3d = (NDIM==3) && std::is_invocable_r<T, decltype(f), T, T, T>(); // f(x,y,z)
    constexpr bool call_vec = std::is_invocable<decltype(f), const TensorView<T,2>&, T*, std::size_t>(); // vector API
    static_assert(std::is_invocable<decltype(f), const TensorView<T,2>&, T*, std::size_t>());
    static_assert(call_coord || call_1d || call_2d || call_3d || call_vec, "no working call");

    if constexpr (call_1d || call_2d || call_3d || call_vec) {
      make_xvec(x, xvec, std::integral_constant<Dimension, NDIM>{});
      if constexpr (call_vec) {
        f(xvec, values.data(), K2NDIM);
      }
      else if constexpr (call_1d || call_2d || call_3d) {
        eval_cube_vec(f, xvec, values);
      }
    }
    else if constexpr (call_coord) {
      eval_cube(f, x, values);
    }
    else {
      //throw "how did we get here?";
      // TODO: how to handle this?
      assert(!"Failed to handle eval call!");
    }
    SYNCTHREADS();
  }
}

/* reference implementation, adapted from madness */
template <typename aT, typename bT, typename cT>
SCOPE
void mTxmq(std::size_t dimi, std::size_t dimj, std::size_t dimk,
           cT* __restrict__ c, const aT* a, const bT* b, std::ptrdiff_t ldb=-1) {
  if (ldb == -1) ldb=dimj;
  /* trivial 2D implementation for devices */
  if (threadIdx.z == 0) {
    for (std::size_t i = threadIdx.y; i < dimi; i += blockDim.y) {
      cT* ci = c + i*dimj; // the row of C all threads in dim x work on
      const aT *aik_ptr = a + i;
      // beta = 0
      for (std::size_t j = threadIdx.x; j < dimj; j += blockDim.x) {
        ci[j] = 0.0;
      }

      for (long k=0; k<dimk; ++k,aik_ptr+=dimi) { /* not parallelized */
        aT aki = *aik_ptr;
        for (std::size_t j = threadIdx.x; j < dimj; j += blockDim.x) {
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
  const size_t dimj = c.dim(1);
  size_t dimi = 1;
  for (size_t n=1; n<t.ndim(); ++n) dimi *= dimj;
  mTxmq(dimi, dimj, dimj, t0, t.data(), pc);
  for (size_t n=1; n<t.ndim(); ++n) {
    mTxmq(dimi, dimj, dimj, t1, t0, pc);
    std::swap(t0,t1);
  }
  /* no need to synchronize here, mTxmq synchronizes */
}

template<Dimension NDIM>
SCOPE
std::array<Slice, NDIM> get_child_slice(Key<NDIM> key, std::size_t K, int child) {
  std::array<Slice,NDIM> slices;
  for (size_t d = 0; d < NDIM; ++d) {
    int b = (child>>d) & 0x1;
    slices[d] = Slice(K*b, K*(b+1));
  }
  return slices;
}

template <typename T, Dimension NDIM>
DEVSCOPE void gaxpy_kernel_impl(
  const T* nodeA, const T* nodeB, T* nodeR,
  const T scalarA, const T scalarB, std::size_t K)
{
  const bool is_t0 = 0 == (threadIdx.x + threadIdx.y + threadIdx.z);
  SHARED TensorView<T, NDIM> nA, nB, nR;
  if (is_t0) {
    nA = TensorView<T, NDIM>(nodeA, K);
    nB = TensorView<T, NDIM>(nodeB, K);
    nR = TensorView<T, NDIM>(nodeR, K);
  }
  SYNCTHREADS();

  foreach_idx(nA, [&](auto... idx) {
    nR(idx...) = scalarA*nA(idx...) + scalarB*nB(idx...);
  });
}

template <typename T, Dimension NDIM>
GLOBALSCOPE void gaxpy_kernel(
  const T* nodeA, const T* nodeB, T* nodeR,
  const int* idxs, const T scalarA, const T scalarB,
  std::size_t N, std::size_t K, const Key<NDIM>& key)
{
  const size_t K2NDIM = std::pow(K, NDIM);
  /* adjust pointers for the function of each block */
  int blockid = blockIdx.x;

  if (idxs[blockid] >= 0){
    int fbIdx = idxs[blockid];
    gaxpy_kernel_impl<T, NDIM>(nullptr == nodeA ? nullptr : &nodeA[K2NDIM*blockid],
                               nullptr == nodeB ? nullptr : &nodeB[K2NDIM*fbIdx],
                               &nodeR[K2NDIM*blockid],
                               scalarA, scalarB, K);
  }
}


// funcA = [f0, f1, f2, f3];
// funcB = [g0, g1, g2, g3];
// idxs = [1, 2, 3, -1];  // index of functions in funcB to add to corresponding
// functions in funcA. -1 means no function to add

// funcR = [{0, 1}, {1, 2}, {3, 0}]; // result of adding {funcA[i], funcB[idxs[i]]}

template <typename T, Dimension NDIM>
void submit_gaxpy_kernel(
  const Key<NDIM>& key,
  const TensorView<T, NDIM+1>& funcA,
  const TensorView<T, NDIM+1>& funcB,
  TensorView<T, NDIM+1>& funcR,
  const int* idxs,
  const T scalarA,
  const T scalarB,
  std::size_t N,
  std::size_t K,
  cudaStream_t stream)
{
  Dim3 thread_dims = Dim3(K, K, 1);

  CALL_KERNEL(gaxpy_kernel, N, thread_dims, 0, stream,
    (funcA.data(), funcB.data(), funcR.data(), idxs, scalarA, scalarB, N, K, key));
  checkSubmit();
}

/**
 * Instantiate for 3D Gaussian
 */
template
void submit_gaxpy_kernel<double, 3>(
  const Key<3>& key,
  const TensorView<double, 4>& funcA,
  const TensorView<double, 4>& funcB,
  TensorView<double, 4>& funcR,
  const int* idxs,
  const double scalarA,
  const double scalarB,
  std::size_t K,
  std::size_t N,
  cudaStream_t stream);

template <typename T, Dimension NDIM>
DEVSCOPE void mul_kernel_impl(
  const T* nodeA, const T* nodeB, T* nodeR, std::size_t K)
{
  const bool is_t0 = 0 == (threadIdx.x + threadIdx.y + threadIdx.z);
  SHARED TensorView<T, NDIM> nA, nB, nR;
  if (is_t0) {
    nA = TensorView<T, NDIM>(nodeA, K);
    nB = TensorView<T, NDIM>(nodeB, K);
    nR = TensorView<T, NDIM>(nodeR, K);
  }
  SYNCTHREADS();

  foreach_idx(nA, [&](auto... idx) {
      nR(idx...) = nA(idx...) * nB(idx...);
  });
}

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
      auto kl = key.translation();
      auto cl = child.translation();
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

  fcoeffs_kernel_impl(D, gldata, fns[blockid], key, K, &tmp[(project_tmp_size<NDIM>(K)*blockid)],
                      phibar_ptr, coeffs_ptr+(blockid*K2NDIM), hgT_ptr, &is_leaf[blockid], thresh);
}

template<typename Fn, typename T, Dimension NDIM>
void submit_fcoeffs_kernel(
  const Domain<NDIM>& D,
  const T* gldata,
  const Fn* fns,
  const Key<NDIM>& key,
  std::size_t N,
  std::size_t K,
  TensorView<T, NDIM+1>& coeffs_view,
  const TensorView<T, 2>& phibar_view,
  const TensorView<T, 2>& hgT_view,
  T* tmp,
  bool* is_leaf_scratch,
  T thresh,
  cudaStream_t stream)
{
  /**
   * Launch the kernel with KxK threads in each of the N blocks.
   * Computation on functions is embarassingly parallel and no
   * synchronization is required.
   */
  Dim3 thread_dims = Dim3(K, K, 1); // figure out how to consider register usage
  /* launch one block per child */
  CALL_KERNEL(fcoeffs_kernel, N, thread_dims, 0, stream,
    (D, gldata, fns, key, N, K, tmp, phibar_view.data(),
    coeffs_view.data(), hgT_view.data(),
    is_leaf_scratch, thresh));
  checkSubmit();
}


template<typename Fn, typename T, Dimension NDIM>
GLOBALSCOPE void fcoeffs_kernel(
  const Domain<NDIM>& D,
  const T* gldata,
  const Fn& fn,
  Key<NDIM> key,
  std::size_t K,
  T* tmp,
  const T* phibar_ptr,
  T* coeffs_ptr,
  const T* hgT_ptr,
  bool *is_leaf,
  T thresh)
{
  fcoeffs_kernel_impl(D, gldata, fn, key, K, tmp,
                      phibar_ptr, coeffs_ptr, hgT_ptr, is_leaf, thresh);
}

template<typename Fn, typename T, Dimension NDIM>
void submit_fcoeffs_kernel(
  const Domain<NDIM>& D,
  const T* gldata,
  const Fn& fns,
  const Key<NDIM>& key,
  std::size_t K,
  TensorView<T, NDIM>& coeffs_view,
  const TensorView<T, 2>& phibar_view,
  const TensorView<T, 2>& hgT_view,
  T* tmp,
  bool* is_leaf_scratch,
  T thresh,
  cudaStream_t stream)
{
  /**
   * Launch the kernel with KxK threads in each of the N blocks.
   * Computation on functions is embarassingly parallel and no
   * synchronization is required.
   */
  Dim3 thread_dims = Dim3(K, K, 1); // figure out how to consider register usage
  /* launch one block per child */
  CALL_KERNEL(fcoeffs_kernel, 1, thread_dims, 0, stream,
    (D, gldata, fns, key, K, tmp, phibar_view.data(),
    coeffs_view.data(), hgT_view.data(),
    is_leaf_scratch, thresh));
  checkSubmit();
}

/**
 * Instantiate for 3D Gaussian
 */

 template
 void submit_fcoeffs_kernel<Gaussian<double, 3>, double, 3>(
   const Domain<3>& D,
   const double* gldata,
   const Gaussian<double, 3>* fns,
   const Key<3>& key,
   std::size_t N,
   std::size_t K,
   TensorView<double, 3+1>& coeffs_view,
   const TensorView<double, 2>& phibar_view,
   const TensorView<double, 2>& hgT_view,
   double* tmp,
   bool* is_leaf_scratch,
   double thresh,
   cudaStream_t stream);


 template
 void submit_fcoeffs_kernel<Gaussian<double, 3>, double, 3>(
   const Domain<3>& D,
   const double* gldata,
   const Gaussian<double, 3>& fns,
   const Key<3>& key,
   std::size_t K,
   TensorView<double, 3>& coeffs_view,
   const TensorView<double, 2>& phibar_view,
   const TensorView<double, 2>& hgT_view,
   double* tmp,
   bool* is_leaf_scratch,
   double thresh,
   cudaStream_t stream);



/**
 * Compress kernels
 */

template<typename T, Dimension NDIM>
DEVSCOPE void compress_kernel_impl(
  Key<NDIM> key,
  std::size_t K,
  T* p_ptr,
  T* result_ptr,
  const T* hgT_ptr,
  T* tmp,
  T* d_sumsq,
  const std::array<const T*, Key<NDIM>::num_children()>& in_ptrs)
{
  const bool is_t0 = 0 == (threadIdx.x + threadIdx.y + threadIdx.z);
  {   // Collect child coeffs and leaf info
    /* construct tensors */
    const size_t K2NDIM    = std::pow(  K,NDIM);
    const size_t TWOK2NDIM = std::pow(2*K,NDIM);
    SHARED TensorView<T,NDIM> s, d, p, workspace;
    SHARED TensorView<T,2> hgT;
    if (is_t0) {
      s = TensorView<T,NDIM>(&tmp[0], 2*K);
      workspace = TensorView<T, NDIM>(&tmp[TWOK2NDIM], 2*K);
      d = TensorView<T,NDIM>(result_ptr, 2*K);
      p = TensorView<T,NDIM>(p_ptr, K);
      hgT = TensorView<T,2>(hgT_ptr, 2*K);
    }
    SYNCTHREADS();
    d = 0.0;
    p = 0.0;

    for (int i = 0; i < Key<NDIM>::num_children(); ++i) {
      auto child_slice = get_child_slice<NDIM>(key, K, i);
      const TensorView<T, NDIM> in(in_ptrs[i], K);
      s(child_slice) = in;
    }
    //filter<T,K,NDIM>(s,d);  // Apply twoscale transformation
    transform<NDIM>(s, hgT, d, workspace);
    if (key.level() > 0) {
      auto child_slice = get_child_slice<NDIM>(key, K, 0);
      p = d(child_slice);
      d(child_slice) = 0.0;
    }
    sumabssq(d, d_sumsq);
  }
}

template<typename T, Dimension NDIM>
GLOBALSCOPE void compress_kernel(
  Key<NDIM> key,
  std::size_t N,
  std::size_t K,
  T* p_ptr,
  T* result_ptr,
  const T* hgT_ptr,
  T* tmp,
  T* d_sumsq,
  const std::array<const T*, Key<NDIM>::num_children()> in_ptrs)
{
  const bool is_t0 = (0 == (threadIdx.x + threadIdx.y + threadIdx.z));
  int blockid = blockIdx.x;
  const size_t K2NDIM    = std::pow(  K,NDIM);
  const size_t TWOK2NDIM = std::pow(2*K,NDIM);
  SHARED std::array<const T*, Key<NDIM>::num_children()> block_in_ptrs;
  if (is_t0) {
    for (std::size_t i = 0; i < Key<NDIM>::num_children(); ++i) {
      block_in_ptrs[i] = (nullptr != in_ptrs[i]) ? &in_ptrs[i][K2NDIM*blockid] : nullptr;
    }
  }
  /* no need to sync threads here */
  compress_kernel_impl(key, K, &p_ptr[K2NDIM*blockid], &result_ptr[TWOK2NDIM*blockid],
                       hgT_ptr, &tmp[compress_tmp_size<NDIM>(K)*blockid], &d_sumsq[blockid],
                       block_in_ptrs);
}

template<typename T, Dimension NDIM>
void submit_compress_kernel(
  const Key<NDIM>& key,
  std::size_t N,
  std::size_t K,
  TensorView<T, NDIM+1>& p_view,
  TensorView<T, NDIM+1>& result_view,
  const TensorView<T, 2>& hgT_view,
  T* tmp,
  T* d_sumsq,
  const std::array<const T*, Key<NDIM>::num_children()>& in_ptrs,
  cudaStream_t stream)
{
  Dim3 thread_dims = Dim3(K, K, 1); // figure out how to consider register usage

  CALL_KERNEL(compress_kernel, N, thread_dims, 0, stream,
    (key, N, K, p_view.data(), result_view.data(), hgT_view.data(), tmp, d_sumsq, in_ptrs));
  checkSubmit();
}


template<typename T, Dimension NDIM>
GLOBALSCOPE void compress_kernel(
  Key<NDIM> key,
  std::size_t K,
  T* p_ptr,
  T* result_ptr,
  const T* hgT_ptr,
  T* tmp,
  T* d_sumsq,
  const std::array<const T*, Key<NDIM>::num_children()> in_ptrs)
{
  const size_t K2NDIM    = std::pow(  K,NDIM);
  const size_t TWOK2NDIM = std::pow(2*K,NDIM);
  /* no need to sync threads here */
  compress_kernel_impl(key, K, p_ptr, result_ptr,
                       hgT_ptr, tmp, d_sumsq,
                       in_ptrs);
}

template<typename T, Dimension NDIM>
void submit_compress_kernel(
  const Key<NDIM>& key,
  std::size_t K,
  TensorView<T, NDIM>& p_view,
  TensorView<T, NDIM>& result_view,
  const TensorView<T, 2>& hgT_view,
  T* tmp,
  T* d_sumsq,
  const std::array<const T*, Key<NDIM>::num_children()>& in_ptrs,
  cudaStream_t stream)
{
  Dim3 thread_dims = Dim3(K, K, 1); // figure out how to consider register usage

  CALL_KERNEL(compress_kernel, 1, thread_dims, 0, stream,
    (key, K, p_view.data(), result_view.data(), hgT_view.data(), tmp, d_sumsq, in_ptrs));
  checkSubmit();
}


/* Instantiations for 3D */
template
void submit_compress_kernel<double, 3>(
  const Key<3>& key,
  std::size_t N,
  std::size_t K,
  TensorView<double, 3+1>& p_view,
  TensorView<double, 3+1>& result_view,
  const TensorView<double, 2>& hgT_view,
  double* tmp,
  double* d_sumsq,
  const std::array<const double*, Key<3>::num_children()>& in_ptrs,
  cudaStream_t stream);

/* Instantiations for 3D */
template
void submit_compress_kernel<double, 3>(
  const Key<3>& key,
  std::size_t K,
  TensorView<double, 3>& p_view,
  TensorView<double, 3>& result_view,
  const TensorView<double, 2>& hgT_view,
  double* tmp,
  double* d_sumsq,
  const std::array<const double*, Key<3>::num_children()>& in_ptrs,
  cudaStream_t stream);



/**
 * kernel for reconstruct
 */

template<typename T, Dimension NDIM>
DEVSCOPE void reconstruct_kernel_impl(
  Key<NDIM> key,
  std::size_t K,
  T* node_ptr,
  bool node_empty,
  T* tmp_ptr,
  const T* hg_ptr,
  const T* from_parent_ptr,
  std::array<T*, Key<NDIM>::num_children()>& r_arr)
{
  const bool is_t0 = (0 == (threadIdx.x + threadIdx.y + threadIdx.z));
  const size_t TWOK2NDIM = std::pow(2*K,NDIM);
  SHARED TensorView<T, NDIM> node, s, workspace, from_parent;
  SHARED TensorView<T, 2> hg;
  if (is_t0) {
    node        = TensorView<T, NDIM>(node_ptr, 2*K);
    s           = TensorView<T, NDIM>(&tmp_ptr[0], 2*K);
    workspace   = TensorView<T, NDIM>(&tmp_ptr[TWOK2NDIM], 2*K);
    hg          = TensorView<T, 2>(hg_ptr, 2*K);
    from_parent = TensorView<T, NDIM>(from_parent_ptr, K);
  }
  SYNCTHREADS();
  s = 0.0;

  if (node_empty) {
    /* if the node was empty we reset it to zero */
    node = 0.0;
  }

  auto child_slice = get_child_slice<NDIM>(key, K, 0);
  if (key.level() != 0) node(child_slice) = from_parent;

  //unfilter<T,K,NDIM>(node.get().coeffs, s);
  transform<NDIM>(node, hg, s, workspace);

  /* extract all r from s
   * NOTE: we could do this on 1<<NDIM blocks but the benefits would likely be small */
  for (int i = 0; i < key.num_children(); ++i) {
    auto child_slice = get_child_slice<NDIM>(key, K, i);
    /* tmp layout: 2K^NDIM for s, K^NDIM for workspace, [K^NDIM]* for r fields */
    auto r = TensorView<T, NDIM>(r_arr[i], K);
    r = s(child_slice);
  }
}


template<typename T, Dimension NDIM>
GLOBALSCOPE void reconstruct_kernel(
  Key<NDIM> key,
  std::size_t N,
  std::size_t K,
  T* node_ptr,
  bool node_empty,
  T* tmp_ptr,
  const T* hg_ptr,
  const T* from_parent_ptr,
  std::array<T*, Key<NDIM>::num_children()> r_arr)
{
  const bool is_t0 = (0 == (threadIdx.x + threadIdx.y + threadIdx.z));
  int blockid = blockIdx.x;
  const size_t TWOK2NDIM = std::pow(2*K,NDIM);
  const size_t K2NDIM    = std::pow(  K,NDIM);

  /* pick the r's for this function */
  SHARED std::array<T*, Key<NDIM>::num_children()> block_r_arr;
  if (is_t0) {
    for (std::size_t i = 0; i < Key<NDIM>::num_children(); ++i) {
      block_r_arr[i] = &r_arr[i][K2NDIM*blockid];
    }
  }
  /* no need to sync threads here, the impl will sync before the r_arr are used */
  reconstruct_kernel_impl(key, K, &node_ptr[TWOK2NDIM*blockid], node_empty,
                          tmp_ptr + blockid*reconstruct_tmp_size<NDIM>(K),
                          hg_ptr, &from_parent_ptr[K2NDIM*blockid],
                          block_r_arr);
}

template<typename T, Dimension NDIM>
void submit_reconstruct_kernel(
  const Key<NDIM>& key,
  std::size_t N,
  std::size_t K,
  TensorView<T, NDIM+1>& node,
  bool node_empty,
  const TensorView<T, 2>& hg,
  const TensorView<T, NDIM+1>& from_parent,
  const std::array<T*, mra::Key<NDIM>::num_children()>& r_arr,
  T* tmp,
  cudaStream_t stream)
{
  /* runs on a single block */
  Dim3 thread_dims = Dim3(K, K, 1); // figure out how to consider register usage
  CALL_KERNEL(reconstruct_kernel, N, thread_dims, 0, stream,
    (key, N, K, node.data(), node_empty, tmp, hg.data(), from_parent.data(), r_arr));
  checkSubmit();
}


template<typename T, Dimension NDIM>
GLOBALSCOPE void reconstruct_kernel(
  Key<NDIM> key,
  std::size_t K,
  T* node_ptr,
  bool node_empty,
  T* tmp_ptr,
  const T* hg_ptr,
  const T* from_parent_ptr,
  std::array<T*, Key<NDIM>::num_children()> r_arr)
{
  const size_t TWOK2NDIM = std::pow(2*K,NDIM);
  const size_t K2NDIM    = std::pow(  K,NDIM);

  /* no need to sync threads here, the impl will sync before the r_arr are used */
  reconstruct_kernel_impl(key, K, node_ptr, node_empty, tmp_ptr,
                          hg_ptr, from_parent_ptr, r_arr);
}

template<typename T, Dimension NDIM>
void submit_reconstruct_kernel(
  const Key<NDIM>& key,
  std::size_t K,
  TensorView<T, NDIM>& node,
  bool node_empty,
  const TensorView<T, 2>& hg,
  const TensorView<T, NDIM>& from_parent,
  const std::array<T*, mra::Key<NDIM>::num_children()>& r_arr,
  T* tmp,
  cudaStream_t stream)
{
  /* runs on a single block */
  Dim3 thread_dims = Dim3(K, K, 1); // figure out how to consider register usage
  CALL_KERNEL(reconstruct_kernel, 1, thread_dims, 0, stream,
    (key, K, node.data(), node_empty, tmp, hg.data(), from_parent.data(), r_arr));
  checkSubmit();
}

/* explicit instantiation for 3D */
template
void submit_reconstruct_kernel<double, 3>(
  const Key<3>& key,
  std::size_t N,
  std::size_t K,
  TensorView<double, 3+1>& node,
  bool node_empty,
  const TensorView<double, 2>& hg,
  const TensorView<double, 4>& from_parent,
  const std::array<double*, Key<3>::num_children()>& r_arr,
  double* tmp,
  cudaStream_t stream);

/* explicit instantiation for 3D */
template
void submit_reconstruct_kernel<double, 3>(
  const Key<3>& key,
  std::size_t K,
  TensorView<double, 3>& node,
  bool node_empty,
  const TensorView<double, 2>& hg,
  const TensorView<double, 3>& from_parent,
  const std::array<double*, Key<3>::num_children()>& r_arr,
  double* tmp,
  cudaStream_t stream);
