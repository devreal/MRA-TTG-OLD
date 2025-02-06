#ifndef FCUBE_FOR_MUL_H_INCL
#define FCUBE_FOR_MUL_H_INCL

#include "tensorview.h"
#include "platform.h"
#include "functiondata.h"
#include "transform.h"
#include "domain.h"
#include "key.h"
#include "types.h"
#include <cassert>

namespace mra {

  template <typename T, Dimension NDIM>
  SCOPE void phi_for_mul(
    const Level np,
    const Level nc,
    const Translation lp,
    const Translation lc,
    TensorView<T, 2>& phi,
    const TensorView<T, 1>& quad_x,
    const size_type K)
  {
    T p[200]; // TODO: fix this to come from workspace
    T scale = pow(2.0, T(np-nc));

    for(size_type mu = 0; mu < K; ++mu) {
      T xmu = scale * (quad_x(mu) + lc) - lp;
      assert(xmu > 1e-15 && xmu < 1.0 + 1e-15);
      legendre_scaling_functions(xmu, K, p);
      for (size_type i = 0; i < K; ++i) phi(i, mu) = p[i];
    }
    T scale_phi = pow(2.0, 0.5*np);
    phi *= scale_phi;
  }


  template <typename T, Dimension NDIM>
  SCOPE void fcube_for_mul(
    const Domain<NDIM>& D,
    const Key<NDIM>& child,
    const Key<NDIM>& parent,
    const TensorView<T,NDIM>& coeffs,
    TensorView<T, NDIM>& result_values,
    const TensorView<T, 2>& phi_old,
    const TensorView<T, 2>& phibar,
    const TensorView<T, 1>& quad_x,
    const size_type K,
    T* workspace)
  {
    if (child.level() < parent.level()) {
      throw std::logic_error("fcube_for_mul: bad child-parent relationship");
    }
    else if (child.level() == parent.level()) {
      // coeffs_to_values()
      transform(coeffs, phibar, result_values, workspace);
      T scale = pow(2.0, 0.5*NDIM*parent.level())/sqrt(D.template get_volume<T>());
      result_values *= scale;
    }
    else {
#ifdef HAVE_DEVICE_ARCH
      extern SHARED T phi[];
#else
      T* phi = new T[K*K*NDIM];
#endif
      SHARED std::array<TensorView<T, 2>, NDIM> phi_views;
      if(is_team_lead()){
        for (int d = 0; d < NDIM; ++d){
          phi_views[d] = TensorView<T, 2>(&phi[d*K*K], K, K);
        }
      }
      SYNCTHREADS();

      for (size_type d=0; d < NDIM; ++d){
        auto parent_l = parent.translation();
        auto child_l = child.translation();
        phi_for_mul<T, NDIM>(parent.level(), child.level(), parent_l[d], child_l[d], phi_views[d], quad_x, K);
      }

      general_transform(coeffs, phi_views, result_values);
      T scale = T(1)/sqrt(D.template get_volume<T>());
      result_values *= scale;
#ifndef HAVE_DEVICE_ARCH
      delete[] phi;
#endif
    }

  }

} // namespace mra

#endif // FCUBE_FOR_MUL_H_INCL
