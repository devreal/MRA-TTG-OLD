#ifndef MRA_GL_H
#define MRA_GL_H

#include <cstddef>
#include <cassert>

#include "mra/misc/platform.h"
#include "mra/misc/types.h"

#include "ttg/buffer.h"

namespace mra {

  namespace detail {
    /// Arrays for points and weights for the Gauss-Legendre quadrature on [0,1].
    /// only available directly on the host
    extern const double gl_data[128][64];

    struct empty_deleter {
      void operator()(const double*) const {}
    };
  } // namespace detail

  /**
   * Host-side functions only
   */

  template<typename T>
  inline ttg::Buffer<const T> GLbuffer() {
    return ttg::Buffer<const T>(std::unique_ptr<const T[], detail::empty_deleter>(&detail::gl_data[0][0]), sizeof(detail::gl_data)/sizeof(T));
  }

  template<typename T>
  inline void GLget(const T** x, const T **w, size_type N) {
    assert(N>0 && N<=64);
    *x = &detail::gl_data[2*(N-1)  ][0];
    *w = &detail::gl_data[2*(N-1)+1][0];
  }

  /// Evaluate the first k Legendre scaling functions. p should be an array of k elements.
  void legendre_scaling_functions(double x, size_type k, double *p);

  /// Evaluate the first k Legendre scaling functions. p should be an array of k elements.
  void legendre_scaling_functions(float x, size_type k, float *p);

  bool GLinitialize();

  template<typename T>
  SCOPE void GLget(const T* glptr, const T** x, const T **w, size_type N) {
    assert(N>0 && N<=64);
    T (*data)[64] = (T(*)[64])glptr;
    *x = &data[2*(N-1)  ][0];
    *w = &data[2*(N-1)+1][0];
  }

} // namespace mra

#endif // MRA_GL_H
