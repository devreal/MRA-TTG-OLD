#ifndef MRA_TWOSCALE_H
#define MRA_TWOSCALE_H

#include "mra/misc/types.h"
namespace mra {

    namespace detail {

        /// Copies the multiwavelet twoscale coefficients into p which should be [2k][2k] and either double or float
        template <typename T>
        void twoscale_get(size_type k, T* p);

        /// Returns true if twoscale coeffs pass basic correctness tests
        bool twoscale_check();
    }

    /// Get the multiwavelet twoscale coefficients of order K (1<=K<=60) and either double or float
    template <typename T>
    void twoscale_get(size_type k, T* p) {
        detail::twoscale_get(k, p);
    }
} // namespace mra

#endif // MRA_TWOSCALE_H
