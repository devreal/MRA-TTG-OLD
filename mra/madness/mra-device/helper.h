#ifndef MADHELPER_H_INCL
#define MADHELPER_H_INCL

#include "tensorview.h"
#include "util.h"

namespace mra{
    /// Make outer product of quadrature points for vectorized algorithms
    template<typename T>
    SCOPE void make_xvec(const TensorView<T,2>& x, TensorView<T,2>& xvec,
                            std::integral_constant<Dimension, 1>) {
        /* uses threads in 3 dimensions */
        xvec = x;
        /* TensorView assignment synchronizes */
    }

    /// Make outer product of quadrature points for vectorized algorithms
    template<typename T>
    SCOPE void make_xvec(const TensorView<T,2>& x, TensorView<T,2>& xvec,
                            std::integral_constant<Dimension, 2>) {
        const std::size_t K = x.dim(1);
        if (threadIdx.z == 0) {
            for (size_t i=threadIdx.y; i<K; i += blockDim.y) {
                for (size_t j=threadIdx.x; j<K; j += blockDim.x) {
                    size_t ij = i*K + j;
                    xvec(0,ij) = x(0,i);
                    xvec(1,ij) = x(1,j);
                }
            }
        }
        SYNCTHREADS();
    }

    /// Make outer product of quadrature points for vectorized algorithms
    template<typename T>
    SCOPE void make_xvec(const TensorView<T,2>& x, TensorView<T,2>& xvec,
                            std::integral_constant<Dimension, 3>) {
        const std::size_t K = x.dim(1);
        for (size_t i=threadIdx.z; i<K; i += blockDim.z) {
            for (size_t j=threadIdx.y; j<K; j += blockDim.y) {
                for (size_t k=threadIdx.x; k<K; k += blockDim.x) {
                    size_t ijk = i*K*K + j*K + k;
                    xvec(0,ijk) = x(0,i);
                    xvec(1,ijk) = x(1,j);
                    xvec(2,ijk) = x(2,k);
                }
            }
        }
        SYNCTHREADS();
    }
}

#endif // MADHELPER_H_INCL
