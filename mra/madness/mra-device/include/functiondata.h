#ifndef MADFUNCTIONDATA_H_INCL
#define MADFUNCTIONDATA_H_INCL

#include "types.h"
#include "domain.h"
#include "../../mratwoscale.h"
#include "tensor.h"
#include "gl.h"

namespace mra {

    /// Convenient co-location of frequently used data
    template <typename T, Dimension NDIM>
    class FunctionData {
        std::size_t K;
        Tensor<T,2> phi; // phi(mu,i) = phi(x[mu],i) --- value of scaling functions at quadrature points on level 0
        Tensor<T,2> phiT; // transpose of phi
        Tensor<T,2> phibar; // phibar(mu,i) = w[mu]*phi(x[mu],i)
        Tensor<T,2> HG; // Two scale filter applied from left to scaling function coeffs
        Tensor<T,2> HGT; // Two scale filter applied from right to scaling function coeffs
        Tensor<T,2> rm, r0, rp; // blocks of the ABGV central derivative operator
        std::unique_ptr<T[]> x, w; // Quadrature points and weights on level 0

        void make_abgv_diff_operator() {
            double iphase = 1.0;
            auto r0_view = r0.current_view();
            auto rm_view = rm.current_view();
            auto rp_view = rp.current_view();
            for (std::size_t i = 0; i < K; ++i) {
                double jphase = 1.0;
                for (std::size_t j = 0; j < K; ++j) {
                    double gammaij = std::sqrt(double((2*i+1)*(2*j+1)));
                    double Kij;
                    if (((i-j)>0) && (((i-j)%2)==1))
                        Kij = 2.0;
                    else
                        Kij = 0.0;

                    r0_view(i,j) = T(0.5*(1.0 - iphase*jphase - 2.0*Kij)*gammaij);
                    rm_view(i,j) = T(0.5*jphase*gammaij);
                    rp_view(i,j) = T(-0.5*iphase*gammaij);
                }
            }
        }

        /// Set phi(mu,i) to be phi(x[mu],i)
        void make_phi() {
            /* retrieve x, w from constant memory */
            const T *x, *w;
            GLget(&x, &w, K);
            T* p = new T[K];
            auto phi_view = phi.current_view();
            for (std::size_t mu = 0; mu < K; ++mu) {
                legendre_scaling_functions(x[mu], K, &p[0]);
                for (std::size_t i = 0; i < K; ++i) {
                    phi_view(mu,i) = p[i];
                }
            }
            delete[] p;
        }

        /// Set phiT(mu,i) to be phiT(x[mu],i)
        void make_phiT() {
            /* retrieve x, w from constant memory */
            const T *x, *w;
            GLget(&x, &w, K);
            T* p = new T[K];
            auto phiT_view = phi.current_view();
            for (std::size_t mu = 0; mu < K; ++mu) {
                legendre_scaling_functions(x[mu], K, &p[0]);
                for (std::size_t i = 0; i < K; ++i) {
                    phiT_view(i, mu) = p[i];
                }
            }
            delete[] p;
        }

        /// Set phibar(mu,i) to be w[mu]*phi(x[mu],i)
        void make_phibar() {
            /* retrieve x, w from constant memory */
            const T *x, *w;
            GLget(&x, &w, K);
            T *p = new T[K];
            auto phibar_view = phibar.current_view();
            for (std::size_t mu = 0; mu < K; ++mu) {
                legendre_scaling_functions(x[mu], K, &p[0]);
                for (std::size_t i = 0; i < K; ++i) {
                    phibar_view(mu,i) = w[mu]*p[i];
                }
            }
            delete[] p;
            // FixedTensor<T,K,2> phi, r;
            // make_phi<T,K>(phi);
            // mTxmq(K, K, K, r.ptr(), phi.ptr(), phibar.ptr());
            // std::cout << r << std::endl; // should be identify matrix
        }

    public:

        FunctionData(std::size_t K)
        : K(K)
        , phi(K, K)
        , phiT(K, K)
        , phibar(K, K)
        , HG(2*K, 2*K)
        , HGT(2*K, 2*K)
        , rm(K, K)
        , r0(K, K)
        , rp(K, K)
        {
            make_phi();
            make_phiT();
            make_phibar();
            twoscale_get(K, HG.data());
            auto HG_view  = HG.current_view();
            auto HGT_view = HGT.current_view();
            for (std::size_t i = 0; i < 2*K; ++i) {
                for (std::size_t j = 0; j < 2*K; ++j) {
                    HGT_view(j,i) = HG_view(i,j);
                }
            }
            make_abgv_diff_operator();
        }

        FunctionData(FunctionData&&) = default;
        FunctionData(const FunctionData&) = delete;
        FunctionData& operator=(FunctionData&&) = default;
        FunctionData& operator=(const FunctionData&) = delete;

        const auto& get_phi() const {return phi;}
        const auto& get_phiT() const {return phiT;}
        const auto& get_phibar() const {return phibar;}
        const auto& get_hg() const {return HG;}
        const auto& get_hgT() const {return HGT;}
        const auto& get_rm() const {return rm;}
        const auto& get_r0() const {return r0;}
        const auto& get_rp() const {return rp;}
    };

}


#endif
