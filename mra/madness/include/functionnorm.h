#ifndef HAVE_MRA_FUNCTIONNORM_H
#define HAVE_MRA_FUNCTIONNORM_H

#include "key.h"
#include "tensor.h"
#include "functionnode.h"
#include "kernels/simple_norm.h"


namespace mra {

#ifdef MRA_CHECK_NORMS
    template<typename T, Dimension NDIM>
    class FunctionNorms {

    public:
      using value_type = T;

    private:
      detail::FunctionNodeBase<T, NDIM>& m_node;
      mra::Tensor<T, 1> m_norms;
      bool m_initial = false;

    public:
      template<typename NodeType>
      FunctionNorms(NodeType&& node)
      : m_node(const_cast<std::decay_t<NodeType>&>(node))
      , m_norms()
      , m_initial(node.norms().empty())
      {
        if (m_initial && std::is_const_v<NodeType>) {
          throw std::runtime_error("Cannot compute norms for a const node!");
        }
        if (!m_initial) {
          m_norms = Tensor<T, 1>(node.count(), ttg::scope::Allocate);
        } else {
          m_node.norms() = Tensor<T, 1>(m_node.count());
        }
      }

      auto& buffer() {
        if (m_initial) {
          /* we will fill the norms of the node */
          return m_node.norms().buffer();
        } else {
          /* compute the norms in our local buffer */
          return m_norms.buffer();
        }
      }

      void compute() {
        if (!m_node.empty()) {
          if (m_initial) {
            assert(m_node.norms().buffer().is_current_on(ttg::device::current_device()));
            assert(!m_node.norms().buffer().empty());
            /* we will fill the norms of the node */
            submit_simple_norm_kernel(m_node.key(), m_node.coeffs().current_view(), m_node.count(), m_node.norms().current_view());
          } else {
            assert(m_norms.buffer().is_current_on(ttg::device::current_device()));
            assert(!m_norms.buffer().empty());
            /* we will compute the norms in our local buffer */
            submit_simple_norm_kernel(m_node.key(), m_node.coeffs().current_view(), m_node.count(), m_norms.current_view());
          }
        }
      }

      /**
       * Verify the norms computed by compute() against the norms of the node.
       * Only performs the validation if the norms was not computed initially by this object.
       * Throws a runtime_error if the norms do not match.
       * Returns true if the norms match.
       * Returns false if the norms were computed initially by this object and no verification is needed.
       */
      bool verify(const std::string& name) const {
        if (!m_node.empty()) {
          if (!m_initial) {
            assert(m_node.norms().buffer().is_valid_on(ttg::device::Device::host()));
            auto* node_norms = m_node.norms().data();
            assert(m_norms.buffer().is_current_on(ttg::device::Device::host()));
            auto* norms = m_norms.current_view().data();
            for (size_type i = 0; i < m_node.count(); ++i) {
              if (std::abs(node_norms[i] - norms[i]) > 1e-15) {
                std::cerr << name << ": failed to verify norm for function " << i << " of " << m_node.key()
                          << ": expected " << norms[i] << ", found " << node_norms[i] << std::endl;
                throw std::runtime_error("Failed to verify norm!");
              }
            }
          }
          return true;
        }
        return false;
      }
    };

#else // MRA_CHECK_NORMS

    template<typename T, Dimension NDIM>
    class FunctionNorms {

    public:
      using value_type = T;

    private:

    public:
      FunctionNorms(const detail::FunctionNodeBase<T, NDIM>& node)
      { }

      auto buffer() {
        return ttg::Buffer<value_type>();
      }

      void compute() {
        return;
      }

      bool verify(std::string name) const {
        return false;
      }
    };
#endif // MRA_CHECK_NORMS

    // deduction guide
    template<typename NodeT>
    FunctionNorms(NodeT&&) -> FunctionNorms<typename std::decay_t<NodeT>::value_type, std::decay_t<NodeT>::ndim()>;

} // namespace mra

#endif // HAVE_MRA_FUNCTIONNORM_H
