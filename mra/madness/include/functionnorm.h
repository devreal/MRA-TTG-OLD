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
      std::vector<detail::FunctionNodeBase<T, NDIM>*> m_nodes;
      std::vector<bool> m_initial;
      mra::Tensor<T, 2> m_norms; // M x N matrix of norms (M: number of nodes, N: number of functions)

    public:
      template<typename NodeT, typename... NodeTs>
      FunctionNorms(NodeT&& node, NodeTs&&... nodes)
      : m_nodes({&const_cast<std::decay_t<NodeT>&>(node), &const_cast<std::decay_t<NodeTs>&>(nodes)...})
      , m_initial({node.norms().empty(), nodes.norms().empty()...})
      {
#ifndef MRA_ENABLE_HOST
        m_norms = Tensor<T, 2>({static_cast<size_type>(sizeof...(NodeTs))+1, node.count()}, ttg::scope::Allocate);
#else
        m_norms = Tensor<T, 2>({static_cast<size_type>(sizeof...(NodeTs))+1, node.count()}, ttg::scope::SyncIn);
#endif // MRA_ENABLE_HOST
        for (int i = 0; i < m_nodes.size(); ++i) {
          auto& node = *m_nodes[i];
          if (m_initial[i]) {
            // copy the norms into the buffer
            node.norms() = typename detail::FunctionNodeBase<T, NDIM>::norm_tensor_type(node.count());
          }
        }
      }

      auto& buffer() {
        return m_norms.buffer();
      }

      void compute() {
        assert(m_norms.buffer().is_current_on(ttg::device::current_device()));
        assert(!m_norms.buffer().empty());
        /* we will compute the norms in our local buffer */
        for (int i = 0; i < m_nodes.size(); ++i) {
          auto& node = *m_nodes[i];
          if (m_initial[i] && !node.empty()){
            submit_simple_norm_kernel(node.key(), node.coeffs().current_view(), node.count(), m_norms.current_view()(i));
          }
        }
      }

      /**
       * Verify the norms computed by compute() against the norms of the node.
       * Only performs the validation if the norms was not computed initially by this object.
       * Throws a runtime_error if the norms do not match.
       */
      void verify(const std::string& name) const {
        assert(m_norms.buffer().is_current_on(ttg::device::current_device()));
        for (int i = 0; i < m_nodes.size(); ++i) {
          auto& node = *m_nodes[i];
          if (node.empty()) continue;
          if (!m_initial[i]) {
            // verify the norms
            assert(node.norms().buffer().is_valid_on(ttg::device::current_device()));
            auto* node_norms = node.norms().data();
            auto norm_view = m_norms.current_view()(i);
            for (size_type j = 0; j < node.count(); ++j) {
              if (std::abs(node_norms[j] - norm_view[j]) > 1e-15) {
                std::cerr << name << ": failed to verify norm for function " << j << " of " << node.key()
                          << ": expected " << norm_view[j] << ", found " << node_norms[j] << std::endl;
                throw std::runtime_error("Failed to verify norm!");
              }
            }
          } else {
            // store the norm into the node
            node.norms().current_view() = m_norms.current_view()(i);
          }
        }
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
    template<typename NodeT, typename... NodeTs>
    FunctionNorms(NodeT&&, NodeTs...) -> FunctionNorms<typename std::decay_t<NodeT>::value_type, std::decay_t<NodeT>::ndim()>;

} // namespace mra

#endif // HAVE_MRA_FUNCTIONNORM_H
