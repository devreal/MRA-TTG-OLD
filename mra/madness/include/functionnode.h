#ifndef HAVE_MRA_FUNCTIONNODE_H
#define HAVE_MRA_FUNCTIONNODE_H

#include "key.h"
#include "tensor.h"
#include "functions.h"

namespace mra {

    namespace detail {

      template<Dimension NDIM, std::size_t... Is>
      std::array<size_type, NDIM> make_dims_helper(size_type N, size_type K, std::index_sequence<Is...>) {
        return std::array<size_type, NDIM>{N, ((void)Is, K)...};
      }
      /* helper to create {N, K, K, K, ...} dims array */
      template<Dimension NDIM>
      std::array<size_type, NDIM> make_dims(size_type N, size_type K) {
        return make_dims_helper<NDIM>(N, K, std::make_index_sequence<NDIM-1>{});
      }
    } // namespace detail

    /* like FunctionReconstructedNode but for N functions */
    template <typename T, Dimension NDIM>
    class FunctionsReconstructedNode : public ttg::TTValue<FunctionsReconstructedNode<T, NDIM>> {
      public: // temporarily make everything public while we figure out what we are doing
        using key_type = Key<NDIM>;
        using tensor_type = Tensor<T,NDIM+1>;
        using view_type   = TensorView<T, NDIM>;
        using const_view_type   = TensorView<const T, NDIM>;
        static constexpr bool is_function_node = true;

      private:
        struct function_metadata {
          T sum = 0.0;
          bool is_leaf = false;
          std::array<bool, Key<NDIM>::num_children()> is_child_leaf = { false };
        };

        key_type m_key; //< Key associated with this node to facilitate computation from otherwise unknown parent/child
        std::vector<function_metadata> m_metadata;
        tensor_type m_coeffs; //< if !is_leaf these are junk (and need not be communicated)

      public:
        FunctionsReconstructedNode() = default;

        /* constructs a node with metadata for N functions and all coefficients zero */
        FunctionsReconstructedNode(const Key<NDIM>& key, size_type N)
        : m_key(key)
        , m_metadata(N)
        , m_coeffs()
        { }

        FunctionsReconstructedNode(const Key<NDIM>& key, size_type N, size_type K)
        : m_key(key)
        , m_metadata(N)
        , m_coeffs(detail::make_dims<NDIM+1>(N, K))
        {}


        FunctionsReconstructedNode(FunctionsReconstructedNode&& other) = default;
        FunctionsReconstructedNode(const FunctionsReconstructedNode& other) = delete;

        FunctionsReconstructedNode& operator=(FunctionsReconstructedNode&& other) = default;
        FunctionsReconstructedNode& operator=(const FunctionsReconstructedNode& other) = delete;

        /**
         * Allocate space for coefficients using K.
         * The node must be empty before and will not be empty afterwards.
         */
        void allocate(size_type K) {
          if (!empty()) throw std::runtime_error("Reallocating non-empty FunctionNode not allowed!");
          size_type N = m_metadata.size();
          if (N == 0) throw std::runtime_error("Cannot reallocate FunctionNode with N = 0");
          m_coeffs = Tensor<T,NDIM+1>(detail::make_dims<NDIM+1>(N, 2*K));
        }

        bool has_children(size_type i) const {
          return !m_metadata[i].is_leaf;
        }

        bool any_have_children() const {
          bool result = false;
          for (size_type i = 0; i < m_metadata.size(); ++i) {
            result |= has_children(i);
          }
          return result;
        }

        void set_all_leaf(bool val) {
          for (auto& data : m_metadata) {
            data.is_leaf = val;
          }
        }

        bool is_all_leaf() const {
          bool all_leaf = true;
          for (auto& data : m_metadata) {
            all_leaf &= data.is_leaf;
          }
          return all_leaf;
        }

        bool& is_leaf(size_type i) {
          return m_metadata[i].is_leaf;
        }

        bool is_leaf(size_type i) const {
          return m_metadata[i].is_leaf;
        }

        bool& is_child_leaf(size_type i, size_type child) {
          return m_metadata[i].is_child_leaf[child];
        }

        bool is_leaf(size_type i, size_type child) const {
          return m_metadata[i].is_child_leaf[child];
        }

        T& sum(size_type i) {
          return m_metadata[i].sum;
        }

        T sum(size_type i) const {
          return m_metadata[i].sum;
        }

        /* with C++23 we could the following:
        auto& coeffs(this FunctionsReconstructedNode&& self) {
          return self.m_coeffs;
        }
        */
        auto& coeffs() {
          return m_coeffs;
        }

        const auto& coeffs() const {
          return m_coeffs;
        }

        view_type coeffs_view(size_type i) {
          /* assuming all dims > 1 == K */
          const size_type K = m_coeffs.dim(1);
          const size_type K2NDIM = std::pow(K, NDIM);
          return view_type(m_coeffs.data() + (i*K2NDIM), K);
        }

        const_view_type coeffs_view(size_type i) const {
          /* assuming all dims > 1 == K */
          const size_type K = m_coeffs.dim(1);
          const size_type K2NDIM = std::pow(K, NDIM);
          return const_view_type(m_coeffs.data() + (i*K2NDIM), K);
        }

        key_type& key() {
          return m_key;
        }

        const key_type& key() const {
          return m_key;
        }

        size_type count() const {
          return m_metadata.size();
        }

        bool empty() const {
          return m_coeffs.empty();
        }

        template <typename Archive>
        void serialize(Archive& ar) {
          throw std::runtime_error("FunctionsCompressedNode::serialize not yet implemented");
#if 0
          ar& this->m_key;
          ar& this->m_metadata;
          ar& this->m_coeffs;
#endif // 0
        }

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int) {
          serialize(ar);
        }
    };


    template <typename T, Dimension NDIM>
    class FunctionsCompressedNode : public ttg::TTValue<FunctionsCompressedNode<T, NDIM>> {
      public: // temporarily make everything public while we figure out what we are doing
        static constexpr bool is_function_node = true;
        using key_type          = Key<NDIM>;
        using view_type         = TensorView<T, NDIM>;
        using const_view_type   = TensorView<const T, NDIM>;

      private:
        key_type m_key; //< Key associated with this node to facilitate computation from otherwise unknown parent/child
        std::vector<std::array<bool, Key<NDIM>::num_children()>> m_is_child_leafs; //< True if that child is leaf on tree
        Tensor<T,NDIM+1> m_coeffs; //< Always significant

      public:
        FunctionsCompressedNode() = default; // needed for serialization

        /* constructs a node for N functions with zero coefficients */
        FunctionsCompressedNode(const Key<NDIM>& key, size_type N)
        : m_key(key)
        , m_coeffs()
        , m_is_child_leafs(N)
        { }

        FunctionsCompressedNode(const Key<NDIM>& key, size_type N, size_type K)
        : m_key(key)
        , m_coeffs(detail::make_dims<NDIM+1>(N, 2*K))
        , m_is_child_leafs(N)
        { }

        /**
         * Allocate space for coefficients using K.
         * The node must be empty before and will not be empty afterwards.
         */
        void allocate(size_type K) {
          if (!empty()) throw std::runtime_error("Reallocating non-empty FunctionNode not allowed!");
          size_type N = m_is_child_leafs.size();
          if (N == 0) throw std::runtime_error("Cannot reallocate FunctionNode with N = 0");
          m_coeffs = Tensor<T,NDIM+1>(detail::make_dims<NDIM+1>(N, 2*K));
        }

        FunctionsCompressedNode(FunctionsCompressedNode&& other) = default;
        FunctionsCompressedNode(const FunctionsCompressedNode& other) = delete;

        FunctionsCompressedNode& operator=(FunctionsCompressedNode&& other) = default;
        FunctionsCompressedNode& operator=(const FunctionsCompressedNode& other) = delete;

        bool has_children(size_type i, int childindex) const {
            assert(childindex < Key<NDIM>::num_children());
            assert(i < m_is_child_leafs.size());
            return !m_is_child_leafs[i][childindex];
        }

        std::array<bool, Key<NDIM>::num_children()>& is_child_leaf(size_type i) {
          return m_is_child_leafs[i];
        }

        bool is_child_leaf(size_type i, size_type child) const {
          return m_is_child_leafs[i][child];
        }

        void set_all_child_leafs() {
          for (auto& node : m_is_child_leafs) {
            for (auto& c : node) {
              c = true;
            }
          }
        }

        auto& coeffs() {
          return m_coeffs;
        }

        const auto& coeffs() const {
          return m_coeffs;
        }

        key_type& key() {
          return m_key;
        }

        const key_type& key() const {
          return m_key;
        }

        size_type count() const {
          return m_is_child_leafs.size();
        }

        bool empty() const {
          return m_coeffs.empty();
        }

        view_type coeffs_view(size_type i) {
          /* assuming all dims > 1 == K */
          const size_type K = m_coeffs.dim(1);
          const size_type K2NDIM = std::pow(K, NDIM);
          return view_type(m_coeffs.data() + (i*K2NDIM), K);
        }

        const_view_type coeffs_view(size_type i) const {
          /* assuming all dims > 1 == K */
          const size_type K = m_coeffs.dim(1);
          const size_type K2NDIM = std::pow(K, NDIM);
          return const_view_type(m_coeffs.data() + (i*K2NDIM), K);
        }

        template <typename Archive>
        void serialize(Archive& ar) {
          throw std::runtime_error("FunctionsCompressedNode::serialize not yet implemented");
#if 0
          ar& this->m_key;
          ar& this->m_is_child_leafs;
          ar& this->m_coeffs;
#endif // 0
        }

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int) {
          serialize(ar);
        }
    };

    template <typename T, Dimension NDIM, typename ostream>
    ostream& operator<<(ostream& s, const FunctionsReconstructedNode<T,NDIM>& node) {
      for (size_type i = 0; i < node.count(); ++i) {
        s << "FunctionsReconstructedNode[" << i << "](" << node.key() << ", leaf " << node.is_leaf(i) << ", norm " << mra::normf(node.coeffs_view(i)) << ")";
      }
      return s;
    }

    template <typename T, Dimension NDIM, typename ostream>
    ostream& operator<<(ostream& s, const FunctionsCompressedNode<T,NDIM>& node) {
      for (size_type i = 0; i < node.count(); ++i) {
        s << "FunctionsCompressedNode[" << i << "](" << node.key() << ", norm " << mra::normf(node.coeffs_view(i)) << ")";
      }
      return s;
    }



} // namespace mra

#endif // HAVE_MRA_FUNCTIONNODE_H
