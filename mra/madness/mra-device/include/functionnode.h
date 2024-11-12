#ifndef HAVE_MRA_FUNCTIONNODE_H
#define HAVE_MRA_FUNCTIONNODE_H

#include "key.h"
#include "tensor.h"
#include "functions.h"

namespace mra {
    template <typename T, Dimension NDIM>
    class FunctionReconstructedNode : public ttg::TTValue<FunctionReconstructedNode<T, NDIM>> {
    public: // temporarily make everything public while we figure out what we are doing
        using key_type = Key<NDIM>;
        using tensor_type = Tensor<T,NDIM>;
        static constexpr bool is_function_node = true;

        key_type key; //< Key associated with this node to facilitate computation from otherwise unknown parent/child
        mutable T sum = 0.0; //< If recurring up tree (e.g., in compress) can use this to also compute a scalar reduction
        bool is_leaf = false; //< True if node is leaf on tree (i.e., no children).
        std::array<bool, Key<NDIM>::num_children()> is_child_leaf = { false };
        tensor_type coeffs; //< if !is_leaf these are junk (and need not be communicated)
        FunctionReconstructedNode() = default; // Default initializer does nothing so that class is POD
        FunctionReconstructedNode(const Key<NDIM>& key, std::size_t K)
        : key(key)
        , coeffs(K)
        {}
        //T normf() const {return (is_leaf ? coeffs.normf() : 0.0);}
        bool has_children() const {return !is_leaf;}

        template <typename Archive>
        void serialize(Archive& ar) {
          ar& this->key;
          ar& this->sum;
          ar& this->is_leaf;
          ar& this->is_child_leaf;
          ar& this->coeffs;
        }

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int) {
          serialize(ar);
        }
    };

    template <typename T, Dimension NDIM>
    class FunctionCompressedNode : public ttg::TTValue<FunctionCompressedNode<T, NDIM>> {
    public: // temporarily make everything public while we figure out what we are doing
        static constexpr bool is_function_node = true;

        Key<NDIM> key; //< Key associated with this node to facilitate computation from otherwise unknown parent/child
        std::array<bool, Key<NDIM>::num_children()> is_child_leaf; //< True if that child is leaf on tree
        Tensor<T,NDIM> coeffs; //< Always significant

        FunctionCompressedNode() = default; // needed for serialization
        FunctionCompressedNode(std::size_t K)
        : coeffs(2*K)
        { }
        FunctionCompressedNode(const Key<NDIM>& key, std::size_t K)
        : key(key)
        , coeffs(2*K)
        { }

        //T normf() const {return coeffs.normf();}
        bool has_children(size_t childindex) const {
            assert(childindex<Key<NDIM>::num_children());
            return !is_child_leaf[childindex];
        }

        template <typename Archive>
        void serialize(Archive& ar) {
          ar& this->key;
          ar& this->is_child_leaf;
          ar& this->coeffs;
        }

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int) {
          serialize(ar);
        }
    };

    template <typename T, Dimension NDIM, typename ostream>
    ostream& operator<<(ostream& s, const FunctionReconstructedNode<T,NDIM>& node) {
      s << "FunctionReconstructedNode(" << node.key << ", leaf " << node.is_leaf << ", norm " << mra::normf(node.coeffs.current_view()) << ")";
      return s;
    }

    template <typename T, Dimension NDIM, typename ostream>
    ostream& operator<<(ostream& s, const FunctionCompressedNode<T,NDIM>& node) {
      s << "FunctionCompressedNode(" << node.key << ", norm " << mra::normf(node.coeffs.current_view()) << ")";
      return s;
    }


    /**
     * Variants for N functions
     */

    namespace detail {
      template<Dimension NDIM, std::size_t... Is>
      std::array<std::size_t, NDIM> make_dims_helper(std::size_t N, std::size_t K, std::index_sequence<Is...>) {
        return std::array<std::size_t, NDIM>{N, ((void)Is, K)...};
      }
      /* helper to create {N, K, K, K, ...} dims array */
      template<Dimension NDIM>
      std::array<std::size_t, NDIM> make_dims(std::size_t N, std::size_t K) {
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
        FunctionsReconstructedNode(const Key<NDIM>& key, std::size_t N)
        : m_key(key)
        , m_metadata(N)
        , m_coeffs()
        { }

        FunctionsReconstructedNode(const Key<NDIM>& key, std::size_t N, std::size_t K)
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
        void allocate(std::size_t K) {
          if (!empty()) throw std::runtime_error("Reallocating non-empty FunctionNode not allowed!");
          std::size_t N = m_metadata.size();
          if (N == 0) throw std::runtime_error("Cannot reallocate FunctionNode with N = 0");
          m_coeffs = Tensor<T,NDIM+1>(detail::make_dims<NDIM+1>(N, 2*K));
        }

        bool has_children(std::size_t i) const {
          return !m_metadata[i].is_leaf;
        }

        bool any_have_children() const {
          bool result = false;
          for (std::size_t i = 0; i < m_metadata.size(); ++i) {
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

        bool& is_leaf(std::size_t i) {
          return m_metadata[i].is_leaf;
        }

        bool is_leaf(std::size_t i) const {
          return m_metadata[i].is_leaf;
        }

        bool& is_child_leaf(std::size_t i, std::size_t child) {
          return m_metadata[i].is_child_leaf[child];
        }

        bool is_leaf(std::size_t i, std::size_t child) const {
          return m_metadata[i].is_child_leaf[child];
        }

        T& sum(std::size_t i) {
          return m_metadata[i].sum;
        }

        T sum(std::size_t i) const {
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

        view_type coeffs_view(std::size_t i) {
          /* assuming all dims > 1 == K */
          const std::size_t K = m_coeffs.dim(1);
          const std::size_t K2NDIM = std::pow(K, NDIM);
          return view_type(m_coeffs.data() + (i*K2NDIM), K);
        }

        const_view_type coeffs_view(std::size_t i) const {
          /* assuming all dims > 1 == K */
          const std::size_t K = m_coeffs.dim(1);
          const std::size_t K2NDIM = std::pow(K, NDIM);
          return const_view_type(m_coeffs.data() + (i*K2NDIM), K);
        }

        key_type& key() {
          return m_key;
        }

        const key_type& key() const {
          return m_key;
        }

        std::size_t count() const {
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
        FunctionsCompressedNode(const Key<NDIM>& key, std::size_t N)
        : m_key(key)
        , m_coeffs()
        , m_is_child_leafs(N)
        { }

        FunctionsCompressedNode(const Key<NDIM>& key, std::size_t N, std::size_t K)
        : m_key(key)
        , m_coeffs(detail::make_dims<NDIM+1>(N, 2*K))
        , m_is_child_leafs(N)
        { }

        /**
         * Allocate space for coefficients using K.
         * The node must be empty before and will not be empty afterwards.
         */
        void allocate(std::size_t K) {
          if (!empty()) throw std::runtime_error("Reallocating non-empty FunctionNode not allowed!");
          std::size_t N = m_is_child_leafs.size();
          if (N == 0) throw std::runtime_error("Cannot reallocate FunctionNode with N = 0");
          m_coeffs = Tensor<T,NDIM+1>(detail::make_dims<NDIM+1>(N, 2*K));
        }

        FunctionsCompressedNode(FunctionsCompressedNode&& other) = default;
        FunctionsCompressedNode(const FunctionsCompressedNode& other) = delete;

        FunctionsCompressedNode& operator=(FunctionsCompressedNode&& other) = default;
        FunctionsCompressedNode& operator=(const FunctionsCompressedNode& other) = delete;

        bool has_children(std::size_t i, size_t childindex) const {
            assert(childindex < Key<NDIM>::num_children());
            assert(i < m_is_child_leafs.size());
            return !m_is_child_leafs[i][childindex];
        }

        std::array<bool, Key<NDIM>::num_children()>& is_child_leaf(std::size_t i) {
          return m_is_child_leafs[i];
        }

        bool is_child_leaf(std::size_t i, std::size_t child) const {
          return m_is_child_leafs[i][child];
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

        std::size_t count() const {
          return m_is_child_leafs.size();
        }

        bool empty() const {
          return m_coeffs.empty();
        }

        view_type coeffs_view(std::size_t i) {
          /* assuming all dims > 1 == K */
          const std::size_t K = m_coeffs.dim(1);
          const std::size_t K2NDIM = std::pow(K, NDIM);
          return view_type(m_coeffs.data() + (i*K2NDIM), K);
        }

        const_view_type coeffs_view(std::size_t i) const {
          /* assuming all dims > 1 == K */
          const std::size_t K = m_coeffs.dim(1);
          const std::size_t K2NDIM = std::pow(K, NDIM);
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
      for (std::size_t i = 0; i < node.count(); ++i) {
        s << "FunctionsReconstructedNode[" << i << "](" << node.key() << ", leaf " << node.is_leaf(i) << ", norm " << mra::normf(node.coeffs_view(i)) << ")";
      }
      return s;
    }

    template <typename T, Dimension NDIM, typename ostream>
    ostream& operator<<(ostream& s, const FunctionsCompressedNode<T,NDIM>& node) {
      for (std::size_t i = 0; i < node.count(); ++i) {
        s << "FunctionsCompressedNode[" << i << "](" << node.key() << ", norm " << mra::normf(node.coeffs_view(i)) << ")";
      }
      return s;
    }



} // namespace mra

#endif // HAVE_MRA_FUNCTIONNODE_H
