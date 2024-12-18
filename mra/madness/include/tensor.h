#ifndef TTG_MRA_TENSOR_H
#define TTG_MRA_TENSOR_H

#include <algorithm>
#include <numeric>
#include <array>

#include <ttg.h>
#include <ttg/serialization.h>
#include <ttg/serialization/std/array.h>
#include <madness/world/world.h>

#include "tensorview.h"
#include "sparsity.h"

namespace mra {

  template<typename T, Dimension NDIM, template<size_type> class SparsityT = Dense, class Allocator = std::allocator<T>>
  class Tensor : public ttg::TTValue<Tensor<T, NDIM, Allocator>> {
  public:
    using value_type = std::decay_t<T>;
    using allocator_type = Allocator;
    using view_type = TensorView<value_type, NDIM, SparsityT>;
    using const_view_type = std::add_const_t<TensorView<value_type, NDIM>>;
    using buffer_type = ttg::Buffer<value_type, allocator_type>;
    using sparsity_type = SparsityT<sizeof(T)>;
    using tensor_type = Tensor<T, NDIM, SparsityT, Allocator>;

    static constexpr Dimension ndim() { return NDIM; }

    using dims_array_t = std::array<size_type, ndim()>;

    SCOPE static constexpr bool is_sparse() { return !std::is_same_v<sparsity_type, Dense>; }

    //template<typename Archive>
    //friend madness::archive::ArchiveSerializeImpl<Archive, Tensor>;

  private:
    using ttvalue_type = ttg::TTValue<Tensor<T, NDIM, Allocator>>;
    dims_array_t m_dims = {0};
    sparsity_type m_sparsity;
    buffer_type  m_buffer;
    mutable value_type  *m_sparsity_stage = nullptr;

    // (Re)allocate the tensor memory
    void realloc() {
      m_buffer.reset(size());
    }

    template<std::size_t... Is>
    static auto create_dims_array(size_type dim, std::index_sequence<Is...>) {
      return std::array{((void)Is, dim)...};
    }

    size_type buffer_size() const {
      return size() + m_sparsity.num_values();
    }

  public:
    Tensor() = default;

    /* generic */
    template<typename SparsityU = SparsityT>
    requires(std::is_same_v<SparsityU, Dense>)
    explicit Tensor(size_type dim)
    : ttvalue_type()
    , m_dims(create_dims_array(dim, std::make_index_sequence<NDIM>{}))
    , m_sparsity(dim),
    , m_buffer(buffer_size())
    { }

    template<typename... Dims, typename = std::enable_if_t<(sizeof...(Dims) > 0)>>
    Tensor(size_type d0, Dims... dims)
    : ttvalue_type()
    , m_dims({d0, static_cast<size_type>(dims)...})
    , m_sparsity(d0),
    , m_buffer(buffer_size())
    {
      static_assert(sizeof...(Dims) == NDIM,
                    "Number of arguments does not match number of Dimensions.");
    }

    Tensor(const std::array<size_type, NDIM>& dims)
    : ttvalue_type()
    , m_dims(dims)
    , m_sparsity(dims[0]),
    , m_buffer(buffer_size())
    {
      // TODO: make this static_assert (clang 14 doesn't get it)
      assert(dims.size() == NDIM);
                    //"Number of arguments does not match number of Dimensions.");
    }

    ~Tensor() {
      if constexpr (is_sparse()) {
        if (m_sparsity_stage != nullptr) {
          Allocator().deallocate(m_sparsity_stage, m_sparsity.num_values());
        }
      }
    }

    Tensor(tensor_type&& other) = default;

    Tensor& operator=(tensor_type&& other) = default;

    /* Disable copy construction.
     * There is no way we can copy data from anywhere else but the host memory space
     * so let's not even try. */
    Tensor(const tensor_type& other) = delete;

    Tensor& operator=(const tensor_type& other) = delete;

    size_type size() const {
      return std::reduce(&m_dims[0], &m_dims[ndim()], 1, std::multiplies<size_type>{});
    }

    size_type dim(Dimension dim) const {
      return m_dims[dim];
    }

    auto& buffer() {
      return m_buffer;
    }

    const auto& buffer() const {
      return m_buffer;
    }

    /* returns a view for the current memory space
     * TODO: handle const correctness (const Tensor should return a const TensorView)*/
    view_type current_view() {
      return view_type(m_buffer.current_device_ptr(), m_dims);
    }

    /* returns a view for the current memory space
     * TODO: handle const correctness (const Tensor should return a const TensorView)*/
    const view_type current_view() const {
      return view_type(m_buffer.current_device_ptr(), m_dims);
    }

    bool empty() const {
      return m_buffer.empty();
    }

    template <typename Archive>
    void serialize(Archive &ar) {
      ar &m_dims & m_sparsity &m_buffer;
    }

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int) {
      serialize(ar);
    }

    void transfer_to_device(ttg::device::Stream stream) const {
      if constexpr (is_sparse()) {
        /* TODO: fix allocation */
        m_sparsity_stage = Allocator().allocate(m_sparsity.num_values());
        sparsity_type sparsity(m_sparsity_stage, m_dims[0]);
        sparsity = m_sparsity;
        /* TODO: encapsulate this */
        cudaMemcpyAsync(m_buffer.current_device_ptr(), m_sparsity_stage, m_sparsity.num_values(), cudaMemcpyHostToDevice, stream);
      }
    }

    void transfer_to_host(ttg::device::Stream stream) const {
      if constexpr (is_sparse()) {
        /* TODO: fix allocation */
        m_sparsity_stage = Allocator().allocate(m_sparsity.num_values());
        sparsity_type sparsity(m_sparsity_stage, m_dims[0]);
        /* TODO: encapsulate this */
        cudaMemcpyAsync(m_sparsity_stage, m_buffer.current_device_ptr(), m_sparsity.num_values(), cudaMemcpyDeviceToHost, stream);
      }
    }

    void complete_transfer() const {
      if constexpr (is_sparse()) {
        Allocator().deallocate(m_sparsity_stage, m_sparsity.num_values());
        m_sparsity_stage = nullptr;
      }
    }
  };

  /**
   * Convenience type aliases for sparse and dense tensors.
   */
  template<typename T, Dimension NDIM>
  using SparseTensor = Tensor<T, NDIM, Sparsity>;
  template<typename T, Dimension NDIM>
  using DenseTensor = Tensor<T, NDIM, Dense>;

  /**
   * Output operator for tensors.
   */
  template <typename tensorT>
  requires(tensorT::is_tensor)
  std::ostream&
  operator<<(std::ostream& s, const tensorT& t) {
    if (t.size() == 0) {
      s << "[empty tensor]\n";
      return s;
    }

    const Dimension ndim = t.ndim();

    auto dims = t.dims();
    size_type maxdim = std::max_element(dims.begin(), dims.end());
    size_type index_width = std::max(std::log10(maxdim), 6.0);
    std::ios::fmtflags oldflags = s.setf(std::ios::scientific);
    long oldprec = s.precision();
    long oldwidth = s.width();

    const Dimension lastdim = ndim-1;
    const size_type lastdimsize = t.dim(lastdim);

    for (auto it=t.begin(); it!=t.end(); ) {
      const auto& index = it.index();
      s.unsetf(std::ios::scientific);
      s << '[';
      for (Dimension d=0; d<(ndim-1); d++) {
        s.width(index_width);
        s << index[d];
        s << ",";
      }
      s << " *]";
      // s.setf(std::ios::scientific);
      s.setf(std::ios::fixed);
      for (size_type i=0; i<lastdimsize; ++i,++it) { //<<< it incremented here!
        // s.precision(4);
        s << " ";
        //s.precision(8);
        //s.width(12);
        s.precision(6);
        s.width(10);
        s << *it;
      }
      s.unsetf(std::ios::scientific);
      if (it != t.end()) s << std::endl;
    }

    s.setf(oldflags,std::ios::floatfield);
    s.precision(oldprec);
    s.width(oldwidth);

    return s;
  }
} // namespace mra

#endif // TTG_MRA_TENSOR_H
