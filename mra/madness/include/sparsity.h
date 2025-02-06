#ifndef HAVE_MRA_SPARSITY_H
#define HAVE_MRA_SPARSITY_H

#include <algorithm>
#include <cstdint>

#include "types"

namespace mra {

  /**
   * Encoding of sparsity information using a byte array.
   *
   * This is a primitive form of encoding sparsity information, using one byte per entry.
   * This encoding is used on the accelerator and allows multiple thread-blocks to update
   * the state of one entry (e.g., from non-zero to allocated).
   *
   * \tparam ValueT The type of the values used by the owning container.
   *                Sparsity will ensure proper alignment for this type but will
   *                repurpose the memory to the type needed to encode sparsity.
   */
  template<typename ValueT>
  struct SparsityArray {
    using value_type = std::decay_t<ValueT>;
    using unit_type = std::byte;

  private:
    size_type  m_count = 0;
    unit_type* m_data  = nullptr;  // size_t[] encoding sparsity flags
    bool       m_owned = false;

    enum class SparsityType : std::byte {
      SPARSE    = 0,    // memory is not allocated
      ALLOCATED = 1<<0, // memory is allocated
      NONZERO    = ALLOCATED ^ 1<<1, // allocated and non-sparse
    };

    static constexpr const size_type unit_size = sizeof(unit_type);
    static constexpr const size_type one = 1;

    static constexpr size_type num_units(size_type count) {
      return count;
    }

    class sparsity_iterator {
      unit_type* m_start = nullptr;
      unit_type* m_pos   = nullptr;
      SparsityType m_type;

    public:
      sparsity_iterator(const unit_type* data, SparsityType type)
      : m_start(data)
      , m_pos(data)
      , m_type(type)
      { }

      sparsity_iterator& operator++() {
        while (!(*(++m_pos) & m_type))
        { }
        return *this;
      }

      size_type operator*() const {
        return (m_pos - m_data); // return the index
      }

      bool operator!=(const sparsity_iterator& other) const {
        return m_pos != other.m_pos;
      }
    }; // class sparsity_iterator

  public:

    constexpr SparsityArray() = default;

    /* creates an owning SparsityArray object */
    SparsityArray(size_type count)
    : m_count(count)
    , m_data(new unit_type[num_units(count)])
    , m_owned(true)
    {
      set_all_zero();
    }

    /* creates a non-owning Sparsity object */
    SCOPE SparsityArray(value_type *data, size_type count)
    : m_count(count)
    , m_data(static_cast<unit_type*>(data))
    , m_owned(false)
    { }

    SparsityArray(const SparsityArray& s)
    : m_count(s.m_count)
    , m_data(new unit_type[num_units(m_count)])
    , m_owned(true)
    {
      std::copy_n(s.m_data, m_count, m_data);
    }

    /* creates a non-owning Sparsity object */
    template<typename SparsityT>
    SparsityRange(const SparsityT& s)
    : m_count(s.size())
    {
      assert(m_count == s.size());
      apply(s);
    }

    SparsityArray(SparsityArray&& other) = default;

    SparsityArray& operator=(SparsityArray&& other) = default;

    /* copy assignment is deleted so we don't accidentally allocate memory
      * instead, create a new object instead and call apply() */
    SparsityArray& operator=(const SparsityArray& other) = delete;

    ~SparsityArray() {
      if (m_owned) {
        delete[] m_data;
        m_data = nullptr;
      }
    }

    /* returns true if value is not zero */
    SCOPE bool is_nonzero(size_type id) const {
      return m_data[id] == SparsityType::NONZERO;
    }

    /**
     * Returns true if the given id is allocated.
     */
    SCOPE bool is_allocated(size_type id) const {
      return (m_data[id] & SparsityType::ALLOCATED);
    }

    /**
     * The number of non-zero entries.
     */
    size_type count() const {
      size_type res = 0;
      for (size_type i = 0; i < num_units(m_count); ++i) {
        res += (m_data[i] == SparsityType::NONZERO) ? 1 : 0;
      }
      return res;
    }

    /**
     * The maximum number of entries total.
     */
    size_type size() const {
      return m_count;
    }

    /**
     * The offset of a given id, i.e., the sum of
     * all non-zero entries before the given id.
     */
    size_type offset(size_type id) const {
      size_type offset = 0;
      for (size_type i = 0; i < id; ++i) {
        offset += !!m_data[i];
      }
      return offset;
    }

    /**
     * Mark the given id as allocated and non-zero.
     */
    SCOPE void set_nonzero(size_type id) {
      m_data[id] = SparsityType::NONZERO;
    }

    /**
     * Mark the given id as allocated only, if it was allocated before.
     * Otherwise the id is marked as unallocated and zero.
     */
    SCOPE void set_zero(size_type id) {
      if (m_data[id] == SparsityType::NONZERO) {
        m_data[id] = SparsityType::ALLOCATED;
      }
    }

    /**
     * Mark all ids as allocated and non-zero.s
     */
    void set_all_zero() {
      std::fill(m_data, m_data+m_count, SparsityType::SPARSE);
    }

    /**
     * Mark all ids as allocated and non-zero.s
     */
    void set_all_nonzero() {
      std::fill(m_data, m_data+m_count, SparsityType::NONZERO);
    }

    /**
     * Mark all ids as allocated only.
     */
    void set_all_allocated() {
      for (size_type i = 0; i < m_count; ++i) {
        unset(i);
      }
    }

    /**
     * Mark the given id as unallocated and zero.
     */
    void remove(size_type id) {
      m_data[id] = SparsityType::SPARSE;
    }

    /* apply sparsity information from input
     * the count must be the same on both sparsity objects
     * and both sparsity objects must point to the same memory space */
    SparsityArray& operator=(const SparsityArray& s) {
      assert(m_count == s.m_count);
      apply(s);
      return *this;
    }

    void apply(const SparsityArray& s) {
      assert(m_count == s.m_count);
      std::copy_n(s.m_data, s.m_count, m_data);
    }

    template<typename SparsityT>
    void apply(const SparsityT s) {
      assert(m_count == s.size());
      for (size_type i = 0; i < m_count; ++i) {
        if (s.is_nonzero(i)) {
          set_nonzero(i);
        } else {
          remove(i);
        }
      }
    }

    /* form the union with the given SparsityArray */
    void union(const SparsityArray& s) {
      assert(m_count == s.m_count);
      for (size_type i = 0; i < m_count; ++i) {
        if (s.is_nonzero(i)) {
          set_nonzero(i);
        }
      }
    }

    /* form the union with the given sparsity */
    template<typename SparsityT>
    void union(const SparsityT& s) {
      assert(m_count == s.size());
      for (auto iter = s.begin_nonzero(); iter != s.end_nonzero(); ++iter) {
        set_nonzero(*iter);
      }
    }

    /* form the intersection with the given sparsity array */
    void intersect(const SparsityArray& s) {
      assert(m_count == s.m_count);
      for (size_type i = 0; i < m_count; ++i) {
        if (!s.is_nonzero(i)) {
          remove(i);
        }
      }
    }

    /* form the intersection with the given sparsity */
    template<typename SparsityT>
    void intersect(const SparsityT& s) {
      assert(m_count == s.size());
      size_type i = 0;
      for (auto iter = s.begin_nonzero(); iter != s.end_nonzero(); ++iter, ++i) {
        while (i < *iter) {
          remove(i++);
        }
      }
      while (i < m_count) {
        remove(i++);
      }
    }

    /* returns the number of bytes needed to track sparsity for count entries */
    static constexpr size_type num_values_required(size_type count) {
      return (count + sizeof(value_type)-1) / sizeof(value_type);
    }


    using iterator = sparsity_iterator;

    iterator begin_nonzero() {
      return iterator(m_data, SparsityType::NONZERO);
    }

    iterator end_nonzero() {
      return iterator(m_data+m_count, SparsityType::NONZERO);
    }

    iterator begin_allocated() {
      return iterator(m_data, SparsityType::ALLOCATED);
    }

    iterator begin_allocated() {
      return iterator(m_data+m_count, SparsityType::ALLOCATED);
    }
  };


  /**
   * Encoding of sparsity information using ranges.
   *
   * This encoding is more efficient than SparsityArray but is less flexible
   * with a higher cost for updates. Also does not support concurrent updates on the device.
   *
   * \tparam ValueT The type of the values used by the owning container.
   *                Sparsity will ensure proper alignment for this type but will
   *                repurpose the memory to the type needed to encode sparsity.
   */
  template<typename ValueT>
  struct SparsityRange {
    using value_type = std::decay_t<ValueT>;
    using unit_type = std::byte;

  private:

    struct Range {
      ssize_type from = -1; // inclusive
      ssize_type to   = -1; // inclusive

      Range() = default;

      Range(size_type i)
      : from(i)
      , to(i)
      { }

      void add(size_type i) {
        if (from == -1) {
          from = i;
          to   = i;
        } else {
          to = i;
        }
      }

      bool is_contiguous(size_type i) const {
        return to == i-1 || from == i+1;
      }

      bool contains(size_type i) const {
        return from <= i && i <= to;
      }

      template <typename Archive>
      void serialize(Archive &ar) {
        ar & from & to;
      }

      template <typename Archive>
      void serialize(Archive &ar, const unsigned int) {
        serialize(ar);
      }

    };

    class sparsity_iterator {
      std::vector<Range>::iterator m_iter;
      size_type m_id = 0;

    public:
      sparsity_iterator(const std::vector<Range>::iterator& ranges_iter)
      : m_iter(ranges_iter)
      {
        if (m_iter != ranges.end()) {
          m_id = m_iter->from;
        }
      }

      sparsity_iterator& operator++() {
        if (m_iter->to == m_id) {
          ++m_iter;
          if (m_iter != ranges.end()) {
            m_id = m_iter->from;
          }
        } else {
          ++m_id;
        }
        return *this;
      }

      bool operator!=(const sparsity_iterator& other) const {
        return m_iter != other.m_iter || m_id != other.m_id;
      }

      size_type operator*() const {
        return m_id;
      }
    }; // class sparsity_iterator

    size_type  m_count = 0;
    std::vector<Range> m_non_zero_ranges;   // ranges of non-zero entries
    std::vector<Range> m_allocated_ranges;  // ranges of allocated entries

    void add(size_type id, std::vector<Range>& ranges) {
      for (auto it = ranges.begin(); it != ranges.end(); ++it) {
        if (it->contains(id)) {
          return;
        }
        if (it->from > id) {
          auto next = ++it;
          auto prev = --it;
          if (next != ranges.end() && next->from == id+1) {
            next->from = id;
          } else if (it != ranges.begin() && prev->to == id-1) {
            prev->to = id;
          } else {
            ranges.insert(it, Range(id));
          }
          return;
        }
      }
      // add to the end
      ranges.push_back(Range(id));
    }

    void remove(size_type id, std::vector<Range>& ranges) {
      for (auto it = ranges.begin(); it != ranges.end(); ++it) {
        if (it->contains(id)) {
          if (it->from == it->to) {
            /* remove entire range */
            ranges.erase(it);
          } else if (it->from == id) {
            /* remove from beginning of range */
            it->from++;
          } else if (it->to == id) {
            /* remove from end of range */
            it->to--;
          } else {
            /* split the range */
            auto next = ++it;
            it->to = id-1;
            ranges.insert(next, Range(id+1, it->to));
            }
          }
          return;
        }
      }
    }

    bool contains(size_type id, const std::vector<Range>& ranges) const {
      for (const auto& r : ranges) {
        if (r.contains(id)) {
          return true;
        }
      }
      return false;
    }

  public:

    constexpr SparsityRange() = default;

    /* creates an owning SparsityArray object */
    SparsityRange(size_type count)
    : m_count(count)
    { }

    /* creates a non-owning Sparsity object */
    template<typename SparsityT>
    SparsityRange(const SparsityT& s)
    : m_count(s.size())
    {
      assert(m_count == s.size());
      apply(s);
    }

    /* copy construction allocates memory */
    SparsityRange(const SparsityRange&) = default;

    /* move construction */
    SparsityRange(SparsityRange&& other) = default;

    SparsityRange& operator=(SparsityRange&& other) = default;

    SparsityRange& operator=(const SparsityRange& other) = default;

    ~SparsityRange() = default;

    /* returns true if value is not zero */
    bool is_nonzero(size_type id) const {
      return contains(id, m_non_zero_ranges);
    }

    /**
     * Returns true if the given id is allocated.
     */
    bool is_allocated(size_type id) const {
      return contains(id, m_allocated_ranges);
    }

    /**
     * The number of non-zero entries.
     */
    size_type count_nonzero() const {
      size_type res = 0;
      for (const auto& r : m_non_zero_ranges) {
        res += r.to - r.from + 1;
      }
      return res;
    }

    /**
     * The maximum number of entries total.
     */
    size_type size() const {
      return m_count;
    }

    /**
     * The offset of a given id, i.e., the sum of
     * all non-zero entries before the given id.
     */
    size_type offset(size_type id) const {
      size_type offset = 0;
      for (const auto& r : m_allocated_ranges) {
        if (r.to < id) {
          offset += r.to - r.from + 1;
        } else {
          return offset + id - r.from;
        }
      }
      return offset;
    }

    /**
     * Mark the given id as allocated and non-zero.
     */
    void set_nonzero(size_type id) {
      add(id, m_non_zero_ranges);
      // if it's nonzero it's also allocated
      add(id, m_allocated_ranges);
    }

    void set_zero(size_type id) {
      remove(id, m_non_zero_ranges);
    }

    /**
     * Mark the given id as allocated only, if it was allocated before.
     * Otherwise the id is marked as unallocated and zero.
     */
    void set_allocated(size_type id) {
      add(id, m_allocated_ranges);
    }

    void set_all_zero() {
      m_non_zero_ranges.clear();
      m_allocated_ranges.clear();
    }

    /**
     * Mark all ids as allocated and non-zeros
     */
    void set_all_nonzero() {
      m_non_zero_ranges.clear();
      m_non_zero_ranges.push_back(Range(0, m_count));
      m_allocated_ranges.clear();
      m_allocated_ranges.push_back(Range(0, m_count));
    }

    /**
     * Mark all ids as allocated only.
     */
    void set_all_allocated() {
      m_allocated_ranges.clear();
      m_allocated_ranges.push_back(Range(0, m_count));
    }

    /**
     * Mark the given id as and zero. It will still be marked as allocated.
     */
    void set_zero(size_type id) {
      remove(id, m_non_zero_ranges);
    }

    /**
     * Remove the id from the sparsity information.
     * Marks it both zero and not allocated.
     */
    void remove(size_type id) {
      remove(id, m_non_zero_ranges);
      remove(id, m_allocated_ranges);
    }

    /* apply sparsity information from input
     * the count must be the same on both sparsity objects
     * and both sparsity objects must point to the same memory space */
    SparsityRange& operator=(const SparsityArray& s) {
      assert(m_count == s.m_count);
      apply(s);
      return *this;
    }

    void apply(const SparsityArray& s) {
      m_non_zero_ranges.clear();
      m_allocated_ranges.clear();
      Range ra  = {-1, -1}; // range for allocated entries
      Range rnz = {-1, -1}; // range for nonzero entries
      auto add_to_range = [&](size_type i, Range& r, std::vector<Range>& ranges) {
        if (r.is_contiguous(i)) {
          r.add(i);
        } else {
          if (r.from != -1) {
            ranges.push_back(r);
          }
          r = Range(i);
        }
      };
      /* iterate over all entries and form the ranges for non-zero and allocated elements */
      for (size_type i = 0; i < m_count; ++i) {
        if (s.is_nonzero(i)) {
          add_to_range(i, rnz, m_non_zero_ranges);
          add_to_range(i, ra,  m_allocated_ranges);
        } else if (s.is_allocated(i)) {
          add_to_range(i, ra,  m_allocated_ranges);
        }
      }
    }

    /* returns the number of bytes needed to track sparsity for count entries */
    static constexpr size_type num_values_required(size_type count) {
      return (count + sizeof(value_type)-1) / sizeof(value_type);
    }


    template <typename Archive>
    void serialize(Archive &ar) {
      ar & m_count & m_non_zero_ranges & m_allocated_ranges;
    }

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int) {
      serialize(ar);
    }

    using iterator = sparsity_iterator;

    iterator begin_nonzero() const {
      return iterator(m_non_zero_ranges.begin());
    }

    iterator end_nonzero() const {
      return iterator(m_non_zero_ranges.end());
    }

    iterator begin_allocated() const {
      return iterator(m_allocated_ranges.begin());
    }

    iterator end_allocated() const {
      return iterator(m_allocated_ranges.end());
    }
  };


  /**
   * Sparsity information for a fully dense tensor.
   */
  template<typename ValueT>
  struct Dense {
    using unit_type = std::byte;

  private:
    size_type  m_count = 0;

  public:

    constexpr Dense() = default;

    /* creates an owning Dense object */
    constexpr Dense(size_type count)
    : m_count(count)
    { }

    /* creates a non-owning Dense object
     * does not capture the data */
    constexpr Dense(unit_type *data, size_type count)
    : m_count(count)
    { }

    constexpr Dense& operator=(const Dense& other) = default;
    constexpr Dense& operator=(Dense&& other) = default;

    ~Dense() = default;

    /* returns true if value is not zero */
    constexpr bool is_nonzero(size_type id) const {
      return true;
    }

    /**
     * Returns true if the given id is allocated.
     */
    constexpr bool is_allocated(size_type id) const {
      return true;
    }

    /**
     * The number of non-zero entries.
     */
    constexpr size_type count() const {
      return m_count;
    }

    /**
     * The maximum number of entries total.
     */
    constexpr size_type size() const {
      return m_count;
    }

    /**
     * The offset of a given id, i.e., the sum of
     * all non-zero entries before the given id.
     */
    constexpr size_type offset(size_type id) const {
      return id;
    }

    /* apply sparsity information from input
     * the count must be the same on both sparsity objects
     * and both sparsity objects must point to the same memory space */
    constexpr Dense& operator=(const Dense& s) = default;
    constexpr Dense& operator=(Dense&& s) = default;

    /* returns the number of underlying values needed to track sparsity for count entries */
    static constexpr size_type num_values_required(size_type count) {
      return 0
    }

    template <typename Archive>
    void serialize(Archive &ar) {
      ar & m_count;
    }

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int) {
      serialize(ar);
    }

  };
} //namespace mra


#endif // HAVE_MRA_SPARSITY_H
