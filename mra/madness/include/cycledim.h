#ifndef CYCLEDIM_H_INCL
#define CYCLEDIM_H_INCL

#include "platform.h"
#include "types.h"
#include "tensorview.h"

namespace mra{
  namespace detail{
    template<typename T, Dimension NDIM>
    SCOPE void cycledim(const TensorView<T, NDIM>& in, TensorView<T, NDIM>& out, int nshift, int start, int end){
      std::array<int, NDIM> shifts;
      // support python-style negative indexing
      if (end < 0) {
        end = NDIM - end;
      }
      if (start < 0) {
        start = NDIM - start;
      }
      // compute new index positions
      for (int i = 0; i < NDIM; ++i) {
        if (i >= start && i < end) {
          shifts[i] = (i - start + nshift) % (end - start) + start;
        } else {
          shifts[i] = i;
        }
      }
      // assign using new index positions
      foreach_idxs(in, [&](auto... idxs){
        std::array<int, NDIM> newidxs;
        std::array<int, NDIM> idxs_arr = {idxs...};
        /* mutate the indices */
        for (int i = 0; i < NDIM; ++i) {
          newidxs[shifts[i]] = idxs_arr[i];
        }
        T val = in(idxs...);
        auto do_assign = [&]<std::size_t... Is>(std::index_sequence<Is...>){
          out(newidxs[Is]...) = val;
        };
        do_assign(std::make_index_sequence<NDIM>{});
      });
    }
  }
}

#endif // CYCLEDIM_H_INCL