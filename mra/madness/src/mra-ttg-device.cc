#include <ttg.h>
#include "tensor.h"
#include "tensorview.h"
#include "functionnode.h"
#include "functiondata.h"
#include "kernels.h"
#include "gaussian.h"
#include "functionfunctor.h"
#include "key.h"
#include "domain.h"

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

using namespace mra; // we're lazy

#ifdef MRA_ENABLE_HOST
#define TASKTYPE void
constexpr const ttg::ExecutionSpace Space = ttg::ExecutionSpace::Host;
#else
#define TASKTYPE ttg::device::Task
constexpr const ttg::ExecutionSpace Space = ttg::ExecutionSpace::CUDA;
#endif


template <mra::Dimension NDIM>
auto make_start(const ttg::Edge<mra::Key<NDIM>, void>& ctl) {
    auto func = [](const mra::Key<NDIM>& key) { ttg::sendk<0>(key); };
    return ttg::make_tt<mra::Key<NDIM>>(func, ttg::edges(), edges(ctl), "start", {}, {"control"});
}

template<typename FnT, typename T, mra::Dimension NDIM>
auto make_project(
  const ttg::Buffer<mra::Domain<NDIM>>& db,
  const ttg::Buffer<FnT>& fb,
  std::size_t N,
  std::size_t K,
  const mra::FunctionData<T, NDIM>& functiondata,
  const T thresh, /// should be scalar value not complex
  ttg::Edge<mra::Key<NDIM>, void> control,
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> result)
{
  /* create a non-owning buffer for domain and capture it */
  auto fn = [&, N, K, thresh, gl = mra::GLbuffer<T>()]
            (const mra::Key<NDIM>& key) -> TASKTYPE {
    using tensor_type = typename mra::Tensor<T, NDIM+1>;
    using key_type = typename mra::Key<NDIM>;
    using node_type = typename mra::FunctionsReconstructedNode<T, NDIM>;
    node_type result(key, N); // empty for fast-paths, no need to zero out
    auto outputs = ttg::device::forward();
    auto* fn_arr = fb.host_ptr();
    bool all_initial_level = true;
    for (std::size_t i = 0; i < N; ++i) {
      if (key.level() >= initial_level(fn_arr[i])) {
        all_initial_level = false;
        break;
      }
    }
    //std::cout << "project " << key << " all initial " << all_initial_level << std::endl;
    if (all_initial_level) {
      std::vector<mra::Key<NDIM>> bcast_keys;
      /* TODO: children() returns an iteratable object but broadcast() expects a contiguous memory range.
                We need to fix broadcast to support any ranges */
      for (auto child : children(key)) bcast_keys.push_back(child);

#ifndef MRA_ENABLE_HOST
      outputs.push_back(ttg::device::broadcastk<0>(std::move(bcast_keys)));
#else
      ttg::broadcastk<0>(std::move(bcast_keys));
#endif
      result.set_all_leaf(false);
    } else {
      bool all_negligible = true;
      for (std::size_t i = 0; i < N; ++i) {
        all_negligible &= mra::is_negligible<FnT,T,NDIM>(
                                    fn_arr[i], db.host_ptr()->template bounding_box<T>(key),
                                    mra::truncate_tol(key,thresh));
      }
      //std::cout << "project " << key << " all negligible " << all_negligible << std::endl;
      if (all_negligible) {
        result.set_all_leaf(true);
      } else {
        /* here we actually compute: first select a device */
        //result.is_leaf = fcoeffs(f, functiondata, key, thresh, coeffs);
        /**
         * BEGIN FCOEFFS HERE
         * TODO: figure out a way to outline this into a function or coroutine
         */
        // allocate tensor
        result = node_type(key, N, K);
        tensor_type& coeffs = result.coeffs();

        /* global function data */
        // TODO: need to make our own FunctionData with dynamic K
        const auto& phibar = functiondata.get_phibar();
        const auto& hgT = functiondata.get_hgT();

        /* temporaries */
        /* TODO: have make_scratch allocate pinned memory for us */
        auto is_leafs = std::make_unique_for_overwrite<bool[]>(N);
        auto is_leafs_scratch = ttg::make_scratch(is_leafs.get(), ttg::scope::Allocate, N);
        const std::size_t tmp_size = fcoeffs_tmp_size<NDIM>(K)*N;
        auto tmp = std::make_unique_for_overwrite<T[]>(tmp_size);
        auto tmp_scratch = ttg::make_scratch(tmp.get(), ttg::scope::Allocate, tmp_size);

        /* coeffs don't have to be synchronized into the device */
        coeffs.buffer().reset_scope(ttg::scope::Allocate);

        /* TODO: cannot do this from a function, had to move it into the main task */
  #ifndef MRA_ENABLE_HOST
        co_await ttg::device::select(db, gl, fb, coeffs.buffer(), phibar.buffer(),
                                    hgT.buffer(), tmp_scratch, is_leafs_scratch);
  #endif
        auto coeffs_view = coeffs.current_view();
        auto phibar_view = phibar.current_view();
        auto hgT_view    = hgT.current_view();
        T* tmp_device = tmp_scratch.device_ptr();
        bool *is_leafs_device = is_leafs_scratch.device_ptr();
        auto *f_ptr   = fb.current_device_ptr();
        auto& domain = *db.current_device_ptr();
        auto  gldata = gl.current_device_ptr();

        /* submit the kernel */
        submit_fcoeffs_kernel(domain, gldata, f_ptr, key, N, K, coeffs_view,
                              phibar_view, hgT_view, tmp_device,
                              is_leafs_device, thresh, ttg::device::current_stream());

        /* wait and get is_leaf back */
  #ifndef MRA_ENABLE_HOST
        co_await ttg::device::wait(is_leafs_scratch);
  #endif
        for (std::size_t i = 0; i < N; ++i) {
          result.is_leaf(i) = is_leafs[i];
        }
        /**
         * END FCOEFFS HERE
         */
      }

      //std::cout << "project " << key << " all leaf " << result.is_all_leaf() << std::endl;
      if (!result.is_all_leaf()) {
        std::vector<mra::Key<NDIM>> bcast_keys;
        for (auto child : children(key)) bcast_keys.push_back(child);
#ifndef MRA_ENABLE_HOST
        outputs.push_back(ttg::device::broadcastk<0>(std::move(bcast_keys)));
#else
        ttg::broadcastk<0>(bcast_keys);
#endif
      }

    }
#ifndef MRA_ENABLE_HOST
    outputs.push_back(ttg::device::send<1>(key, std::move(result))); // always produce a result
    co_await std::move(outputs);
#else
    ttg::send<1>(key, std::move(result));
#endif
  };

  ttg::Edge<mra::Key<NDIM>, void> refine("refine");
  return ttg::make_tt<Space>(std::move(fn), ttg::edges(fuse(control, refine)), ttg::edges(refine,result), "project");
}

template<mra::Dimension NDIM, typename Value, std::size_t I, std::size_t... Is>
static auto select_compress_send(const mra::Key<NDIM>& key, Value&& value,
                                 std::size_t child_idx,
                                 std::index_sequence<I, Is...>) {
  if (child_idx == I) {
#ifndef MRA_ENABLE_HOST
    return ttg::device::send<I>(key.parent(), std::forward<Value>(value));
#else
    return ttg::send<I>(key.parent(), std::forward<Value>(value));
#endif
  } else if constexpr (sizeof...(Is) > 0){
    return select_compress_send(key, std::forward<Value>(value), child_idx, std::index_sequence<Is...>{});
  }
  /* if we get here we messed up */
  throw std::runtime_error("Mismatching number of children!");
}

/* forward a reconstructed function node to the right input of do_compress
 * this is a device task to prevent data from being pulled back to the host
 * even though it will not actually perform any computation */
template<typename T, mra::Dimension NDIM>
static TASKTYPE do_send_leafs_up(const mra::Key<NDIM>& key, const mra::FunctionsReconstructedNode<T, NDIM>& node) {
  /* drop all inputs from nodes that are not leafs, they will be upstreamed by compress */
  if (!node.any_have_children()) {
#ifndef MRA_ENABLE_HOST
    co_await select_compress_send(key, node, key.childindex(), std::make_index_sequence<mra::Key<NDIM>::num_children()>{});
#else
    select_compress_send(key, node, key.childindex(), std::make_index_sequence<mra::Key<NDIM>::num_children()>{});
#endif
  }
}


/// Make a composite operator that implements compression for a single function
template <typename T, mra::Dimension NDIM>
static auto make_compress(
  const std::size_t N,
  const std::size_t K,
  const mra::FunctionData<T, NDIM>& functiondata,
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>>& in,
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>>& out)
{
  static_assert(NDIM == 3); // TODO: worth fixing?

  constexpr const std::size_t num_children = mra::Key<NDIM>::num_children();
  // creates the right number of edges for nodes to flow from send_leafs_up to compress
  // send_leafs_up will select the right input for compress
  auto create_edges = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    return ttg::edges(((void)Is, ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>>{})...);
  };
  auto send_to_compress_edges = create_edges(std::make_index_sequence<num_children>{});
  /* append out edge to set of edges */
  auto compress_out_edges = std::tuple_cat(send_to_compress_edges, std::make_tuple(out));
  /* use the tuple variant to handle variable number of inputs while suppressing the output tuple */
  auto do_compress = [&, N, K](const mra::Key<NDIM>& key,
                         //const std::tuple<const FunctionsReconstructedNodeTypes&...>& input_frns
                         const mra::FunctionsReconstructedNode<T,NDIM> &in0,
                         const mra::FunctionsReconstructedNode<T,NDIM> &in1,
                         const mra::FunctionsReconstructedNode<T,NDIM> &in2,
                         const mra::FunctionsReconstructedNode<T,NDIM> &in3,
                         const mra::FunctionsReconstructedNode<T,NDIM> &in4,
                         const mra::FunctionsReconstructedNode<T,NDIM> &in5,
                         const mra::FunctionsReconstructedNode<T,NDIM> &in6,
                         const mra::FunctionsReconstructedNode<T,NDIM> &in7) -> TASKTYPE {
    //const typename ::detail::tree_types<T,K,NDIM>::compress_in_type& in,
    //typename ::detail::tree_types<T,K,NDIM>::compress_out_type& out) {
      constexpr const auto num_children = mra::Key<NDIM>::num_children();
      constexpr const auto out_terminal_id = num_children;
      mra::FunctionsCompressedNode<T,NDIM> result(key, N); // The eventual result
      // create empty, may be reset if needed
      mra::FunctionsReconstructedNode<T, NDIM> p(key, N);

      auto set_child_info = [&](mra::FunctionsCompressedNode<T, NDIM>& result){
        for (std::size_t i = 0; i < N; ++i) {  // Collect child leaf info
          result.is_child_leaf(i) = std::array{in0.is_leaf(i), in1.is_leaf(i), in2.is_leaf(i),
                                               in3.is_leaf(i), in4.is_leaf(i), in5.is_leaf(i),
                                               in6.is_leaf(i), in7.is_leaf(i)};
        }
      };

      /* check if all inputs are empty */
      bool all_empty = in0.empty() && in1.empty() && in2.empty() && in3.empty() &&
                       in4.empty() && in5.empty() && in6.empty() && in7.empty();

      if (all_empty) {
        set_child_info(result);
        /* all data is still on the host so the coefficients are zero */
        for (std::size_t i = 0; i < N; ++i) {
          p.sum(i) = 0.0;
        }
        //std::cout << "compress " << key << " all empty " << std::endl;
      } else {

        /* some inputs are on the device so submit a kernel */

        // allocate the result
        result = mra::FunctionsCompressedNode<T, NDIM>(key, N, K);
        auto& d = result.coeffs();
        set_child_info(result);
        p = mra::FunctionsReconstructedNode<T, NDIM>(key, N, K);

        /* d and p don't have to be synchronized into the device */
        d.buffer().reset_scope(ttg::scope::Allocate);
        p.coeffs().buffer().reset_scope(ttg::scope::Allocate);

        /* stores sumsq for each child and for result at the end of the kernel */
        const std::size_t tmp_size = compress_tmp_size<NDIM>(K)*N;
        auto tmp = std::make_unique_for_overwrite<T[]>(tmp_size);
        const auto& hgT = functiondata.get_hgT();
        auto tmp_scratch = ttg::make_scratch(tmp.get(), ttg::scope::Allocate, tmp_size);
        auto d_sumsq = std::make_unique_for_overwrite<T[]>(N);
        auto d_sumsq_scratch = ttg::make_scratch(d_sumsq.get(), ttg::scope::Allocate, N);
  #ifndef MRA_ENABLE_HOST
        auto input = ttg::device::Input(p.coeffs().buffer(), d.buffer(), hgT.buffer(),
                                        tmp_scratch, d_sumsq_scratch);
        auto select_in = [&](const auto& in) {
          if (!in.empty()) {
            input.add(in.coeffs().buffer());
          }
        };
        select_in(in0); select_in(in1);
        select_in(in2); select_in(in3);
        select_in(in4); select_in(in5);
        select_in(in6); select_in(in7);

        co_await ttg::device::select(input);
  #endif

        /* some constness checks for the API */
        static_assert(std::is_const_v<std::remove_reference_t<decltype(in0)>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(in0.coeffs())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(in0.coeffs().buffer())>>);
        static_assert(std::is_const_v<std::remove_reference_t<std::remove_reference_t<decltype(*in0.coeffs().buffer().current_device_ptr())>>>);

        /* assemble input array and submit kernel */
        //auto input_ptrs = std::apply([](auto... ins){ return std::array{(ins.coeffs.buffer().current_device_ptr())...}; });
        auto get_ptr = [](const auto& in) {
          return in.empty() ? nullptr : in.coeffs().buffer().current_device_ptr();
        };
        auto input_ptrs = std::array{get_ptr(in0), get_ptr(in1), get_ptr(in2), get_ptr(in3),
                                     get_ptr(in4), get_ptr(in5), get_ptr(in6), get_ptr(in7)};

        auto coeffs_view = p.coeffs().current_view();
        auto rcoeffs_view = d.current_view();
        auto hgT_view = hgT.current_view();

        submit_compress_kernel(key, N, K, coeffs_view, rcoeffs_view, hgT_view,
                              tmp_scratch.device_ptr(), d_sumsq_scratch.device_ptr(), input_ptrs,
                              ttg::device::current_stream());

        /* wait for kernel and transfer sums back */
  #ifndef MRA_ENABLE_HOST
        co_await ttg::device::wait(d_sumsq_scratch);
  #endif

        for (std::size_t i = 0; i < N; ++i) {
          auto sumsqs = std::array{in0.sum(i), in1.sum(i), in2.sum(i), in3.sum(i),
                                   in4.sum(i), in5.sum(i), in6.sum(i), in7.sum(i)};
          auto child_sumsq = std::reduce(sumsqs.begin(), sumsqs.end());
          p.sum(i) = d_sumsq[i] + child_sumsq; // result sumsq is last element in sumsqs
          //std::cout << "compress " << key << " fn " << i << "/" << N << " d_sumsq " << d_sumsq[i]
          //          << " child_sumsq " << child_sumsq << " sum " << p.sum(i) << std::endl;
        }

      }

      // Recur up
      if (key.level() > 0) {
        // will not return
#ifndef MRA_ENABLE_HOST
        co_await ttg::device::forward(
          // select to which child of our parent we send
          //ttg::device::send<0>(key, std::move(p)),
          select_compress_send(key, std::move(p), key.childindex(), std::make_index_sequence<num_children>{}),
          // Send result to output tree
          ttg::device::send<out_terminal_id>(key, std::move(result)));
#else
          select_compress_send(key, std::move(p), key.childindex(), std::make_index_sequence<num_children>{});
          ttg::send<out_terminal_id>(key, std::move(result));
#endif
      } else {
        for (std::size_t i = 0; i < N; ++i) {
          std::cout << "At root of compressed tree fn " << i << ": total normsq is " << p.sum(i) << std::endl;
        }
#ifndef MRA_ENABLE_HOST
        co_await ttg::device::forward(
          // Send result to output tree
          ttg::device::send<out_terminal_id>(key, std::move(result)));
#else
        ttg::send<out_terminal_id>(key, std::move(result));
#endif
      }
  };
  return std::make_tuple(ttg::make_tt<Space>(&do_send_leafs_up<T,NDIM>, edges(in), send_to_compress_edges, "send_leaves_up"),
                         ttg::make_tt<Space>(std::move(do_compress), send_to_compress_edges, compress_out_edges, "do_compress"));
}

template <typename T, mra::Dimension NDIM>
auto make_reconstruct(
  const std::size_t N,
  const std::size_t K,
  const mra::FunctionData<T, NDIM>& functiondata,
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> in,
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> out,
  const std::string& name = "reconstruct")
{
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T,NDIM>> S("S");  // passes scaling functions down

  auto do_reconstruct = [&, N, K](const mra::Key<NDIM>& key,
                                  mra::FunctionsCompressedNode<T, NDIM>&& node,
                                  const mra::FunctionsReconstructedNode<T, NDIM>& from_parent) -> TASKTYPE {
    const std::size_t tmp_size = reconstruct_tmp_size<NDIM>(K)*N;
    auto tmp = std::make_unique_for_overwrite<T[]>(tmp_size);
    const auto& hg = functiondata.get_hg();
    auto tmp_scratch = ttg::make_scratch(tmp.get(), ttg::scope::Allocate, tmp_size);
    mra::KeyChildren<NDIM> children(key);

    bool node_empty = node.empty();

    // Send empty interior node to result tree
    auto r_empty = mra::FunctionsReconstructedNode<T,NDIM>(key, N);
    r_empty.set_all_leaf(false);
#ifndef MRA_ENABLE_HOST
    // forward() returns a vector that we can push into
    auto sends = ttg::device::forward(ttg::device::send<1>(key, std::move(r_empty)));
    auto do_send = [&]<std::size_t I, typename S>(auto& child, S&& node) {
          sends.push_back(ttg::device::send<I>(child, std::forward<S>(node)));
    };
#else
    ttg::send<1>(key, std::move(r_empty));
    auto do_send = []<std::size_t I, typename S>(auto& child, S&& node) {
      ttg::send<I>(child, std::forward<S>(node));
    };
#endif // MRA_ENABLE_HOST

    // array of child nodes
    std::array<mra::FunctionsReconstructedNode<T,NDIM>, mra::Key<NDIM>::num_children()> r_arr;
    for (auto it=children.begin(); it!=children.end(); ++it) {
      const mra::Key<NDIM> child= *it;
      auto& r = r_arr[it.index()];
      r = mra::FunctionsReconstructedNode<T,NDIM>(key, N);
      for (std::size_t i = 0; i < N; ++i) {
        r.is_leaf(i) = node.is_child_leaf(i, it.index());
      }
    }

    if (node.empty() && from_parent.empty()) {
      //std::cout << "reconstruct " << key << " node and parent empty " << std::endl;
      /* both the node and the parent are empty so we can shortcut with empty results */
      for (auto it=children.begin(); it!=children.end(); ++it) {
        const mra::Key<NDIM> child= *it;
        auto& r = r_arr[it.index()];
        if (r.is_all_leaf()) {
          do_send.template operator()<1>(child, std::move(r));
        } else {
          do_send.template operator()<0>(child, std::move(r));
        }
      }
#ifndef MRA_ENABLE_HOST
      // won't return
      co_await std::move(sends);
      assert(0);
#else  // MRA_ENABLE_HOST
      return; // we're done
#endif // MRA_ENABLE_HOST
    } else if (node.empty()) {
      /* parent node not empty so allocate a new compressed node */
      //std::cout << "reconstruct " << key << " allocating previously empoty node " << std::endl;
      node.allocate(K);
      node.coeffs().buffer().reset_scope(ttg::scope::Allocate);
    }

    /* once we are here we know we need to invoke the reconstruct kernel */

    /* populate the vector of r's
     * TODO: TTG/PaRSEC supports only a limited number of inputs so for higher dimensions
     *       we may have to consolidate the r's into a single buffer and pick them apart afterwards.
     *       That will require the ability to ref-count 'parent buffers'. */
    for (int i = 0; i < key.num_children(); ++i) {
      r_arr[i].allocate(K);
      // no need to send this data to the device
      r_arr[i].coeffs().buffer().reset_scope(ttg::scope::Allocate);
    }

#ifndef MRA_ENABLE_HOST
    // helper lambda to pick apart the std::array
    auto make_inputs = [&]<std::size_t... Is>(std::index_sequence<Is...>){
      return ttg::device::Input(hg.buffer(), node.coeffs().buffer(), tmp_scratch,
                                (r_arr[Is].coeffs().buffer())...);
    };
    auto inputs = make_inputs(std::make_index_sequence<mra::Key<NDIM>::num_children()>{});
    if (!from_parent.empty()) {
      inputs.add(from_parent.coeffs().buffer());
    }
    /* select a device */
    co_await ttg::device::select(inputs);
#endif

    // helper lambda to pick apart the std::array
    auto assemble_tensor_ptrs = [&]<std::size_t... Is>(std::index_sequence<Is...>){
      return std::array{(r_arr[Is].coeffs().current_view().data())...};
    };
    auto r_ptrs = assemble_tensor_ptrs(std::make_index_sequence<mra::Key<NDIM>::num_children()>{});
    auto node_view = node.coeffs().current_view();
    auto hg_view = hg.current_view();
    auto from_parent_view = from_parent.coeffs().current_view();
    submit_reconstruct_kernel(key, N, K, node_view, node_empty, hg_view, from_parent_view,
                              r_ptrs, tmp_scratch.device_ptr(), ttg::device::current_stream());

    for (auto it=children.begin(); it!=children.end(); ++it) {
      const mra::Key<NDIM> child= *it;
      mra::FunctionsReconstructedNode<T,NDIM>& r = r_arr[it.index()];
      r.key() = child;
      if (r.is_all_leaf()) {
        do_send.template operator()<1>(child, std::move(r));
      } else {
        do_send.template operator()<0>(child, std::move(r));
      }
    }
#ifndef MRA_ENABLE_HOST
    co_await std::move(sends);
#endif // MRA_ENABLE_HOST
  };


  auto s = ttg::make_tt<Space>(std::move(do_reconstruct), ttg::edges(in, S), ttg::edges(S, out), name, {"input", "s"}, {"s", "output"});

  if (ttg::default_execution_context().rank() == 0) {
    s->template in<1>()->send(mra::Key<NDIM>{0,{0}},
                              mra::FunctionsReconstructedNode<T,NDIM>(mra::Key<NDIM>{0,{0}}, N)); // Prime the flow of scaling functions
  }

  return s;
}


static std::mutex printer_guard;
template <typename keyT, typename valueT>
auto make_printer(const ttg::Edge<keyT, valueT>& in, const char* str = "", const bool doprint=true) {
  auto func = [str,doprint](const keyT& key, const valueT& value) {
    // sanity check
    assert(value.coeffs().buffer().is_current_on(ttg::device::Device()));
    if (doprint) {
      std::lock_guard<std::mutex> obolus(printer_guard);
      std::cout << str << " (" << key << "," << value << ")" << std::endl;
    }
  };
  return ttg::make_tt(func, ttg::edges(in), ttg::edges(), "printer", {"input"});
}

template<typename T, mra::Dimension NDIM>
auto make_gaxpy(ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> in1,
              ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> in2,
              ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> out,
              const T scalarA, const T scalarB, const size_t N, const size_t K)
{
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> S1, S2; // to balance trees

  auto func = [N, K, scalarA, scalarB](
            const mra::Key<NDIM>& key,
            const mra::FunctionsCompressedNode<T, NDIM>& t1,
            const mra::FunctionsCompressedNode<T, NDIM>& t2) -> TASKTYPE {

    auto sends = ttg::device::forward();
    auto send_out = [&]<typename S>(S&& out){
#ifndef MRA_ENABLE_HOST
      sends.push_back(ttg::device::send<0>(key, std::forward<S>(out)));
#else
      ttg::send<0>(key, std::forward<S>(out));
#endif
    };

    if (t1.empty() && t2.empty()) {
      /* send out an empty result */
      auto out = mra::FunctionsCompressedNode<T, NDIM>(key, N); // out -> result
      send_out(std::move(out));
    } else {

      auto out = mra::FunctionsCompressedNode<T, NDIM>(key, N, K);

  #ifndef MRA_ENABLE_HOST
      auto input = ttg::device::Input(out.coeffs().buffer());
      if (!t1.empty()) {
        input.add(t1.coeffs().buffer());
      }
      if (!t2.empty()) {
        input.add(t2.coeffs().buffer());
      }
      co_await ttg::device::select(input);
  #endif
      auto t1_view = t1.coeffs().current_view();
      auto t2_view = t2.coeffs().current_view();
      auto out_view = out.coeffs().current_view();

      submit_gaxpy_kernel(key, t1_view, t2_view, out_view,
                          scalarA, scalarB, N, K, ttg::device::current_stream());

      send_out(std::move(out));
    }


//     /* balance trees if needed by sending empty nodes to missing inputs */
//     auto balance_trees = [&]<std::size_t I>(){
//       std::vector<mra::Key<NDIM>> child_keys;
//       for (auto child : children(key)) {
//         child_keys.push_back(child);
//       }
//       // TODO: do we care about the key here? if so we have to send instead
//       auto t = mra::FunctionsReconstructedNode<T, NDIM>(key, N);
//       // mark all functions as leafs
//       t.set_all_leaf(true);
// #ifndef MRA_ENABLE_HOST
//       sends.push_back(ttg::device::broadcast<I>(
//                         std::move(child_keys), std::move(t)));
// #else
//       ttg::broadcast<I>(std::move(child_keys), std::move(t));
// #endif // MRA_ENABLE_HOST
//     };

//     if (t1.is_all_leaf() && !t2.is_all_leaf()) {
//       /* broadcast an empty node for t1 to all children */
//       balance_trees.template operator()<1>();
//     } else if (!t1.is_all_leaf() && t2.is_all_leaf()) {
//       /* broadcast an empty node for t2 to all children */
//       balance_trees.template operator()<2>();
//     }

// #ifndef MRA_ENABLE_HOST
//     co_await std::move(sends);
// #endif // MRA_ENABLE_HOST
  };

  return ttg::make_tt<Space>(std::move(func),
                             ttg::edges(ttg::fuse(S1, in1), ttg::fuse(S2, in2)),
                             ttg::edges(out, S1, S2), "gaxpy",
                             {"in1", "in2"},
                             {"out", "S1", "S2"});
}

template<typename T, mra::Dimension NDIM>
auto make_multiply(ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> in1,
              ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> in2,
              ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> out,
              const mra::FunctionData<T, NDIM>& functiondata,
              const ttg::Buffer<mra::Domain<NDIM>>& db, const size_t N, const size_t K)
{
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> S1, S2; // to balance trees

  auto func = [&, N, K](
            const mra::Key<NDIM>& key,
            const mra::FunctionsReconstructedNode<T, NDIM>& t1,
            const mra::FunctionsReconstructedNode<T, NDIM>& t2) -> TASKTYPE {

    auto sends = ttg::device::forward();
    auto send_out = [&]<typename S>(S&& out){
#ifndef MRA_ENABLE_HOST
      sends.push_back(ttg::device::send<0>(key, std::forward<S>(out)));
#else
      ttg::send<0>(key, std::forward<S>(out));
#endif
    };

    if (t1.empty() || t2.empty()) {
      /* send out an empty result */
      auto out = mra::FunctionsReconstructedNode<T, NDIM>(key, N);
      send_out(std::move(out));
    } else {
      auto out = mra::FunctionsReconstructedNode<T, NDIM>(key, N, K);
      const auto& phibar = functiondata.get_phibar();
      const auto& phiT = functiondata.get_phiT();
      const std::size_t tmp_size = multiply_tmp_size<NDIM>(K)*N;
      auto tmp = std::make_unique_for_overwrite<T[]>(tmp_size);
      auto tmp_scratch = ttg::make_scratch(tmp.get(), ttg::scope::Allocate, tmp_size);

  #ifndef MRA_ENABLE_HOST
      auto input = ttg::device::Input(out.coeffs().buffer(), phibar.buffer(), phiT.buffer(),
                                      tmp_scratch);
      // if (!t1.empty()) {
      //   input.add(t1.coeffs().buffer());
      // }
      // if (!t2.empty()) {
      //   input.add(t2.coeffs().buffer());
      // }
      // pass in phibar, phiT, gl, and tmp_scratch to select
      co_await ttg::device::select(input);
  #endif
      auto t1_view = t1.coeffs().current_view();
      auto t2_view = t2.coeffs().current_view();
      auto out_view = out.coeffs().current_view();

      auto phiT_view = phiT.current_view();
      auto phibar_view = phibar.current_view();
      auto& D = *db.current_device_ptr();
      T* tmp_device = tmp_scratch.device_ptr();

      submit_multiply_kernel(D, t1_view, t2_view, out_view, phiT_view, phibar_view,
                          N, K, key, tmp_device, ttg::device::current_stream());

      send_out(std::move(out));
    }

#ifndef MRA_ENABLE_HOST
    co_await std::move(sends);
#endif // MRA_ENABLE_HOST
  };

  return ttg::make_tt<Space>(std::move(func),
                             ttg::edges(ttg::fuse(S1, in1), ttg::fuse(S2, in2)),
                             ttg::edges(out, S1, S2), "multiply",
                             {"in1", "in2"},
                             {"out", "S1", "S2"});
}

template <typename T, Dimension NDIM>
auto make_norm(size_type N, size_type K,
               ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> input,
               ttg::Edge<mra::Key<NDIM>, T> result){
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> leaf_e, inner_e; // distribute to either leaf or inner node task
  ttg::Edge<mra::Key<NDIM>, T> norm_e; // norm edge

  /**
   * Leaf-node tasks take no norm from children
   */
  auto leaf_fn = [N, K](const mra::Key<NDIM>& key,
                        const mra::FunctionsCompressedNode<T, NDIM>& in) -> TASKTYPE {
    T norm;
    auto norm_scratch = ttg::make_scratch(&norm, ttg::scope::Allocate);

#ifndef MRA_ENABLE_HOST
    co_await ttg::device::select(norm_scratch, in.coeffs().buffer());
    auto node_view = in.coeffs().current_view();
    auto norm_ptr = norm_scratch.device_ptr();
    submit_norm_kernel(key, N, K, node_view, norm_ptr, ttg::device::current_stream());
    // wait for the norm to come back
    co_await ttg::device::wait(norm_scratch);
    // send norm upstream
    co_await ttg::device::send<0>(key.parent(), std::forward<T>(norm));
#else
    auto node_view = in.coeffs().current_view();
    auto norm_ptr = norm_scratch.device_ptr();
    submit_norm_kernel(key, N, K, node_view, norm_ptr, ttg::device::current_stream());
    // send upstream
    ttg::send<0>(key.parent(), std::forward<T>(norm));
#endif // MRA_ENABLE_HOST
  };

  auto leaf_tt = ttg::make_tt<Space>(std::move(leaf_fn),
                                     ttg::edges(leaf_e),         // leaf node input
                                     ttg::edges(norm_e),         // norm output
                                     "norm-leaf"),

  /**
   * Inner node tasks take norms from children.
   */
  auto inner_fn = [N, K](const mra::Key<NDIM>& key,
                         const mra::FunctionsCompressedNode<T, NDIM>& in,
                         const T& child_norm_sum) -> TASKTYPE {
    T norm;

#ifndef MRA_ENABLE_HOST
    auto norm_scratch = ttg::device::make_scratch(&norm, ttg::scope::Allocate);
    co_await ttg::device::select(norm_scratch, in.coeffs().buffer());
    auto node_view = in.coeffs().current_view();
    auto norm_ptr = norm_scratch.device_ptr();
    submit_norm_kernel(key, N, K, node_view, norm_ptr, ttg::device::current_stream());
    // wait for the norm to come back
    co_await ttg::device::wait(norm_scratch);
    T result_norm = norm + child_norm_sum;

    if (key.level() == 0) {
      // send to output
      co_await ttg::device::send<1>(key.parent(), std::forward<T>(result_norm));
    } else {
      // send upstream
      co_await ttg::device::send<0>(key.parent(), std::forward<T>(result_norm));
    }
#else  // MRA_ENABLE_HOST
    auto node_view = in.coeffs().current_view();
    submit_norm_kernel(key, N, K, node_view, &norm, ttg::device::current_stream());
    T result_norm = norm + child_norm_sum;
    if (key.level() == 0) {
      // send to output
      ttg::send<1>(key.parent(), std::forward<T>(result_norm));
    } else {
      // send upstream
      ttg::send<0>(key.parent(), std::forward<T>(result_norm));
    }
#endif // MRA_ENABLE_HOST
  };

  auto inner_tt = ttg::make_tt<Space>(std::move(inner_fn),
                                      ttg::edges(inner_e, norm_e),      // inner node input
                                      ttg::edges(norm_e, result), // norm and result output
                                      "norm-inner"),

  // reducer to form the sum of all children before the sum is passed into the inner_fn
  inner_tt->set_input_reducer<1>([](T& a, const T& b){
    a += b;
  }, mra::Key<NDIM>::num_children());

  /**
   * Task to select whether a node is leaf or inner and forward accordingly
   */
  auto select_fn = [](const mra::Key<NDIM>& key,
                      const mra::FunctionsCompressedNode<T, NDIM>& in) -> TASKTYPE {
    if (in.is_all_child_leaf()) {
      // send to leaf function
      co_await ttg::device::send<0>(key, in);
    } else {
      // send to inner function
      co_await ttg::device::send<1>(key, in);
    }
  };

  auto select_tt = ttg::make_tt<Space>(std::move(select_fn),
                                        ttg::edges(input),     // main input
                                        ttg::edges(leaf_e, inner_e),      // leaf and inner output
                                        "norm-select");

  /* compile everything into tasks */
  return std::make_tuple(std::move(leaf_tt),
                         std::move(inner_tt),
                         std::move(select_tt));
}

// computes from bottom up
// task2: receive norm from children, compute on self, send send up

/**
 * Test MRA projection with K coefficients in each of the NDIM dimension on
 * N random Gaussian functions.
 */
template<typename T, mra::Dimension NDIM>
void test(std::size_t N, std::size_t K) {
  auto functiondata = mra::FunctionData<T,NDIM>(K);
  mra::Domain<NDIM> D;
  D.set_cube(-6.0,6.0);

  srand48(5551212); // for reproducible results
  for (int i = 0; i < 10000; ++i) drand48(); // warmup generator

  ttg::Edge<mra::Key<NDIM>, void> project_control;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> project_result, reconstruct_result, multiply_result;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> compress_result, gaxpy_result;

  // define N Gaussians
  std::vector<mra::Gaussian<T, NDIM>> gaussians;
  gaussians.reserve(N);
  // T expnt = 1000.0;
  for (int i = 0; i < N; ++i) {
    T expnt = 1500 + 1500*drand48();
    mra::Coordinate<T,NDIM> r;
    for (size_t d=0; d<NDIM; d++) {
      r[d] = T(-6.0) + T(12.0)*drand48();
    }
    std::cout << "Gaussian " << i << " expnt " << expnt << std::endl;
    gaussians.emplace_back(D, expnt, r);
  }

  // put it into a buffer
  auto gauss_buffer = ttg::Buffer<mra::Gaussian<T, NDIM>>(gaussians.data(), N);
  auto db = ttg::Buffer<mra::Domain<NDIM>>(&D);
  auto start = make_start(project_control);
  auto project = make_project(db, gauss_buffer, N, K, functiondata, T(1e-6), project_control, project_result);
  auto compress = make_compress(N, K, functiondata, project_result, compress_result);
  auto reconstruct = make_reconstruct(N, K, functiondata, compress_result, reconstruct_result);
  auto gaxpy = make_gaxpy(compress_result, compress_result, gaxpy_result, T(1.0), T(-1.0), N, K);
  auto multiply = make_multiply(reconstruct_result, reconstruct_result, multiply_result, functiondata, db, N, K);
  auto printer =   make_printer(project_result,    "projected    ", false);
  auto printer2 =  make_printer(compress_result,   "compressed   ", false);
  auto printer3 =  make_printer(reconstruct_result,"reconstructed", false);
  auto printer4 = make_printer(gaxpy_result, "gaxpy", false);
  auto printer5 = make_printer(multiply_result, "multiply", false);

  auto connected = make_graph_executable(start.get());
  assert(connected);

  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;
  if (ttg::default_execution_context().rank() == 0) {
      //std::cout << "Is everything connected? " << connected << std::endl;
      //std::cout << "==== begin dot ====\n";
      //std::cout << Dot()(start.get()) << std::endl;
      //std::cout << "====  end dot  ====\n";

      beg = std::chrono::high_resolution_clock::now();
      // This kicks off the entire computation
      start->invoke(mra::Key<NDIM>(0, {0}));
  }
  ttg::execute();
  ttg::fence();

  if (ttg::default_execution_context().rank() == 0) {
    end = std::chrono::high_resolution_clock::now();
    std::cout << "TTG Execution Time (milliseconds) : "
              << (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()) / 1000
              << std::endl;
  }
}

template<typename T, mra::Dimension NDIM>
void test_pcr(std::size_t N, std::size_t K) {
  auto functiondata = mra::FunctionData<T,NDIM>(K);
  mra::Domain<NDIM> D;
  D.set_cube(-6.0,6.0);

  srand48(5551212); // for reproducible results
  for (int i = 0; i < 10000; ++i) drand48(); // warmup generator

  ttg::Edge<mra::Key<NDIM>, void> project_control;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> project_result, reconstruct_result, multiply_result;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> compress_result, compress_reconstruct_result, gaxpy_result;
  ttg::Edge<mra::Key<NDIM>, T> norm_result;

  // define N Gaussians
  std::vector<mra::Gaussian<T, NDIM>> gaussians;
  gaussians.reserve(N);
  // T expnt = 1000.0;
  for (int i = 0; i < N; ++i) {
    T expnt = 1500 + 1500*drand48();
    mra::Coordinate<T,NDIM> r;
    for (size_t d=0; d<NDIM; d++) {
      r[d] = T(-6.0) + T(12.0)*drand48();
    }
    std::cout << "Gaussian " << i << " expnt " << expnt << std::endl;
    gaussians.emplace_back(D, expnt, r);
  }

  // put it into a buffer
  auto gauss_buffer = ttg::Buffer<mra::Gaussian<T, NDIM>>(gaussians.data(), N);
  auto db = ttg::Buffer<mra::Domain<NDIM>>(&D);
  auto start = make_start(project_control);
  auto project = make_project(db, gauss_buffer, N, K, functiondata, T(1e-6), project_control, project_result);
  // C(P)
  auto compress = make_compress(N, K, functiondata, project_result, compress_result);
  // // R(C(P))
  auto reconstruct = make_reconstruct(N, K, functiondata, compress_result, reconstruct_result);
  // C(R(C(P)))
  auto compress_r = make_compress(N, K, functiondata, reconstruct_result, compress_reconstruct_result);

  // C(R(C(P))) - C(P)
  auto gaxpy = make_gaxpy(compress_reconstruct_result, compress_result, gaxpy_result, T(1.0), T(-1.0), N, K);
  // | C(R(C(P))) - C(P) |
  auto norm  = make_norm(N, K, gaxpy_result, norm_result);
  // final check
  auto norm_check = ttg::make_tt([&](const mra::Key<NDIM>& key, const T& norm){
    // TODO: check for the norm within machine precision
    std::cout << "Final norm: " << norm << std::endl;
  }, ttg::edges(norm_result), ttg::edges(), "norm-check");

  auto connected = make_graph_executable(start.get());
  assert(connected);

  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;
  if (ttg::default_execution_context().rank() == 0) {
      //std::cout << "Is everything connected? " << connected << std::endl;
      //std::cout << "==== begin dot ====\n";
      //std::cout << Dot()(start.get()) << std::endl;
      //std::cout << "====  end dot  ====\n";

      beg = std::chrono::high_resolution_clock::now();
      // This kicks off the entire computation
      start->invoke(mra::Key<NDIM>(0, {0}));
  }
  ttg::execute();
  ttg::fence();

  if (ttg::default_execution_context().rank() == 0) {
    end = std::chrono::high_resolution_clock::now();
    std::cout << "TTG Execution Time (milliseconds) : "
              << (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()) / 1000
              << std::endl;
  }
}

int main(int argc, char **argv) {
  ttg::initialize(argc, argv);
  mra::GLinitialize();

  // test<double, 3>(1, 10);
  test_pcr<double, 3>(1, 10);

  ttg::finalize();
}
