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
#include "options.h"
#include <any>

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

using namespace mra; // we're lazy

#ifdef MRA_ENABLE_HOST
#define TASKTYPE void
constexpr const ttg::ExecutionSpace Space = ttg::ExecutionSpace::Host;
#elif defined(MRA_ENABLE_CUDA)
#define TASKTYPE ttg::device::Task
constexpr const ttg::ExecutionSpace Space = ttg::ExecutionSpace::CUDA;
#elif defined(MRA_ENABLE_HIP)
#define TASKTYPE ttg::device::Task
constexpr const ttg::ExecutionSpace Space = ttg::ExecutionSpace::HIP;
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
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> result,
  const char *name = "project")
{
  /* create a non-owning buffer for domain and capture it */
  auto fn = [&, N, K, thresh, gl = mra::GLbuffer<T>()]
            (const mra::Key<NDIM>& key) -> TASKTYPE {
    using tensor_type = typename mra::Tensor<T, NDIM+1>;
    using key_type = typename mra::Key<NDIM>;
    using node_type = typename mra::FunctionsReconstructedNode<T, NDIM>;
    node_type result(key, N); // empty for fast-paths, no need to zero out
#ifndef MRA_ENABLE_HOST
    auto outputs = ttg::device::forward();
#endif // MRA_ENABLE_HOST
    auto* fn_arr = fb.host_ptr();
    bool all_initial_level = true;
    for (std::size_t i = 0; i < N; ++i) {
      if (key.level() >= initial_level(fn_arr[i])) {
        all_initial_level = false;
        break;
      }
    }
    if (all_initial_level) {
      //std::cout << "project " << key << " all initial " << std::endl;
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
        T* tmp_ptr = tmp_scratch.device_ptr();
        bool *is_leafs_device = is_leafs_scratch.device_ptr();
        auto *f_ptr   = fb.current_device_ptr();
        auto& domain = *db.current_device_ptr();
        auto  gldata = gl.current_device_ptr();

        /* submit the kernel */
        submit_fcoeffs_kernel(domain, gldata, f_ptr, key, N, K, tmp_ptr,
                              phibar_view, hgT_view, coeffs_view,
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
  return ttg::make_tt<Space>(std::move(fn), ttg::edges(fuse(control, refine)), ttg::edges(refine,result), name);
}

template<mra::Dimension NDIM, typename Value, std::size_t I, std::size_t... Is>
static auto select_send_up(const mra::Key<NDIM>& key, Value&& value,
                           std::index_sequence<I, Is...>, const char *name = "select_send_up") {
  if (key.childindex() == I) {
    //std::cout << name << "-select_send_up " << key << " sending to " << key.parent() << " on " << I << std::endl;
#ifndef MRA_ENABLE_HOST
    return ttg::device::send<I>(key.parent(), std::forward<Value>(value));
#else
    return ttg::send<I>(key.parent(), std::forward<Value>(value));
#endif
  } else if constexpr (sizeof...(Is) > 0){
    return select_send_up(key, std::forward<Value>(value), std::index_sequence<Is...>{}, name);
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
    co_await select_send_up(key, node, std::make_index_sequence<mra::Key<NDIM>::num_children()>{}, "do_send_leafs_up");
#else
    select_send_up(key, node, std::make_index_sequence<mra::Key<NDIM>::num_children()>{}, "do_send_leafs_up");
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
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>>& out,
  const char *name = "compress")
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
  auto do_compress = [&, N, K, name](const mra::Key<NDIM>& key,
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

      /* check if all inputs are empty */
      bool all_empty = in0.empty() && in1.empty() && in2.empty() && in3.empty() &&
                       in4.empty() && in5.empty() && in6.empty() && in7.empty();

      if (all_empty) {
        // Collect child leaf info
        mra::apply_leaf_info(result, in0, in1, in2, in3, in4, in5, in6, in7);
        /* all data is still on the host so the coefficients are zero */
        for (std::size_t i = 0; i < N; ++i) {
          p.sum(i) = 0.0;
        }
        p.set_all_leaf(false);
        // std::cout << name << " " << key << " all empty, all children leafs " << result.is_all_child_leaf() << " ["
        //           << in0.is_all_leaf() << ", " << in1.is_all_leaf() << ", "
        //           << in2.is_all_leaf() << ", " << in3.is_all_leaf() << ", "
        //           << in4.is_all_leaf() << ", " << in5.is_all_leaf() << ", "
        //           << in6.is_all_leaf() << ", " << in7.is_all_leaf() << "] "
        //           << std::endl;
      } else {

        /* some inputs are on the device so submit a kernel */

        // allocate the result
        result = mra::FunctionsCompressedNode<T, NDIM>(key, N, K);
        auto& d = result.coeffs();
        // Collect child leaf info
        mra::apply_leaf_info(result, in0, in1, in2, in3, in4, in5, in6, in7);
        p = mra::FunctionsReconstructedNode<T, NDIM>(key, N, K);
        p.set_all_leaf(false);
        assert(p.is_all_leaf() == false);

        //std::cout << name << " " << key << " all leafs " << result.is_all_child_leaf() << std::endl;

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
        auto input_views = std::array{in0.coeffs().current_view(), in1.coeffs().current_view(), in2.coeffs().current_view(), in3.coeffs().current_view(),
                                      in4.coeffs().current_view(), in5.coeffs().current_view(), in6.coeffs().current_view(), in7.coeffs().current_view()};

        auto coeffs_view = p.coeffs().current_view();
        auto rcoeffs_view = d.current_view();
        auto hgT_view = hgT.current_view();

        submit_compress_kernel(key, N, K, coeffs_view, rcoeffs_view, hgT_view,
                              tmp_scratch.device_ptr(), d_sumsq_scratch.device_ptr(), input_views,
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
          select_send_up(key, std::move(p), std::make_index_sequence<num_children>{}, "compress"),
          // Send result to output tree
          ttg::device::send<out_terminal_id>(key, std::move(result)));
#else
          select_send_up(key, std::move(p), std::make_index_sequence<num_children>{}, "compress");
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
                         ttg::make_tt<Space>(std::move(do_compress), send_to_compress_edges, compress_out_edges, name));
}

template <typename T, mra::Dimension NDIM>
auto make_reconstruct(
  const std::size_t N,
  const std::size_t K,
  const mra::FunctionData<T, NDIM>& functiondata,
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> in,
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> out,
  const char* name = "reconstruct")
{
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T,NDIM>> S("S");  // passes scaling functions down

  auto do_reconstruct = [&, N, K](const mra::Key<NDIM>& key,
                                  const mra::FunctionsCompressedNode<T, NDIM>& node,
                                  const mra::FunctionsReconstructedNode<T, NDIM>& from_parent) -> TASKTYPE {
    const std::size_t tmp_size = reconstruct_tmp_size<NDIM>(K)*N;
    auto tmp = std::make_unique_for_overwrite<T[]>(tmp_size);
    const auto& hg = functiondata.get_hg();
    auto tmp_scratch = ttg::make_scratch(tmp.get(), ttg::scope::Allocate, tmp_size);
    mra::KeyChildren<NDIM> children(key);

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
      // collect leaf information
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
      return ttg::device::Input(hg.buffer(), tmp_scratch,
                                (r_arr[Is].coeffs().buffer())...);
    };
    auto inputs = make_inputs(std::make_index_sequence<mra::Key<NDIM>::num_children()>{});
    if (!from_parent.empty()) {
      inputs.add(from_parent.coeffs().buffer());
    }
    if (!node.empty()) {
      inputs.add(node.coeffs().buffer());
    }
    /* select a device */
    co_await ttg::device::select(inputs);
#endif

    // helper lambda to pick apart the std::array
    auto assemble_tensors = [&]<std::size_t... Is>(std::index_sequence<Is...>){
      return std::array{(r_arr[Is].coeffs().current_view())...};
    };
    auto r_ptrs = assemble_tensors(std::make_index_sequence<mra::Key<NDIM>::num_children()>{});
    auto node_view = node.coeffs().current_view();
    auto hg_view = hg.current_view();
    auto from_parent_view = from_parent.coeffs().current_view();
    submit_reconstruct_kernel(key, N, K, node_view, hg_view, from_parent_view,
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
  auto func = [str,doprint](const keyT& key, const valueT& value) -> TASKTYPE {
    if (doprint) {
#ifndef MRA_ENABLE_HOST
      /* pull the data back to the host */
      co_await ttg::device::select(value.coeffs().buffer());
      co_await ttg::device::wait(value.coeffs().buffer());
#endif // MRA_ENABLE_HOST

      // sanity check
      assert(value.coeffs().buffer().is_current_on(ttg::device::Device()));
      std::lock_guard<std::mutex> obolus(printer_guard);
      std::cout << str << " (" << key << "," << value << ")" << std::endl;
    }
  };
  return ttg::make_tt<Space>(func, ttg::edges(in), ttg::edges(), "printer", {"input"});
}

template<typename T, mra::Dimension NDIM>
auto make_gaxpy(ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> in1,
              ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> in2,
              ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> out,
              const T scalarA, const T scalarB, const size_t N, const size_t K,
              const char* name = "gaxpy")
{
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> S1, S2; // to balance trees

  auto func = [N, K, scalarA, scalarB, name](
            const mra::Key<NDIM>& key,
            const mra::FunctionsCompressedNode<T, NDIM>& t1,
            const mra::FunctionsCompressedNode<T, NDIM>& t2) -> TASKTYPE {

#ifndef MRA_ENABLE_HOST
    auto sends = ttg::device::forward();
    auto send_out = [&]<typename S>(S&& out){
      sends.push_back(ttg::device::send<0>(key, std::forward<S>(out)));
    };
#else
    auto send_out = [&]<typename S>(S&& out){
      ttg::send<0>(key, std::forward<S>(out));
    };
#endif

    //std::cout << name << " " << key << " t1 empty " << t1.empty() << " t2 empty " << t2.empty() << std::endl;

    /**
     * We can forward inputs only if the scalars are right and if the other input is empty and
     * all its children are leafs. Otherwise we need to compute the GAXPY and/or adjust the leaf information.
     */
    if ((t1.empty() && t2.empty())) {
      /* send out an empty result */
      auto out = mra::FunctionsCompressedNode<T, NDIM>(key, N); // out -> result
      mra::apply_leaf_info(out, t1, t2);
      //std::cout << name << " " << key << " t1 empty, t2 empty, all leafs " << out.is_all_child_leaf() << std::endl;
      send_out(std::move(out));
    } else if ((scalarA == 0.0 || (t1.empty() && scalarB == 1.0)) && t1.is_all_child_leaf()) {
      // just send t2, t1 is empty and all children are leafs
      send_out(t2);
    } else if ((scalarB == 0.0 || (t2.empty() && scalarA == 1.0)) && t2.is_all_child_leaf()) {
      // just send t1, t2 is empty and all children are leafs
      send_out(t1);
    } else {

      auto out = mra::FunctionsCompressedNode<T, NDIM>(key, N, K);
      out.coeffs().buffer().reset_scope(ttg::scope::Allocate);
      /* adapt the leaf information of the result: if the children of both nodes are leafs then
       * the children of the output node are leafs as well. */

      mra::apply_leaf_info(out, t1, t2);
      //std::cout << name << " " << key << " all leafs " << out.is_all_child_leaf() << std::endl;
#if 0
      for (size_type i = 0; i < N; ++i) {
        for (auto child : children(key)) {
          auto childidx = child.childindex();
          out.is_child_leaf(i)[childidx] = t1.is_child_leaf(i)[childidx] && t2.is_child_leaf(i)[childidx];
        }
      }
#endif // 0

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

    std::vector<mra::Key<NDIM>> child_keys_left, child_keys_right;
    /**
     * Check for each child whether all functions are leafs in t1 and t2.
     * For any child where all functions are leafs in only one side, we need to broadcast an empty node.
     */
    for (auto child : children(key)) {
      const auto childidx = child.childindex();
      bool t1_all_child_leaf = true, t2_all_child_leaf = true;
      for (size_type i = 0; i < N && (t1_all_child_leaf | t2_all_child_leaf); ++i) {
        if (!t1.is_child_leaf(i, childidx)) {
          t1_all_child_leaf = false;
        }
        if (!t2.is_child_leaf(i, childidx)) {
          t2_all_child_leaf = false;
        }
      }
      if (t1_all_child_leaf && !t2_all_child_leaf) {
        //std::cout << name << " " << key << " balancing tree to left " << child << std::endl;
        child_keys_left.push_back(child);
      }
      if (!t1_all_child_leaf && t2_all_child_leaf) {
        //std::cout << name << " " << key << " balancing tree to right " << child << std::endl;
        child_keys_right.push_back(child);
      }
    }
    if (child_keys_left.size() > 0) {
#ifndef MRA_ENABLE_HOST
      sends.push_back(ttg::device::broadcast<1>(
                        std::move(child_keys_left),
                        mra::FunctionsCompressedNode<T, NDIM>(key, N)));
#else
      ttg::broadcast<1>(std::move(child_keys_left),
                        mra::FunctionsCompressedNode<T, NDIM>(key, N));
#endif // MRA_ENABLE_HOST
    }
    if (child_keys_right.size() > 0) {
#ifndef MRA_ENABLE_HOST
      sends.push_back(ttg::device::broadcast<2>(
                        std::move(child_keys_right),
                        mra::FunctionsCompressedNode<T, NDIM>(key, N)));
#else
      ttg::broadcast<2>(std::move(child_keys_right),
                        mra::FunctionsCompressedNode<T, NDIM>(key, N));
#endif // MRA_ENABLE_HOST
    }

#ifndef MRA_ENABLE_HOST
     co_await std::move(sends);
#endif // MRA_ENABLE_HOST
  };

  return ttg::make_tt<Space>(std::move(func),
                             ttg::edges(ttg::fuse(S1, in1), ttg::fuse(S2, in2)),
                             ttg::edges(out, S1, S2), name,
                             {"in1", "in2"},
                             {"out", "S1", "S2"});
}

template<typename T, mra::Dimension NDIM>
auto make_gaxpy(ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> in1,
              ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> out,
              const T scalarA,
              const size_t N,
              const size_t K,
              const char* name = "gaxpy_ovld")
{
  auto func = [N](const mra::Key<NDIM>& key){
    return FunctionsCompressedNode<T, NDIM>(key, N);
  };

  auto rank = ttg::default_execution_context().rank();

  auto pmap = [rank](const mra::Key<NDIM>& key){
    return rank;
  };

  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> dummy{"gaxpy_dummy", true,
                                                                          {std::any{}, std::move(func), std::move(pmap)}};


  return make_gaxpy(in1, dummy, out, scalarA, 0.0, N, K, name);
}

template<typename T, mra::Dimension NDIM>
auto make_multiply(ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> in1,
              ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> in2,
              ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> out,
              const mra::FunctionData<T, NDIM>& functiondata,
              const ttg::Buffer<mra::Domain<NDIM>>& db, const size_t N, const size_t K,
              const char* name = "multiply")
{
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> S1, S2; // to balance trees

  auto func = [&, N, K](
            const mra::Key<NDIM>& key,
            const mra::FunctionsReconstructedNode<T, NDIM>& t1,
            const mra::FunctionsReconstructedNode<T, NDIM>& t2) -> TASKTYPE {

#ifndef MRA_ENABLE_HOST
    auto sends = ttg::device::forward();
    auto send_out = [&]<typename S>(S&& out){
      sends.push_back(ttg::device::send<0>(key, std::forward<S>(out)));
    };
#else
    auto send_out = [&]<typename S>(S&& out){
      ttg::send<0>(key, std::forward<S>(out));
    };
#endif

    if (t1.empty() || t2.empty()) {
      /* send out an empty result */
      auto out = mra::FunctionsReconstructedNode<T, NDIM>(key, N);
      mra::apply_leaf_info(out, t1, t2);
      send_out(std::move(out));
    } else {
      auto out = mra::FunctionsReconstructedNode<T, NDIM>(key, N, K);
      mra::apply_leaf_info(out, t1, t2);
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
                             ttg::edges(out, S1, S2), name,
                             {"in1", "in2"},
                             {"out", "S1", "S2"});
}


/* forward a reconstructed function node to the right input of do_compress
 * this is a device task to prevent data from being pulled back to the host
 * even though it will not actually perform any computation */
template<typename T, mra::Dimension NDIM>
static TASKTYPE send_norms_up(const mra::Key<NDIM>& key, const mra::Tensor<T, 1>& node) {
#ifndef MRA_ENABLE_HOST
  co_await select_send_up(key, node, std::make_index_sequence<mra::Key<NDIM>::num_children()>{}, "send-norms-up");
#else
  select_send_up(key, node, std::make_index_sequence<mra::Key<NDIM>::num_children()>{}, "send-norms-up");
#endif
}


template <typename T, Dimension NDIM>
auto make_norm(size_type N, size_type K,
               ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> input,
               ttg::Edge<mra::Key<NDIM>, mra::Tensor<T, 1>> result,
               const char* name = "norm") {
  static_assert(NDIM == 3); // TODO: worth fixing?
  using norm_tensor_type = mra::Tensor<T, 1>;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> node_e;
  ttg::Edge<mra::Key<NDIM>, norm_tensor_type> norm_e0, norm_e1, norm_e2, norm_e3, norm_e4, norm_e5, norm_e6, norm_e7;
  static constexpr const int num_children = mra::Key<NDIM>::num_children();

  /**
   * Takes a tensor of norms from each child
   */
  auto norm_fn = [N, K, name](const mra::Key<NDIM>& key,
                              const norm_tensor_type& norm0,
                              const norm_tensor_type& norm1,
                              const norm_tensor_type& norm2,
                              const norm_tensor_type& norm3,
                              const norm_tensor_type& norm4,
                              const norm_tensor_type& norm5,
                              const norm_tensor_type& norm6,
                              const norm_tensor_type& norm7,
                              const mra::FunctionsCompressedNode<T, NDIM>& in) -> TASKTYPE {
    // TODO: pass ttg::scope::Allocate once that's possible
    // TODO: reuse of one of the input norms?
    auto norms_result = norm_tensor_type(N);
    //std::cout << name << " " << key << std::endl;
#ifndef MRA_ENABLE_HOST
    co_await ttg::device::select(norms_result.buffer(), in.coeffs().buffer(),
                                 norm0.buffer(), norm1.buffer(), norm2.buffer(), norm3.buffer(),
                                 norm4.buffer(), norm5.buffer(), norm6.buffer(), norm7.buffer());
#endif // MRA_ENABLE_HOST
    auto node_view = in.coeffs().current_view();
    auto norm_result_view = norms_result.current_view();
    std::array<const T*, mra::Key<NDIM>::num_children()> child_norms =
        {norm0.buffer().current_device_ptr(), norm1.buffer().current_device_ptr(),
         norm2.buffer().current_device_ptr(), norm3.buffer().current_device_ptr(),
         norm4.buffer().current_device_ptr(), norm5.buffer().current_device_ptr(),
         norm6.buffer().current_device_ptr(), norm7.buffer().current_device_ptr()};
    submit_norm_kernel(key, N, K, node_view, norm_result_view, child_norms, ttg::device::current_stream());

#ifndef MRA_ENABLE_HOST
    if (key.level() == 0) {
      // send to result
      //std::cout << name << " " << key << " sending to result " << std::endl;
      co_await ttg::device::send<num_children>(key, std::move(norms_result));
    } else {
      // send norms upstream
      co_await select_send_up(key, std::move(norms_result), std::make_index_sequence<num_children>{}, "norm");
    }
#else
    if (key.level() == 0) {
      // send to result
      //std::cout << "norm send to result " << key << std::endl;
      ttg::send<num_children>(key, std::move(norms_result));
    } else {
      // send norms upstream
      //std::cout << "norm send up " << key << std::endl;
      select_send_up(key, std::move(norms_result), std::make_index_sequence<num_children>{}, "norm");
    }
#endif // MRA_ENABLE_HOST
  };

  auto norm_tt = ttg::make_tt<Space>(std::move(norm_fn),
                                     ttg::edges(norm_e0, norm_e1, norm_e2, norm_e3, norm_e4, norm_e5, norm_e6, norm_e7, node_e),
                                     ttg::edges(norm_e0, norm_e1, norm_e2, norm_e3, norm_e4, norm_e5, norm_e6, norm_e7, result),
                                     name);

  /**
   * Task to dispatch incoming compressed nodes and forward empty
   * nodes for children that do not exist
   */
  auto dispatch_fn = [N, K, name]
                   (const mra::Key<NDIM>& key,
                    const mra::FunctionsCompressedNode<T, NDIM>& in) -> TASKTYPE {
    //std::cout << name << "-dispatch " << key << " sending node to " << num_children << std::endl;
#ifndef MRA_ENABLE_HOST
    auto sends = ttg::device::forward(ttg::device::send<num_children>(key, in));
#else  // MRA_ENABLE_HOST
    ttg::send<num_children>(key, in);
#endif // MRA_ENABLE_HOST
    /* feed empty tensor to all */
    for (auto child : children(key)) {
      bool is_all_leaf = true;
      const auto childidx = child.childindex();
      for (size_type i = 0; i < N && is_all_leaf; ++i) {
        is_all_leaf &= in.is_child_leaf(i, childidx);
      }
      //std::cout << "norm dispatch " << key << " child " << child << " all leaf " << is_all_leaf << std::endl;
      if (is_all_leaf) {
        //std::cout << name << "-dispatch " << key << " sending empty norms to child " << childidx << " " << child << std::endl;
        // pass up a null tensor
#ifndef MRA_ENABLE_HOST
        sends.push_back(select_send_up(child, mra::Tensor<T, 1>(), std::make_index_sequence<num_children>{}, "dispatch"));
#else  // MRA_ENABLE_HOST
        select_send_up(child, mra::Tensor<T, 1>(), std::make_index_sequence<num_children>{}, "dispatch");
#endif // MRA_ENABLE_HOST
      } else {
        /* if not all children are leafs the norm task will receive norms from somewhere
         * so there is nothing to be done here */
        //std::cout << name << "-dispatch " << key << " child " << childidx << " " << child << " has not all leaf" << std::endl;
      }
    }
#ifndef MRA_ENABLE_HOST
    co_await std::move(sends);
#endif // MRA_ENABLE_HOST
  };

  auto dispatch_tt = ttg::make_tt<Space>(std::move(dispatch_fn),
                                         ttg::edges(input),     // main input
                                         ttg::edges(norm_e0, norm_e1, norm_e2, norm_e3,
                                                    norm_e4, norm_e5, norm_e6, norm_e7, node_e),
                                         "norm-dispatch");
  /* compile everything into tasks */
  return std::make_tuple(std::move(norm_tt),
                         std::move(dispatch_tt));
}

template <typename T, Dimension NDIM>
auto make_derivative(size_type N, size_type K,
               ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> in,
               ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> result,
               const mra::FunctionData<T, NDIM>& functiondata,
               const ttg::Buffer<mra::Domain<NDIM>>& db,
               const T g1,
               const T g2,
               const Dimension axis,
               const int bc_left,
               const int bc_right,
               const std::string& name = "derivative")
{
  // TODO: we could generalize this to NDIM by using the tuple-based API
  static_assert(NDIM == 3, "Derivative currently only supported in 3D!");
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> left, center, right;
  // output terminal offsets
  constexpr static const int LEFT  = 0;
  constexpr static const int CENTER  = 1;
  constexpr static const int RIGHT = 2;
  constexpr static const int RESULT = RIGHT+1;

  if (axis < 0 || axis >= NDIM) {
    throw std::runtime_error("Invalid axis for derivative");
  }

  auto dispatch_fn = [&, axis](const mra::Key<NDIM>& key,
                        const mra::FunctionsReconstructedNode<T, NDIM>& in_node) -> TASKTYPE {

    //std::cout << "derivative dispatch " << key << " axis " << axis << std::endl;
#ifndef MRA_ENABLE_HOST
    // forward() returns a vector that we can push into
    auto sends = ttg::device::forward(ttg::device::send<CENTER>(key, in_node));
    auto do_send = [&]<std::size_t I, typename S>(auto& child, S&& node) {
      sends.push_back(ttg::device::send<I>(child, std::forward<S>(node)));
    };
#else
    ttg::send<CENTER>(key, in_node);
    auto do_send = []<std::size_t I, typename S>(auto& k, S&& node) {
      ttg::send<I>(k, std::forward<S>(node));
    };
#endif // MRA_ENABLE_HOST


    bool has_right = !key.is_right_boundary(axis);
    bool has_left  = !key.is_left_boundary(axis);

    //std::cout << "derivative dispatch " << key << " axis " << D << " has_left " << has_left << " has_right " << has_right << std::endl;

    if (has_right){
      mra::Key<NDIM> right_key = key.neighbor(axis, 1);
      //std::cout << "derivative dispatch " << key << " has right, sending to right neighbor " << right_key << std::endl;
      do_send.template operator()<LEFT>(right_key, in_node);
    } else {
      //std::cout << "derivative dispatch " << key << " has no right, sending to right input " << key << std::endl;
      do_send.template operator()<RIGHT>(key, mra::FunctionsReconstructedNode<T, NDIM>());
    }

    if (has_left){
      mra::Key<NDIM> left_key = key.neighbor(axis, -1);
      //std::cout << "derivative dispatch " << key << " has left, sending to left neighbor " << left_key << std::endl;
      do_send.template operator()<RIGHT>(left_key, in_node);
    } else {
      //std::cout << "derivative dispatch " << key << " has no left, sending to left input " << key << std::endl;
      do_send.template operator()<LEFT>(key, mra::FunctionsReconstructedNode<T, NDIM>());
    }
  };

  auto dispatch_tt = ttg::make_tt<Space>(std::move(dispatch_fn),
                                         ttg::edges(in),
                                         ttg::edges(left, center, right),
                                         name+"-dispatch");

  auto derivative_fn = [&, N, K, g1, g2, axis, bc_left, bc_right](
                              const mra::Key<NDIM>& key,
                              const mra::FunctionsReconstructedNode<T, NDIM>& left,
                              const mra::FunctionsReconstructedNode<T, NDIM>& center,
                              const mra::FunctionsReconstructedNode<T, NDIM>& right) -> TASKTYPE {

    /* tuple of references to inputs */
    auto inputs = std::array{std::cref(left), std::cref(center), std::cref(right)};

#ifndef MRA_ENABLE_HOST
    // forward() returns a vector that we can push into
    auto sends = ttg::device::forward();
    auto do_send = [&]<std::size_t I, typename S>(auto& child, S&& node) {
      sends.push_back(ttg::device::send<I>(child, std::forward<S>(node)));
    };
#else
    auto do_send = []<std::size_t I, typename S>(auto& k, S&& node) {
      ttg::send<I>(k, std::forward<S>(node));
    };
#endif // MRA_ENABLE_HOST

    //std::cout << "derivative " << key << std::endl;

    if (center.empty()){
      //std::cout << "derivative " << key << " center empty" << std::endl;
      /**
       * We received an empty center. If the left node is not empty, we need to refine it to the left child.
       * If the right node is not empty, we need to refine it to the right child.
       * These children will eventually receive a non-empty center node.
       */
      for (auto child : children(key)) {
        if (!left.empty()){
          if (child.is_left_child(axis)) { // skip right children
            //std::cout << "derivative " << key << " left not empty, sending to left child " << child << std::endl;
            do_send.template operator()<LEFT>(child, left);
          }
        }

        if (!right.empty()){
          if (child.is_right_child(axis)) { // skip left children
            //std::cout << "derivative " << key << " right not empty, sending to right child " << child << std::endl;
            do_send.template operator()<RIGHT>(child, right);
          }
        }
      }
      /* send an empty node as the result */
      do_send.template operator()<RESULT>(key, mra::FunctionsReconstructedNode<T, NDIM>());
    } else { // center is not empty

      /**
       * Check if we have to refine down in all dimensions. This is necessary if one
       * of the neighbors is empty and we are not at a boundary.
       */
      bool need_refinement = (left.empty() && !key.is_left_boundary(axis)) || (right.empty() && !key.is_right_boundary(axis));

      if (need_refinement) {

        auto make_empty = []{ return mra::FunctionsReconstructedNode<T, NDIM>(); };

        /**
         * Send center to all children.
         */
        for (auto child : children(key)) {
          //std::cout << "derivative " << key << " left or right empty, sending center " << center.key() << " to child " << child << " center input" << std::endl;
          do_send.template operator()<CENTER>(child, center);

          //std::cout << "derivative " << key << " center not empty, balance axis " << D << std::endl;

          /**
           * Handle left input
           */

          /* only refine down if the left node is not empty or we are at a boundary
           * if the left node is empty the children will receive a left node from their neighbor */
          if (!left.empty() || key.is_left_boundary(axis)) {
            if (child.is_left_child(axis)) { // skip right children
              // send left (if not empty) or center to left children
              if (key.is_left_boundary(axis)) {
                //std::cout << "derivative " << key << " center not empty, sending " << make_empty().key()
                //          << " to left child " << child << " left input" << std::endl;
                do_send.template operator()<LEFT>(child, make_empty());
              } else {
                //std::cout << "derivative " << key << " center not empty, sending " << left.key()
                //          << " to left child " << child << " left input" << std::endl;
                do_send.template operator()<LEFT>(child, left);
              }
            }
          }

          /* Send the center node to the left inputs of right children */
          if (child.is_right_child(axis)) { // skip left children
            //std::cout << "derivative " << key << " center not empty, sending " << center.key()
            //          << " to right child " << child << " left input" << std::endl;
            do_send.template operator()<LEFT>(child, center);
          }

          /**
           * Handle right input
           */
          if (!right.empty() || key.is_right_boundary(axis)) {
            if (child.is_right_child(axis)) { // skip left children
              if (key.is_right_boundary(axis)) {
                //std::cout << "derivative " << key << " center not empty, sending " << make_empty().key()
                //          << " to right child " << child << " right input" << std::endl;
                do_send.template operator()<RIGHT>(child, make_empty());
              } else {
                //std::cout << "derivative " << key << " center not empty, sending " << right.key()
                //          << " to right child " << child << " right input" << std::endl;
                do_send.template operator()<RIGHT>(child, right);
              }
            }
          }

          /* Send the center node to the right inputs of left children */
          if (child.is_left_child(axis)) { // skip right children
            //std::cout << "derivative " << key << " center not empty, sending " << center.key()
            //          << " to left child " << child << " right input" << std::endl;
            do_send.template operator()<RIGHT>(child, center);
          }
        }

        /* send an empty node as result */
        do_send.template operator()<RESULT>(key, mra::FunctionsReconstructedNode<T, NDIM>());
      } else {
        /**
         * We can finally compute the derivative.
         */
        if ((!left.empty() || key.is_left_boundary(axis)) && (!right.empty() || key.is_right_boundary(axis))){
          mra::FunctionsReconstructedNode<T, NDIM> result(key, N, K);
          result.set_all_leaf(true);
          ttg::Buffer<T> tmp = ttg::Buffer<T>(derivative_tmp_size<NDIM>(K)*N);
          const Tensor<T, 2+1>& operators = functiondata.get_operators();
          const Tensor<T, 2>& phibar= functiondata.get_phibar();
          const Tensor<T, 2>& phi= functiondata.get_phi();
          const Tensor<T, 1>& quad_x = functiondata.get_quad_x();

#ifndef MRA_ENABLE_HOST
          co_await ttg::device::select(db, left.coeffs().buffer(), center.coeffs().buffer(),
                                      right.coeffs().buffer(), result.coeffs().buffer(), operators.buffer(),
                                      phibar.buffer(), phi.buffer(), quad_x.buffer(), tmp);
#endif // MRA_ENABLE_HOST

          auto& D = *db.current_device_ptr();
          auto result_view = result.coeffs().current_view();
          submit_derivative_kernel(D, key, left.key(), center.key(), right.key(), left.coeffs().current_view(),
                                  center.coeffs().current_view(), right.coeffs().current_view(), operators.current_view(),
                                  result_view, phi.current_view(), phibar.current_view(), quad_x.current_view(),
                                  tmp.current_device_ptr(), N, K, g1, g2, axis, bc_left, bc_right, ttg::device::current_stream());

          do_send.template operator()<RESULT>(key, std::move(result));
        }
      }
#ifndef MRA_ENABLE_HOST
      co_await std::move(sends);
#endif
    }
  };

  auto deriv_tt = ttg::make_tt<Space>(std::move(derivative_fn),
                             ttg::edges(left, center, right),
                             ttg::edges(left, center, right, result),
                             name);

  return std::make_tuple(std::move(deriv_tt),
                         std::move(dispatch_tt));
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
  auto D = std::make_unique<mra::Domain<NDIM>[]>(1);
  D[0].set_cube(-6.0,6.0);
  T g1 = 0;
  T g2 = 0;
  Dimension axis = 1;

  srand48(5551212); // for reproducible results
  for (int i = 0; i < 10000; ++i) drand48(); // warmup generator

  ttg::Edge<mra::Key<NDIM>, void> project_control;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> project_result, reconstruct_result, multiply_result;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> compress_result, gaxpy_result;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> derivative_result;

  // define N Gaussians
  auto gaussians = std::make_unique<mra::Gaussian<T, NDIM>[]>(N);
  // T expnt = 1000.0;
  for (int i = 0; i < N; ++i) {
    T expnt = 1500 + 1500*drand48();
    mra::Coordinate<T,NDIM> r;
    for (size_t d=0; d<NDIM; d++) {
      r[d] = T(-6.0) + T(12.0)*drand48();
    }
    std::cout << "Gaussian " << i << " expnt " << expnt << std::endl;
    gaussians[i] = mra::Gaussian<T, NDIM>(D[0], expnt, r);
  }

  // put it into a buffer
  auto gauss_buffer = ttg::Buffer<mra::Gaussian<T, NDIM>>(std::move(gaussians), N);
  auto db = ttg::Buffer<mra::Domain<NDIM>>(std::move(D), 1);
  auto start = make_start(project_control);
  auto project = make_project(db, gauss_buffer, N, K, functiondata, T(1e-6), project_control, project_result);
  auto compress = make_compress(N, K, functiondata, project_result, compress_result);
  auto reconstruct = make_reconstruct(N, K, functiondata, compress_result, reconstruct_result);
  auto gaxpy = make_gaxpy(compress_result, compress_result, gaxpy_result, T(1.0), T(-1.0), N, K);
  auto multiply = make_multiply(reconstruct_result, reconstruct_result, multiply_result, functiondata, db, N, K);
  auto derivative = make_derivative(N, K, multiply_result, derivative_result, functiondata, db, g1, g2, axis,
                                    FunctionData<T, NDIM>::BC_DIRICHLET, FunctionData<T, NDIM>::BC_DIRICHLET, "derivative");
  auto printer =   make_printer(project_result,    "projected    ", false);
  auto printer2 =  make_printer(compress_result,   "compressed   ", false);
  auto printer3 =  make_printer(reconstruct_result,"reconstructed", false);
  auto printer4 = make_printer(gaxpy_result, "gaxpy", false);
  auto printer5 = make_printer(multiply_result, "multiply", false);
  auto printer6 = make_printer(derivative_result, "derivative", false);

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
  auto D = std::make_unique<mra::Domain<NDIM>[]>(1);
  D[0].set_cube(-6.0,6.0);

  srand48(5551212); // for reproducible results
  for (int i = 0; i < 10000; ++i) drand48(); // warmup generator

  ttg::Edge<mra::Key<NDIM>, void> project_control;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> project_result, reconstruct_result, multiply_result;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> compress_result, compress_reconstruct_result, gaxpy_result;
  ttg::Edge<mra::Key<NDIM>, mra::Tensor<T, 1>> norm_result;

  // define N Gaussians
  auto gaussians = std::make_unique<mra::Gaussian<T, NDIM>[]>(N);
  // T expnt = 1000.0;
  for (int i = 0; i < N; ++i) {
    T expnt = 1500 + 1500*drand48();
    mra::Coordinate<T,NDIM> r;
    for (size_t d=0; d<NDIM; d++) {
      r[d] = T(-6.0) + T(12.0)*drand48();
    }
    std::cout << "Gaussian " << i << " expnt " << expnt << std::endl;
    gaussians[i] = mra::Gaussian<T, NDIM>(D[0], expnt, r);
  }

  // put it into a buffer
  auto gauss_buffer = ttg::Buffer<mra::Gaussian<T, NDIM>>(std::move(gaussians), N);
  auto db = ttg::Buffer<mra::Domain<NDIM>>(std::move(D), 1);
  auto start = make_start(project_control);
  auto project = make_project(db, gauss_buffer, N, K, functiondata, T(1e-6), project_control, project_result);
  // C(P)
  auto compress = make_compress(N, K, functiondata, project_result, compress_result, "compress-cp");
  // // R(C(P))
  auto reconstruct = make_reconstruct(N, K, functiondata, compress_result, reconstruct_result, "reconstruct-rcp");
  // C(R(C(P)))
  auto compress_r = make_compress(N, K, functiondata, reconstruct_result, compress_reconstruct_result, "compress-crcp");

  // C(R(C(P))) - C(P)
  auto gaxpy = make_gaxpy(compress_reconstruct_result, compress_result, gaxpy_result, T(1.0), T(-1.0), N, K);
  // | C(R(C(P))) - C(P) |
  auto norm  = make_norm(N, K, gaxpy_result, norm_result);
  // final check
  auto norm_check = ttg::make_tt([&](const mra::Key<NDIM>& key, const mra::Tensor<T, 1>& norms){
    // TODO: check for the norm within machine precision
    auto norms_arr = norms.buffer().current_device_ptr();
    for (size_type i = 0; i < N; ++i) {
      std::cout << "Final norm " << i << ": " << norms_arr[i] << std::endl;
    }
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

template<typename T, mra::Dimension NDIM>
void test_derivative(std::size_t N, std::size_t K, Dimension axis, T precision) {
  auto functiondata = mra::FunctionData<T,NDIM>(K);
  auto D = std::make_unique<mra::Domain<NDIM>[]>(1);
  D[0].set_cube(-6.0,6.0);
  T g1 = 0;
  T g2 = 0;

  srand48(5551212); // for reproducible results
  for (int i = 0; i < 10000; ++i) drand48(); // warmup generator

  ttg::Edge<mra::Key<NDIM>, void> project_control, project_d_control;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> project_result, reconstruct_result, multiply_result;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> compress_result, compress_derivative_result, gaxpy_result;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> compress_d_result;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> derivative_result, project_d_result;
  ttg::Edge<mra::Key<NDIM>, mra::Tensor<T, 1>> norm_result;

  // define N Gaussians
  auto gaussians = std::make_unique<mra::Gaussian<T, NDIM>[]>(N);
  auto gaussians_deriv = std::make_unique<mra::GaussianDerivative<T, NDIM>[]>(N);
  T expnt = 1000.0;
  T factor = expnt;

  for (int i = 0; i < N; ++i) {
    // T expnt = 1500 + 1500*drand48();
    mra::Coordinate<T,NDIM> r;
    for (size_t d=0; d<NDIM; d++) {
      r[d] = T(-6.0) + T(12.0)*drand48();
    }
    std::cout << "Gaussian " << i << " expnt " << expnt << std::endl;
    std::cout << "GaussianDerivative " << i << " expnt " << expnt << std::endl;
    gaussians[i] = mra::Gaussian<T, NDIM>(D[0], expnt, r);
    gaussians_deriv[i] = mra::GaussianDerivative<T, NDIM>(D[0], expnt, r);
  }

  // put it into a buffer
  auto gauss_buffer = ttg::Buffer<mra::Gaussian<T, NDIM>>(std::move(gaussians), N);
  auto gauss_deriv_buffer = ttg::Buffer<mra::GaussianDerivative<T, NDIM>>(std::move(gaussians_deriv), N);
  auto db = ttg::Buffer<mra::Domain<NDIM>>(std::move(D), 1);
  auto start = make_start(project_control);
  // auto start_d = make_start(project_d_control);
  auto project = make_project(db, gauss_buffer, N, K, functiondata, precision, project_control, project_result);
  auto project_d = make_project(db, gauss_deriv_buffer, N, K, functiondata, precision, project_control, project_d_result);
  // C(P)
  auto compress = make_compress(N, K, functiondata, project_result, compress_result, "compress-cp");
  auto compress_d = make_compress(N, K, functiondata, project_d_result, compress_d_result, "compress-Dcp");
  // // R(C(P))
  auto reconstruct = make_reconstruct(N, K, functiondata, compress_result, reconstruct_result, "reconstruct-rcp");
  // D(R(C(P)))
  auto derivative = make_derivative(N, K, reconstruct_result, derivative_result, functiondata, db, g1, g2, axis,
                                    FunctionData<T, NDIM>::BC_DIRICHLET, FunctionData<T, NDIM>::BC_DIRICHLET, "derivative");

  // C(D(R(C(P))))
  auto compress_r = make_compress(N, K, functiondata, derivative_result, compress_derivative_result, "compress-deriv-crcp");

  // | C(D(R(C(P)))) - factor * C(P) |
  auto gaxpy_r = make_gaxpy(compress_derivative_result, compress_d_result, gaxpy_result, T(1.0), T(-1.0), N, K, "gaxpy");

  auto norm  = make_norm(N, K, gaxpy_result, norm_result);
  // final check
  auto norm_check = ttg::make_tt([&](const mra::Key<NDIM>& key, const mra::Tensor<T, 1>& norms){
    // TODO: check for the norm within machine precision
    auto norms_arr = norms.buffer().current_device_ptr();
    for (size_type i = 0; i < N; ++i) {
      std::cout << "Final norm " << i << ": " << norms_arr[i] << std::endl;
    }
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

  /* options */
  auto opt = mra::OptionParser(argc, argv);
  size_type N = opt.parse("-N", 1);
  size_type K = opt.parse("-K", 10);
  int cores   = opt.parse("-c", -1); // -1: use all cores
  int axis    = opt.parse("-a", 1);
  int log_precision = opt.parse("-p", 4); // default: 1e-4

  ttg::initialize(argc, argv, cores);
  mra::GLinitialize();

  // test<double, 3>(1, 10);
  test_derivative<double, 3>(N, K, axis, std::pow(10, -log_precision));

  ttg::finalize();
}
