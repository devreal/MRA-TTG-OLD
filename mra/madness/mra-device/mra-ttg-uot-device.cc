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

#ifdef TTG_ENABLE_HOST
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
    auto result = node_type(key, N, K);
    tensor_type& coeffs = result.coeffs();
    auto outputs = ttg::device::forward();
    auto* fn_arr = fb.host_ptr();
    bool all_initial_level = true;
    for (std::size_t i = 0; i < N; ++i) {
      all_initial_level &= (key.level() < initial_level(fn_arr[i]));
    }
    //std::cout << "project " << key << " all initial " << all_initial_level << std::endl;
    if (all_initial_level) {
      std::vector<mra::Key<NDIM>> bcast_keys;
      /* TODO: children() returns an iteratable object but broadcast() expects a contiguous memory range.
                We need to fix broadcast to support any ranges */
      for (auto child : children(key)) bcast_keys.push_back(child);

#ifndef TTG_ENABLE_HOST
      outputs.push_back(ttg::device::broadcastk<0>(std::move(bcast_keys)));
#else
      ttg::broadcastk<0>(std::move(bcast_keys));
#endif
      coeffs.current_view() = T(1e7); // set to obviously bad value to detect incorrect use
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
        /* zero coeffs */
        coeffs.current_view() = T(0.0);
        result.set_all_leaf(true);
      } else {
        /* here we actually compute: first select a device */
        //result.is_leaf = fcoeffs(f, functiondata, key, thresh, coeffs);
        /**
         * BEGIN FCOEFFS HERE
         * TODO: figure out a way to outline this into a function or coroutine
         */

        /* global function data */
        // TODO: need to make our own FunctionData with dynamic K
        const auto& phibar = functiondata.get_phibar();
        const auto& hgT = functiondata.get_hgT();

        /* temporaries */
        /* TODO: have make_scratch allocate pinned memory for us */
        auto is_leafs = std::make_unique_for_overwrite<bool[]>(N);
        auto is_leafs_scratch = ttg::make_scratch(is_leafs.get(), ttg::scope::Allocate, N);
        const std::size_t tmp_size = project_tmp_size<NDIM>(K)*N;
        auto tmp = std::make_unique_for_overwrite<T[]>(tmp_size);
        auto tmp_scratch = ttg::make_scratch(tmp.get(), ttg::scope::Allocate, tmp_size);

        /* coeffs don't have to be synchronized into the device */
        coeffs.buffer().reset_scope(ttg::scope::Allocate);

        /* TODO: cannot do this from a function, had to move it into the main task */
  #ifndef TTG_ENABLE_HOST
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
  #ifndef TTG_ENABLE_HOST
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
#ifndef TTG_ENABLE_HOST
        outputs.push_back(ttg::device::broadcastk<0>(std::move(bcast_keys)));
#else
        ttg::broadcastk<0>(bcast_keys);
#endif
      }

    }
#ifndef TTG_ENABLE_HOST
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
#ifndef TTG_ENABLE_HOST
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
#ifndef TTG_ENABLE_HOST
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
      mra::FunctionsCompressedNode<T,NDIM> result(key, N, K); // The eventual result
      auto& d = result.coeffs();
      // allocate even though we might not need it
      mra::FunctionsReconstructedNode<T, NDIM> p(key, N, K);

      /* d and p don't have to be synchronized into the device */
      d.buffer().reset_scope(ttg::scope::Allocate);
      p.coeffs().buffer().reset_scope(ttg::scope::Allocate);

      for (std::size_t i = 0; i < N; ++i) {  // Collect child leaf info
        result.is_child_leaf(i) = std::array{in0.is_leaf(i), in1.is_leaf(i), in2.is_leaf(i),
                                             in3.is_leaf(i), in4.is_leaf(i), in5.is_leaf(i),
                                             in6.is_leaf(i), in7.is_leaf(i)};
      }

      /* stores sumsq for each child and for result at the end of the kernel */
      const std::size_t tmp_size = compress_tmp_size<NDIM>(K)*N;
      auto tmp = std::make_unique_for_overwrite<T[]>(tmp_size);
      const auto& hgT = functiondata.get_hgT();
      auto tmp_scratch = ttg::make_scratch(tmp.get(), ttg::scope::Allocate, tmp_size);
      auto d_sumsq = std::make_unique_for_overwrite<T[]>(N);
      auto d_sumsq_scratch = ttg::make_scratch(d_sumsq.get(), ttg::scope::Allocate, N);
#ifndef TTG_ENABLE_HOST
      co_await ttg::device::select(p.coeffs().buffer(), d.buffer(), hgT.buffer(),
                                   tmp_scratch, d_sumsq_scratch,
                                   in0.coeffs().buffer(), in1.coeffs().buffer(),
                                   in2.coeffs().buffer(), in3.coeffs().buffer(),
                                   in4.coeffs().buffer(), in5.coeffs().buffer(),
                                   in6.coeffs().buffer(), in7.coeffs().buffer());
#endif

      /* some constness checks for the API */
      static_assert(std::is_const_v<std::remove_reference_t<decltype(in0)>>);
      static_assert(std::is_const_v<std::remove_reference_t<decltype(in0.coeffs())>>);
      static_assert(std::is_const_v<std::remove_reference_t<decltype(in0.coeffs().buffer())>>);
      static_assert(std::is_const_v<std::remove_reference_t<std::remove_reference_t<decltype(*in0.coeffs().buffer().current_device_ptr())>>>);

      /* assemble input array and submit kernel */
      //auto input_ptrs = std::apply([](auto... ins){ return std::array{(ins.coeffs.buffer().current_device_ptr())...}; });
      auto input_ptrs = std::array{in0.coeffs().buffer().current_device_ptr(), in1.coeffs().buffer().current_device_ptr(),
                                   in2.coeffs().buffer().current_device_ptr(), in3.coeffs().buffer().current_device_ptr(),
                                   in4.coeffs().buffer().current_device_ptr(), in5.coeffs().buffer().current_device_ptr(),
                                   in6.coeffs().buffer().current_device_ptr(), in7.coeffs().buffer().current_device_ptr()};

      auto coeffs_view = p.coeffs().current_view();
      auto rcoeffs_view = d.current_view();
      auto hgT_view = hgT.current_view();

      submit_compress_kernel(key, N, K, coeffs_view, rcoeffs_view, hgT_view,
                            tmp_scratch.device_ptr(), d_sumsq_scratch.device_ptr(), input_ptrs,
                            ttg::device::current_stream());

      /* wait for kernel and transfer sums back */
#ifndef TTG_ENABLE_HOST
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

      // Recur up
      if (key.level() > 0) {
        // will not return
#ifndef TTG_ENABLE_HOST
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
          std::cout << "At root of compressed tree fn " << i << ": total normsq is " << p.sum(i) + d_sumsq[i] << std::endl;
        }
#ifndef TTG_ENABLE_HOST
        co_await ttg::device::forward(
          // Send result to output tree
          ttg::device::send<out_terminal_id>(key, std::move(result)));
#else
        ttg::send<out_terminal_id>(key, std::move(result));
#endif
      }
  };
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> recur("recur");
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
  ttg::Edge<mra::Key<NDIM>, mra::Tensor<T,NDIM+1>> S("S");  // passes scaling functions down

  auto do_reconstruct = [&, N, K](const mra::Key<NDIM>& key,
                                  mra::FunctionsCompressedNode<T, NDIM>&& node,
                                  const mra::Tensor<T, NDIM+1>& from_parent) -> TASKTYPE {
    const std::size_t K = from_parent.dim(0);
    const std::size_t tmp_size = reconstruct_tmp_size<NDIM>(K)*N;
    auto tmp = std::make_unique_for_overwrite<T[]>(tmp_size);
    const auto& hg = functiondata.get_hg();
    auto tmp_scratch = ttg::make_scratch(tmp.get(), ttg::scope::Allocate, tmp_size);

    // Send empty interior node to result tree
    auto r_empty = mra::FunctionsReconstructedNode<T,NDIM>(key, N, K);
    r_empty.coeffs().current_view() = T(0.0);
    r_empty.set_all_leaf(false);

    /* populate the vector of r's
     * TODO: TTG/PaRSEC supports only a limited number of inputs so for higher dimensions
     *       we may have to consolidate the r's into a single buffer and pick them apart afterwards.
     *       That will require the ability to ref-count 'parent buffers'. */
    std::array<mra::FunctionsReconstructedNode<T,NDIM>, mra::Key<NDIM>::num_children()> r_arr;
    for (int i = 0; i < key.num_children(); ++i) {
      r_arr[i] = mra::FunctionsReconstructedNode<T,NDIM>(key, N, K);
      // no need to send this data to the device
      r_arr[i].coeffs().buffer().reset_scope(ttg::scope::Allocate);
    }

    // helper lambda to pick apart the std::array
    auto do_select = [&]<std::size_t... Is>(std::index_sequence<Is...>){
      return ttg::device::select(hg.buffer(), from_parent.buffer(),
                                 node.coeffs().buffer(), tmp_scratch,
                                 (r_arr[Is].coeffs().buffer())...);
    };
    /* select a device */
#ifndef TTG_ENABLE_HOST
    co_await do_select(std::make_index_sequence<mra::Key<NDIM>::num_children()>{});
#endif

    // helper lambda to pick apart the std::array
    auto assemble_tensor_ptrs = [&]<std::size_t... Is>(std::index_sequence<Is...>){
      return std::array{(r_arr[Is].coeffs().current_view().data())...};
    };
    auto r_ptrs = assemble_tensor_ptrs(std::make_index_sequence<mra::Key<NDIM>::num_children()>{});
    auto node_view = node.coeffs().current_view();
    auto hg_view = hg.current_view();
    auto from_parent_view = from_parent.current_view();
    submit_reconstruct_kernel(key, N, K, node_view, hg_view, from_parent_view,
                              r_ptrs, tmp_scratch.device_ptr(), ttg::device::current_stream());

    // forward() returns a vector that we can push into
#ifndef TTG_ENABLE_HOST
    auto sends = ttg::device::forward(ttg::device::send<1>(key, std::move(r_empty)));
    mra::KeyChildren<NDIM> children(key);
    for (auto it=children.begin(); it!=children.end(); ++it) {
        const mra::Key<NDIM> child= *it;
        mra::FunctionsReconstructedNode<T,NDIM>& r = r_arr[it.index()];
        r.key() = child;
        for (std::size_t i = 0; i < N; ++i) {
          r.is_leaf(i) = node.is_child_leaf(i, it.index());
        }
        if (r.is_all_leaf()) {
          sends.push_back(ttg::device::send<1>(child, std::move(r)));
        }
        else {
          sends.push_back(ttg::device::send<0>(child, std::move(r.coeffs())));
        }
    }
    co_await std::move(sends);
#else
    ttg::send<1>(key, std::move(r_empty));
    mra::KeyChildren<NDIM> children(key);
    for (auto it=children.begin(); it!=children.end(); ++it) {
        const mra::Key<NDIM> child= *it;
        mra::FunctionsReconstructedNode<T,NDIM>& r = r_arr[it.index()];
        r.key() = child;
        for (std::size_t i = 0; i < N; ++i) {
          r.is_leaf(i) = node.is_child_leaf(i, it.index());
        }
        if (r.is_all_leaf()) {
          ttg::send<1>(child, std::move(r));
        }
        else {
          ttg::send<0>(child, std::move(r.coeffs()));
        }
    }
#endif // TTG_ENABLE_HOST
  };


  auto s = ttg::make_tt<Space>(std::move(do_reconstruct), ttg::edges(in, S), ttg::edges(S, out), name, {"input", "s"}, {"s", "output"});

  if (ttg::default_execution_context().rank() == 0) {
    s->template in<1>()->send(mra::Key<NDIM>{0,{0}}, mra::Tensor<T,NDIM+1>(N, K, K, K)); // Prime the flow of scaling functions
  }

  return s;
}


static std::mutex printer_guard;
template <typename keyT, typename valueT>
auto make_printer(const ttg::Edge<keyT, valueT>& in, const char* str = "", const bool doprint=true) {
  auto func = [str,doprint](const keyT& key, const valueT& value) {
    if (doprint) {
      std::lock_guard<std::mutex> obolus(printer_guard);
      std::cout << str << " (" << key << "," << value << ")" << std::endl;
    }
  };
  return ttg::make_tt(func, ttg::edges(in), ttg::edges(), "printer", {"input"});
}

template<typename T, mra::Dimension NDIM>
auto make_add(ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> in1,
              ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> in2,
              ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> out,
              const T scalarA, const T scalarB, const mra::TensorView<int, 1> idxs, const size_t N, const size_t K) {
  auto func = [&, N, K, scalarA, scalarB](const mra::Key<NDIM>& key,
   const mra::FunctionsReconstructedNode<T, NDIM>& t1, const mra::FunctionsReconstructedNode<T, NDIM>& t2)
   -> TASKTYPE {

    auto out = mra::FunctionsReconstructedNode<T, NDIM>(key, N, K);

    #ifndef TTG_ENABLE_HOST
      co_await ttg::device::select(in1.coeffs().buffer(), in2.coeffs().buffer(),
                                    out.coeffs().buffer());
    #endif

    auto t1_view = t1.coeffs().current_view();
    auto t2_view = t2.coeffs().current_view();
    auto out_view = out.coeffs().current_view();

    submit_add_kernel<T, NDIM>(key, t1_view, t2_view, out_view, idxs,
                                scalarA, scalarB, N, K, ttg::device::current_stream());

    #ifndef TTG_ENABLE_HOST
        co_await ttg::device::forward(ttg::device::send<0>(key, std::move(out)));
    else
        ttg::send<0>(key, std::move(out));
    #endif
    };

  return ttg::make_tt<Space>(func, ttg::edges(in1, in2), ttg::edges(out), "add", {"in1", "in2"}, {"out"});
}

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
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> project_result, reconstruct_result, add_result;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> compress_result;

  // define N Gaussians
  std::vector<mra::Gaussian<T, NDIM>> gaussians;
  gaussians.reserve(N);
  T expnt = 30000.0;
  for (int i = 0; i < N; ++i) {
    //T expnt = 1500 + 1500*drand48();
    mra::Coordinate<T,NDIM> r;
    for (size_t d=0; d<NDIM; d++) {
      r[d] = T(-6.0) + T(12.0)*drand48();
    }
    std::cout << "Gaussian " << i << " expnt " << expnt << std::endl;
    gaussians.emplace_back(D, expnt, r);
  }

  int *idxs = new int[N];
  for (int i = 0; i < N; ++i) idxs[i] = i;
  auto idxsT = mra::TensorView<int, 1>(idxs, N);

  // put it into a buffer
  auto gauss_buffer = ttg::Buffer<mra::Gaussian<T, NDIM>>(gaussians.data(), N);
  auto db = ttg::Buffer<mra::Domain<NDIM>>(&D);
  auto start = make_start(project_control);
  auto project = make_project(db, gauss_buffer, N, K, functiondata, T(1e-6), project_control, project_result);
  auto compress = make_compress(N, K, functiondata, project_result, compress_result);
  auto reconstruct = make_reconstruct(N, K, functiondata, compress_result, reconstruct_result);
  auto add = make_add(reconstruct_result, reconstruct_result, add_result, T(1.0), T(1.0), idxsT, N, K);
  auto printer =   make_printer(project_result,    "projected    ", false);
  auto printer2 =  make_printer(compress_result,   "compressed   ", false);
  auto printer3 =  make_printer(reconstruct_result,"reconstructed", false);
  auto printer4 = make_printer(add_result, "added", false);

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

  // delete[] idxs;

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

  test<double, 3>(10, 10);

  ttg::finalize();
}
