#ifndef MRA_TASKS_PROJECT_H
#define MRA_TASKS_PROJECT_H

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

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

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

namespace mra{
	template<typename FnT, typename T, mra::Dimension NDIM>
	auto make_project(
		const ttg::Buffer<mra::Domain<NDIM>>& db,
		const ttg::Buffer<FnT>& fb,
		std::size_t N,
		std::size_t K,
		int max_level,
		const mra::FunctionData<T, NDIM>& functiondata,
		const T thresh, /// should be scalar value not complex
		ttg::Edge<mra::Key<NDIM>, void> control,
		ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> result,
		const char *name = "project")
	{
		/* create a non-owning buffer for domain and capture it */
		auto fn = [&, N, K, max_level, thresh, gl = mra::GLbuffer<T>()]
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

				if (max_level > 0){
					if (!all_initial_level && result.key().level() < max_level) { // && pass in max_level
						std::vector<mra::Key<NDIM>> bcast_keys;
						for (auto child : children(key)) bcast_keys.push_back(child);
#ifndef MRA_ENABLE_HOST
          	outputs.push_back(ttg::device::broadcastk<0>(std::move(bcast_keys)));
#else
          	ttg::broadcastk<0>(bcast_keys);
#endif
					}
					if (key.level() == max_level) {
						result.set_all_leaf(true);
					}
					else {
						result.set_all_leaf(false);
					}
				}
				else {
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
} // namespace mra

#endif // MRA_TASKS_PROJECT_H
