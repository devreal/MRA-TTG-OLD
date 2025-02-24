#ifndef MRA_TASKS_COMMON_H_INCL
#define MRA_TASKS_COMMON_H_INCL

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
#ifndef TASKTYPE
#define TASKTYPE void
constexpr const ttg::ExecutionSpace Space = ttg::ExecutionSpace::Host;
#endif
#elif defined(MRA_ENABLE_CUDA)
#ifndef TASKTYPE
#define TASKTYPE ttg::device::Task
constexpr const ttg::ExecutionSpace Space = ttg::ExecutionSpace::CUDA;
#endif
#elif defined(MRA_ENABLE_HIP)
#ifndef TASKTYPE
#define TASKTYPE ttg::device::Task
constexpr const ttg::ExecutionSpace Space = ttg::ExecutionSpace::HIP;
#endif
#endif

namespace mra{

	template <mra::Dimension NDIM>
	auto make_start(const ttg::Edge<mra::Key<NDIM>, void>& ctl) {
			auto func = [](const mra::Key<NDIM>& key) { ttg::sendk<0>(key); };
			return ttg::make_tt<mra::Key<NDIM>>(func, ttg::edges(), edges(ctl), "start", {}, {"control"});
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

} // namespace mra

#endif // MRA_TASKS_COMMON_H_INCL
