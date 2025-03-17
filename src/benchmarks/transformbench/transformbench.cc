
#include "mra/misc/platform.h"
#include "mra/kernels/transform.h"
#include "mra/tensor/tensor.h"
#include "mra/tensor/tensorview.h"
#include "mra/misc/options.h"
#include "mra/misc/types.h"

#include <ttg.h>

using namespace mra; // lazy

template<typename T>
static
LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
GLOBALSCOPE void transform_kernel(TensorView<T, 3+1> A, TensorView<T, 2+1> B, TensorView<T, 3+1> C, TensorView<T, 3+1> workspace) {

  SHARED TensorView<T, 3> a, c, w;
  SHARED TensorView<T, 2> b;
  if (is_team_lead()) {
    a = A(blockIdx.x);
    b = B(blockIdx.x);
    c = C(blockIdx.x);
    w = workspace(blockIdx.x);
  }
  SYNCTHREADS();

  transform(a, b, c, w.data());
#if 0
  T* pc = c.data();
  const T* pb = b.data();
  const T* pa = a.data();
  const size_type dimj = b.dim(1);
  size_type dimi = 1;
  for (size_type n=1; n<a.ndim(); ++n) dimi *= dimj;
  for (int i = 0; i < nrep; i++) {
    mTxmq(dimi, dimj, dimj, pc, pa, pb);
  }
#endif // 0
}

template<typename T>
static void submit_transform_bench(int N, int K, TensorView<T, 3+1> A, TensorView<T, 2+1> B, TensorView<T, 3+1> C, TensorView<T, 3+1> workspace) {
  Dim3 thread_dims = max_thread_dims(K);
  CALL_KERNEL(transform_kernel, N, thread_dims, 0, ttg::device::current_stream(), (A, B, C, workspace));
  checkSubmit();
}

int main(int argc, char **argv) {

  auto opt = mra::OptionParser(argc, argv);
  int nreps = opt.parse("-n", 100);
  int nblocks = opt.parse("-N", 10);
  int K = opt.parse("-K", 10);
  ttg::initialize(argc, argv);
  ttg::Edge<int, void> e; // control edge
  auto start = ttg::make_tt([&](){
    for (int i = 0; i < nreps; i++) {
      ttg::sendk<0>(i);
    }
  }, ttg::edges(), ttg::edges(e));
  auto tt = ttg::make_tt<Space>([&](const int& key) -> TASKTYPE {
    auto a = Tensor<double, 3+1>(nblocks, K, K, K); // nblocks x size^3 elements
    auto b = Tensor<double, 2+1>(nblocks, K, K); // size^2 elements
    auto c = Tensor<double, 3+1>(nblocks, K, K, K); // size^3 elements
    auto workspace = Tensor<double, 3+1>(nblocks, K, K, K); // size^3 elements
#ifndef MRA_ENABLE_HOST
    co_await ttg::device::select(a.buffer(), b.buffer(), c.buffer(), workspace.buffer());
#endif // MRA_ENABLE_HOST
    submit_transform_bench(nblocks, K, a.current_view(), b.current_view(), c.current_view(), workspace.current_view());
  }, ttg::edges(e), ttg::edges());

  auto connected = ttg::make_graph_executable(start.get());
  assert(connected);


  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;
  beg = std::chrono::high_resolution_clock::now();
  start->invoke(); // kick off
  ttg::execute();
  ttg::fence();
  end = std::chrono::high_resolution_clock::now();

  auto time = (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()) / 1000;
  std::cout << "TTG Execution Time (milliseconds) : "
            << time
            << "; Flops: " << nreps * K * K * K * K * 3 * nblocks
            << "; Gflop/s: " << (1e-6 * nreps * K * K * K * K * 3 * nblocks) / time
            << std::endl;
  ttg::finalize();
}