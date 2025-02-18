#include "platform.h"
#include "transform.h"
#include "tensor.h"
#include "tensorview.h"

#include <ttg.h>

using namespace mra; // lazy

static
LAUNCH_BOUNDS(1024)
GLOBALSCOPE void mTxm_kernel(int nrep, TensorView<double, 3+1> A, TensorView<double, 2+1> B, TensorView<double, 3+1> C) {

  SHARED TensorView<double, 3> a, b, c;
  if (is_team_lead()) {
    a = A(blockIdx.x);
    b = B(blockIdx.x);
    c = C(blockIdx.x);
  }
  SYNCTHREADS();
  T* pc = c.data();
  const T* pb = b.data();
  const T* pa = a.data();
  const size_type dimj = b.dim(1);
  size_type dimi = 1;
  for (size_type n=1; n<a.ndim(); ++n) dimi *= dimj;
  for (int i = 0; i < nrep; i++) {
    mTxmq(dimi, dimj, dimj, pc, pa, pb);
  }

}

static submit_mTxm_bench(int nblocks, int nrep, TensorView<double, 3+1>& A, TensorView<double, 2+1>& B, TensorView<double, 3+1>& C) {
  Dim3 thread_dims = max_thread_dims(a.dim(0));
  CALL_KERNEL(mTxm_kernel, nblocks, thread_dims, 0, stream, (nrep, A, B, C));
  checkSubmit();
}

int main(int argc, char **argv) {

  auto opt = mra::OptionParser(argc, argv);
  int nreps = opt.parse("-n", 100);
  int nblocks = opt.parse("-b", 1);
  int size = opt.parse("-s", 10);
  ttg::initialize(argc, argv);
  ttg::Edge<void, void> e; // control edge
  auto tt = ttg::make_tt([&]() -> ttg::Task {
    auto a = Tensor<double, 3+1>(nblocks, size); // nblocks x size^3 elements
    auto b = Tensor<double, 2+1>(nblocks, size); // size^2 elements
    auto c = Tensor<double, 3+1>(nblocks, size); // size^3 elements
    co_await ttg::device::select(a.buffer(), b.buffer(), c.buffer());
    submit_mTxm_bench(nblocks, nreps, a.current_view(), b.current_view(), c.current_view());
  }, ttg::edges(e), ttg::edges());

  auto connected = ttg::make_graph_executable(tt.get());
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
            << "; Gflops: " << (1e-6 * nreps * size * size * size * size * nblocks) / time
            << std::endl;
  ttg::finalize();
}