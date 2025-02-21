#include <cassert>
#include <ttg.h>
#include "kernels/inner.h"
#include "tensor.h"

#include <madness/tensor/tensor.h>
#include <madness/world/print.h>


int main(int argc, char **argv) {

  ttg::initialize(argc, argv);

	constexpr const mra::size_type K = 3;

  madness::Tensor<double> nA(K, K, K), nB(K, K, K);

  mra::Tensor<double, 3> nodeA(K), nodeB(K);
  mra::Tensor<double, 4> nodeC(K);
  mra::TensorView<double, 3> nodeAv = nodeA.current_view();
  mra::TensorView<double, 3> nodeBv = nodeB.current_view();
  mra::TensorView<double, 4> nodeCv = nodeC.current_view();

	for (int i=0; i<K; ++i){
    for (int j=0; j<K; ++j){
      for (int k=0; k<K; ++k){
        nA(i,j,k) = 1+i+j+k;
        nB(i,j,k) = 1-i-j-k;
        nodeAv(i,j,k) = 1+i+j+k;
        nodeBv(i,j,k) = 1-i-j-k;
      }
	  }
  }

  mra::detail::inner(nodeAv, nodeBv, nodeCv, 2, 2);

  madness::Tensor<double> nC = madness::inner(nA, nB, 2, 2);
  std::cout << nodeCv << std::endl;

  madness::print(nC);

	ttg::execute();
  ttg::fence();

  ttg::finalize();
}
