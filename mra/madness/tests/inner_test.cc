#include <cassert>
#include <ttg.h>
#include "kernels/inner.h"
#include "tensor.h"

int main(int argc, char **argv) {

  ttg::initialize(argc, argv);

	constexpr const mra::size_type K = 3;

  mra::Tensor<double, 2> nodeA(K), nodeB(K), nodeC(K);
  mra::TensorView<double, 2> nodeAv = nodeA.current_view();
  mra::TensorView<double, 2> nodeBv = nodeB.current_view();
  mra::TensorView<double, 2> nodeCv = nodeC.current_view();

	for (int i=0; i<nodeA.size(); ++i){
    for (int j=0; j<nodeA.size(); ++j){
      nodeAv(i,j) = 1+i+j;
      nodeBv(i,j) = 1-i-j;
      nodeCv(i,j) = 100.0;
    }
	}

  mra::detail::inner(nodeAv, nodeBv, nodeCv);

  std::cout << nodeC << std::endl;

	ttg::execute();
  ttg::fence();

  ttg::finalize();
}
