#include <cassert>
#include <ttg.h>
#include "kernels/gaxpy.h"

int main(int argc, char **argv) {

    ttg::initialize(argc, argv);

    double scalarA = 1.0, scalarB = -1.0;

    double *nodeA = new double[6];
    double *nodeB = new double[6];
    double *nodeR = new double[6];

    for (int i=0; i<6; ++i){
            nodeA[i] = i;
            nodeB[i] = i;
            nodeR[i] = 100.0;
        }

    mra::detail::gaxpy_kernel_impl<double, 2>(nodeA, nodeB, nodeR, scalarA, scalarB, 3);

    for (int i = 0; i < 6; ++i) {
        assert(nodeR[i] == 0.0);
    }

  ttg::finalize();
}
