#include "einstein.h"

namespace rgsim {

void makeghat(met& gh, const met& g, const fvec& A, const func& h)
{
    for (int m = 0; m < D; ++m) {
        for (int n = 0; n < D; ++n)
            gh[m][n] = g[m][n] - h * A[m] * A[n];
        gh[m][D] = constant(0.0) - h * A[m];
        gh[D][m] = gh[m][D];
    }
    gh[D][D] = constant(0.0) - h;
}

int Field::size() const { int s = 1; for (int v : n) s *= v; return s; }

}
