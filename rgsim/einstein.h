#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include "types.h"
#include "metricoperator.h"

namespace rgsim {

void makeghat(met& gh, const met& g, const fvec& A, const func& h = constant(1.0));

struct Field {
    std::array<int,NS>    n;
    std::array<double,NS> q0, dq;
    std::vector<mat4>     gu;
    std::vector<double>   w;
    int size() const;
};

}
