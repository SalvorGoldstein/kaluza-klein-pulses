#pragma once

#include <functional>
#include "field.h"

namespace rgsim {

// Any system of the form   d_t A = dA,  d_t V = dV   where the right-hand side is a function of (A, V, t).
// systemA is one of them. The function must write its result in dA and dV (same shape as A and V).
typedef std::function<void(const AGrid& A, const AGrid& V, double t, AGrid& dA, AGrid& dV)> RhsA;

// One Runge-Kutta 4 step of size dt, starting at time t.  A and V are replaced by their values at t + dt.
void rk4A(const RhsA& rhs, AGrid& A, AGrid& V, double t, double dt);

}