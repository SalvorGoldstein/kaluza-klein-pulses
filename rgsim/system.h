#pragma once

#include <array>
#include <vector>
#include "types.h"
#include "field.h"

namespace rgsim {

// Codes for the boundary conditions
const int BC_DIRICHLET = 0;   // the value stays fixed
const int BC_NEUMANN   = 1;   // the value follows the point just inside (zero slope)
const int BC_PERIODIC  = 2;   // the grid wraps around
const int BC_OUTGOING  = 3;   // the wave leaves the box

// Time derivative of the vector potential A_mu on a fixed 4D metric.
//
//   pts          coordinates of the grid points
//   A            A_mu on the grid         (4 numbers per point: A_0, A_1, A_2, A_3)
//   V            d_t A_mu on the grid
//   bc[axis][0]  condition on the lower face of the axis, bc[axis][1] on the upper face
//                (axis 0, 1, 2 = the three space coordinates). Periodic must be on both faces.
//   gam          Christoffel symbols of the 4D metric
//   g, ginv      the 4D metric and its inverse
//   Phi, phihat  the internal scale (h = Phi*Phi) and the dilaton
//   t            time
//   dA, dV       RESULT:  dA = d_t A  (= V)    dV = d_t d_t A
void systemA(const PointGrid& pts, const AGrid& A, const AGrid& V,
             const int bc[3][2],
             const christ& gam, const met& g, const met& ginv,
             const func& Phi, const func& phihat, double t,
             AGrid& dA, AGrid& dV);

// The metric data (inverse metric, Christoffel symbols, ...) is computed once per grid point and kept.
// This is valid when the metric does not depend on t (true for all the examples).
// If it does, set metricIsStatic = false (the data is then rebuilt at every call).
// Call resetMetricCache() if you change the metric, Phi, phihat or the grid.
extern bool metricIsStatic;
void resetMetricCache();

}
