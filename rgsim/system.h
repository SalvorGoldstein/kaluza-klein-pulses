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

// ---------------------------------------------------------------------------------------------
// DILATON (dynamical scalar)  --  see README "The dilaton".
// A scalar field phi (with Pi = d_t phi) is evolved together with A on the fixed metric. Action (signature +,-,-,-):
//      L = sqrt|g| [ 1/2 d_mu phi d^mu phi  -  1/4 e^{-a phi} h F_{mu nu} F^{mu nu} ]
//      =>  nabla_mu( h e^{-a phi} F^{mu nu} ) = 0 ,      box phi = (a/4) h e^{-a phi} F_{mu nu} F^{mu nu}
// a = sqrt(3) is the Kaluza-Klein value (5D vacuum Einstein -> 4D Einstein-Maxwell-dilaton).
// This is R_55 = 0 of the 5D equations: it removes the  A_mu * R_55  term of the frozen-scalar equation (the term
// that makes the old systemA blow up when E^2 > B^2) and conserves a positive energy.
// ---------------------------------------------------------------------------------------------
using SGrid = std::vector<std::vector<std::vector<double>>>;
extern double dilatonCoupling;          // a, default sqrt(3)

// Same as systemA, but A, V AND the dilaton (phi, Pi) are evolved. The "phihat" term is not used.
//   dphi = Pi,  dPi = d_t d_t phi          (same boundary conditions as A)
void systemAphi(const PointGrid& pts, const AGrid& A, const AGrid& V, const SGrid& phi, const SGrid& Pi,
                const int bc[3][2],
                const christ& gam, const met& g, const met& ginv, const func& Phi, double t,
                AGrid& dA, AGrid& dV, SGrid& dphi, SGrid& dPi);

// The metric data (inverse metric, Christoffel symbols, ...) is computed once per grid point and kept.
// This is valid when the metric does not depend on t (true for all the examples).
// If it does, set metricIsStatic = false (the data is then rebuilt at every call).
// Call resetMetricCache() if you change the metric, Phi, phihat or the grid.
extern bool metricIsStatic;
// If the metric data is the same at every grid point (spatially homogeneous, e.g. flat FLRW), set this too: the data is then
// computed once per call instead of once per grid point, so a time-dependent metric (metricIsStatic = false) stays cheap.
extern bool metricIsHomogeneous;
void resetMetricCache();

}
