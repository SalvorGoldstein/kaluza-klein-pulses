#pragma once
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>
#include <array>
#include <vector>
#include "types.h"
#include "funcoperator.h"
#include "Geodiff.h"
#include "einstein.h"

// Time derivative of the 5D potential A_mu on a fixed 4D metric.
//
// State Q (NB = 7 blocks of P numbers, P = number of grid points):
//     blocks 0..2 : A_1, A_2, A_3          blocks 3..5 : v_j = d_t A_j
// A_0 has its own block 6 (7 blocks in total, NB).  evolveA0 selects what is done with it:
//   0 (default): A_0 is NOT evolved: it is the prescribed function A0(t, x) (default 0), as before.
//   1          : A_0 is evolved with the Lorenz-gauge condition  d_mu(sqrt(-g) g^{mu nu} A_nu) = 0  solved for d_t A_0
//                (first order in time, no new solver) but it is PASSIVE: the equation of A_j still uses the prescribed
//                A0 (e.g. only the black hole's static field), so A_0 shows the response to the pulse and cannot
//                destabilise anything.  Always stable.
//   2          : same A_0, but A_0 feeds back into the equation of A_j (F_{0j} = d_t A_j - d_j A_0 uses the evolved A_0).
//                Consistent only while the term A_mu*kA of the equation is small; with a strong background field
//                (kA large) A_0 is not a free gauge any more (mu = 0 constraint) and this mode becomes unstable.
// The mu = 0 field equation (a constraint without second time derivative) is not evolved in any mode.

namespace rgsim {

constexpr int NB = 7;            // number of blocks of the state: A_1..A_3, d_t A_1..d_t A_3, A_0
extern int  evolveA0;            // 0: A_0 prescribed;  1: A_0 evolved, passive;  2: A_0 evolved, feeds back (see above)
extern func A0;                  // A_0(t, x): prescribed if !evolveA0, else only its initial value (block 6)

// ---------------------------------------------------------------- the grid
// Three axes, along the coordinates x[1], x[2], x[3].  (x[0] is time, x[4] the extra dimension.)
struct Grid {
    std::array<int,NS>    n;     // number of points on each axis
    std::array<double,NS> q0;    // coordinate of the first point on each axis
    std::array<double,NS> dq;    // spacing on each axis
    std::array<bool,NS>   periodic = {false, false, false};   // true: the n points cover a whole period (wraps around)
                                                              // false: the two end faces use the outgoing-wave condition
    // Optional static background solution (6 blocks like the state).  If set, the outgoing-wave condition on the
    // faces acts on (state - background) only, so the background itself is left alone there.
    const std::vector<double>* bg = nullptr;

    int size() const;                                          // total number of points
    int index(int i0, int i1, int i2) const;                   // position in a flat list

    std::array<dual,DIM> coords(int i0, int i1, int i2, double t = 0.0) const;   // coordinates of the point (i0,i1,i2) at time t
};

// State at t = 0 from the functions A[1..3](x): A_j and d_t A_j; A_0 from the function A0.
state makeInitialState(const Grid& G, const fvec& A);

// ---------------------------------------------------------------- fixed ingredients (set by main)
extern Grid grid;
extern int  P;
extern met  g;                  // 4x4 base metric
extern func Phi, phihat;        // Phi (h = Phi^2) and the 5D dilaton

// ---------------------------------------------------------------- metric data, computed once and cached
// Everything the equation needs from the metric.  It is built through slow autodiff chains, so it is stored per
// grid point, only along the axes it depends on (and rebuilt at each new time only if it depends on time).
struct PointData {
    double ginv[4][4];       // inverse 4D metric
    double Gam[4][4][4];     // Christoffel symbols Gam^l_{mu nu}
    double Gc[4];            // Gc^l = g^{ar} Gam^l_{ar}
    double U[4][4];          // sqrt(-det g) * g^{mu nu}  (for the Lorenz gauge in conservation form)
    double h, Phil;          // h = Phi^2 and Phi
    double box;              // sum g^{ab} d_a d_b phihat
    double dA0[4];           // d_a A_0
    double ddA0[4][4];       // d_a d_b A_0
};
extern std::vector<PointData> pointData;
extern int    pdGridSize;
extern bool   pdBuilt, pdAnalysed, pdTimeDep;
extern std::array<bool,3> pdDep;   // does the data depend on x1, x2, x3 ?
extern double pdTime;
extern met    pdInvg;

// data at grid point p (use this, not pointData[p])
const PointData& pointAt(int p);

// true if g, Phi, phihat or A0 has a non-zero derivative along one of the coordinates kFirst..kLast,
// tested at the grid corners and centre
bool dependsOn(const Grid& G, double t, int kFirst, int kLast);

void ensurePointData(double t = 0.0);

// ---------------------------------------------------------------- the equation of motion
// dQ = dQ/dt at time t.
void rhsField(const state& Q, double t, state& dQ);

}
