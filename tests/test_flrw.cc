// Test of a time-dependent background: flat de Sitter (FLRW) in comoving Cartesian coordinates.
// Maxwell is conformally invariant: with the dilaton coupling switched off (a ~ 0) a pulse in ds^2 = a(t)^2 (c^2 d eta^2 - dx^2),
// d eta = dt / a, must be the SAME as in flat space at the conformal time eta(t) = (1 - exp(-H t)) / H.  We check that.
// usage: ./test_flrw <H in 1/s> <tfac>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include "rgsim/rgsim.h"
#include "example_metric.h"
#include "example_potential.h"
#include "helpers.h"
using namespace std; using namespace rgsim;

static AGrid run(bool flrw, double H, double tEnd, int nr, double size)
{
    flrw_H = H;
    const met g = makeMetric(flrw ? flrw_cartesian : minkowski_cartesian), ginv = inverseMet(g);
    const christ gam = makeChristoffel(g, ginv);
    metricIsStatic = !flrw; metricIsHomogeneous = flrw; resetMetricCache();
    const func Phi = constant(1.0);
    const double dr = size / nr;
    const PointGrid pts = makegrid({nr,nr,nr}, {-0.5*size, -0.5*size, -0.5*size}, {dr,dr,dr});
    const int bc[3][2] = {{BC_PERIODIC,BC_PERIODIC},{BC_PERIODIC,BC_PERIODIC},{BC_PERIODIC,BC_PERIODIC}};
    pc_amp = 0.05; pc_x = 0.37*dr; pc_y = 0.21*dr; pc_z = 0.13*dr; pc_w = size / 10.0;
    AGrid A, V; fillPotential(pts, initialPotentialCartesian(wavepacket_cartesian, +1.0), A, V);
    SGrid phi = makeScalarGrid(pts), Pi = makeScalarGrid(pts);
    const RhsAphi rhs = [&](const AGrid& a, const AGrid& v, const SGrid& p, const SGrid& q, double tt,
                            AGrid& da, AGrid& dv, SGrid& dp, SGrid& dq) {
        systemAphi(pts, a, v, p, q, bc, gam, g, ginv, Phi, tt, da, dv, dp, dq); };
    const double dt = 0.3 * dr / c;
    const int steps = int(tEnd / dt + 0.5);
    const double dtt = tEnd / steps;
    double t = 0;
    for (int s = 0; s < steps; s++) { rk4Aphi(rhs, A, V, phi, Pi, t, dtt); t += dtt; }
    return A;
}

int main(int argc, char** argv)
{
    double H = argc > 1 ? atof(argv[1]) : 2e6, tfac = argc > 2 ? atof(argv[2]) : 1.0;
    dilatonCoupling = 1e-9;                         // dilaton decoupled: pure Maxwell
    const int nr = 32; const double size = 100.0;
    const double tEnd = tfac * 0.5 * size / c;
    const double eta = (1.0 - exp(-H * tEnd)) / H;   // conformal time
    cout << "t = " << tEnd << " s,  a(t) = " << exp(H * tEnd) << ",  conformal time eta = " << eta << " s" << endl;
    AGrid F = run(true, H, tEnd, nr, size);
    AGrid M = run(false, 0.0, eta, nr, size);
    double num = 0, den = 0, mx = 0;
    for (int i = 0; i < nr; i++) for (int j = 0; j < nr; j++) for (int k = 0; k < nr; k++)
        for (int m = 1; m < 4; m++) {
            double a = F[i][j][k].getvalue()[m], b = M[i][j][k].getvalue()[m];
            num += (a-b)*(a-b); den += b*b; mx = max(mx, fabs(b));
        }
    cout << "max|A| flat = " << mx << ",  relative L2 difference FLRW vs flat(eta) = " << sqrt(num/den) << endl;
    return 0;
}
