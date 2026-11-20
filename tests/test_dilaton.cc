// Test: flat space, periodic box, one pulse. Compares the old equation (frozen scalar) with the dilaton equation.
// usage: ./test_dilaton <amp> <mode: 0 = old, 1 = dilaton> [a] [tfac]
// Prints t, max|A|, max|phi|, total energy. In dilaton mode the energy must stay constant (periodic box).
#include <iostream>
#include <cmath>
#include <cstdlib>
#include "rgsim/rgsim.h"
#include "example_metric.h"
#include "example_potential.h"
#include "helpers.h"
using namespace std; using namespace rgsim;

int main(int argc, char** argv)
{
    double amp = argc > 1 ? atof(argv[1]) : 1.0;
    int mode   = argc > 2 ? atoi(argv[2]) : 1;
    if (argc > 3) dilatonCoupling = atof(argv[3]);
    double tfac = argc > 4 ? atof(argv[4]) : 3.0;
    const int nr = argc > 5 ? atoi(argv[5]) : 32; const double size = 100.0;
    const met g = makeMetric(minkowski_cartesian), ginv = inverseMet(g);
    const christ gam = makeChristoffel(g, ginv);
    const func Phi = constant(1.0), phihat = constant(0.0);
    const double dr = size / nr;
    const PointGrid pts = makegrid({nr,nr,nr}, {-0.5*size, -0.5*size, -0.5*size}, {dr,dr,dr});
    const int bc[3][2] = {{BC_PERIODIC,BC_PERIODIC},{BC_PERIODIC,BC_PERIODIC},{BC_PERIODIC,BC_PERIODIC}};
    pc_amp = amp; pc_x = 0.37*dr; pc_y = 0.21*dr; pc_z = 0.13*dr; pc_w = size / 10.0;
    AGrid A, V; fillPotential(pts, initialPotentialCartesian(wavepacket_cartesian, +1.0), A, V);
    SGrid phi(nr, vector<vector<double>>(nr, vector<double>(nr, 0.0))), Pi = phi;

    const double dt = (argc > 6 ? atof(argv[6]) : 0.4) * dr / c;
    const int steps = int(tfac * size / c / dt);
    auto energy = [&]() {
        double a = dilatonCoupling, E = 0;
        for (int i = 0; i < nr; i++) for (int j = 0; j < nr; j++) for (int k = 0; k < nr; k++) {
            auto d = [&](int ax, int comp) {      // d_ax A_comp, centred, periodic
                int p[3] = {i,j,k}, m[3] = {i,j,k};
                p[ax] = (p[ax]+1)%nr; m[ax] = (m[ax]+nr-1)%nr;
                return (A[p[0]][p[1]][p[2]].getvalue()[comp] - A[m[0]][m[1]][m[2]].getvalue()[comp]) / (2*dr);
            };
            auto dphi = [&](int ax) {
                int p[3] = {i,j,k}, m[3] = {i,j,k};
                p[ax] = (p[ax]+1)%nr; m[ax] = (m[ax]+nr-1)%nr;
                return (phi[p[0]][p[1]][p[2]] - phi[m[0]][m[1]][m[2]]) / (2*dr);
            };
            double cx = d(1,3)-d(2,2), cy = d(2,1)-d(0,3), cz = d(0,2)-d(1,1);   // curl (A_1..A_3 = comps 1..3)
            double Bx = d(1,3)-d(2,2);
            (void)Bx;
            // comps: A[...][comp], comp 1,2,3 = x,y,z ; d(ax,comp) with ax 0,1,2 = x,y,z
            cx = d(1,3) - d(2,2); cy = d(2,1) - d(0,3); cz = d(0,2) - d(1,1);
            double p0 = phi[i][j][k];
            double w = (mode ? exp(-a * p0) : 1.0);
            auto v = V[i][j][k].getvalue();
            double q = Pi[i][j][k];
            E += q*q/(2*c*c) + 0.5*(dphi(0)*dphi(0)+dphi(1)*dphi(1)+dphi(2)*dphi(2))
               + w*(v[1]*v[1]+v[2]*v[2]+v[3]*v[3])/(2*c*c) + 0.5*w*(cx*cx+cy*cy+cz*cz);
        }
        return E * dr*dr*dr;
    };

    double t = 0, E0 = energy();
    const RhsA rhsOld = [&](const AGrid& a, const AGrid& v, double tt, AGrid& da, AGrid& dv) {
        systemA(pts, a, v, bc, gam, g, ginv, Phi, phihat, tt, da, dv); };
    const RhsAphi rhsNew = [&](const AGrid& a, const AGrid& v, const SGrid& p, const SGrid& q, double tt,
                               AGrid& da, AGrid& dv, SGrid& dp, SGrid& dq) {
        systemAphi(pts, a, v, p, q, bc, gam, g, ginv, Phi, tt, da, dv, dp, dq); };
    const int every = max(1, steps / 15);
    for (int s = 0; s <= steps; s++) {
        if (s % every == 0 || s == steps) {
            double mA = 0, mp = 0;
            for (int i = 0; i < nr; i++) for (int j = 0; j < nr; j++) for (int k = 0; k < nr; k++) {
                auto a = A[i][j][k].getvalue();
                mA = max(mA, max(fabs(a[1]), max(fabs(a[2]), fabs(a[3]))));
                mp = max(mp, fabs(phi[i][j][k]));
                if (!std::isfinite(a[1]) || !std::isfinite(a[2])) mA = INFINITY;
            }
            printf("t=%.3e  max|A|=%.4e  max|phi|=%.4e  E/E0=%.6f\n", t, mA, mp, energy()/E0);
            if (!std::isfinite(mA)) { printf("BLOW UP\n"); return 1; }
        }
        if (s < steps) {
            if (mode) rk4Aphi(rhsNew, A, V, phi, Pi, t, dt); else rk4A(rhsOld, A, V, t, dt);
            t += dt;
        }
    }
    return 0;
}
