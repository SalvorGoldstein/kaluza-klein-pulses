// main_blackhole: vector potential A_mu on a fixed KERR-NEWMAN black hole, with the 5D-reduced equation (rgsim: systemA + rk4A).
// A pulse starts at a random place around the black hole and spreads out.
// Option: the black hole's own electromagnetic field (Coulomb + magnetic part from its charge and spin) is included as a
// static background, and the pulse is evolved on top of it.  Plots show the pulse only (field minus background-only run).
// The metric is only defined for r > r_s, so the grid starts just above the horizon and the
// inner r-face acts as the limit r -> r_s (there light is frozen in these coordinates); the outer face lets waves leave.
// The grid covers the full 360 degrees of phi (periodic) and a band of theta around the equator
// (the poles are singular in these coordinates).  The angular resolution follows the radial one:
// square cells in the middle of the radial range, and the same step in theta as in phi.
//
// How the library is used here:
//   * the metric g, its inverse and the Christoffel symbols are built once (functions of the coordinates),
//   * the unknowns live on a PointGrid / AGrid:  A (A_mu) and V (d_t A_mu), 4 numbers per grid point,
//   * systemA gives the time derivative of (A, V) and rk4A advances (A, V) by one step.
//   * A_0 is NOT evolved by systemA (its rate is 0): it keeps the value it has at t = 0.
//     For the black hole's own field this is exactly what we want (A_t is static).
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <random>
#include <array>

#include "rgsim/rgsim.h"
#include "example_metric.h"
#include "example_potential.h"
#include "helpers.h"

using namespace std; using namespace rgsim;

int main()
{
    int ipulse, nr, ikn;
    double j1, tfac;

    cout << "which initial pulse? (0 = gaussian, 1 = wavepacket, 2 = soliton, 3 = chirp): ";
    cin >> ipulse;
    cout << "grid end (in units of r_s, for example 10): ";
    cin >> j1;
    cout << "number of radial points: ";
    cin >> nr;
    cout << "duration (1 = light crosses the radial range once, 3 = three times, ...): ";
    cin >> tfac;
    cout << "include the black hole's own electromagnetic field? (0 = no, 1 = yes, Kerr-Newman charge and spin): ";
    cin >> ikn;
    cout << "perfect!" << endl;

    if (ipulse < 0 || ipulse > 3 || nr < 5 || ikn < 0 || ikn > 1 || j1 <= 1.5 || tfac <= 0.0) {
        cerr << "invalid input: need 0 <= pulse <= 3, nr >= 5, end > 1.5 r_s, duration > 0" << endl;
        return 1;
    }

    func (*profile[4])(int) = { pulse_cartesian, wavepacket_cartesian, soliton_cartesian, chirp_cartesian };

    // ---- fixed ingredients: black-hole metric (only valid for r > r_s), inverse metric, Christoffel symbols ----
    const met g    = makeMetric(kerr_newman);
    const met ginv = inverseMet(g);
    const christ gam = makeChristoffel(g, ginv);
    const func Phi    = constant(1.0);                 // h = Phi^2
    const func phihat = constant(0.0);                 // no dilaton: pure wave test

    // ---- the grid ----
    const double r0 = 1.01 * r_s;                        // just above the horizon: the r = r_s limit
    const double r1 = j1 * r_s;
    const double R  = r1 - r0;
    const double dr = R / (nr - 1);

    // phi: full circle, nph points with square cells at the middle radius
    const double rmid = 0.5 * (r0 + r1);
    const int    nph  = max(48, int(lround(2.0 * pi * rmid / dr)));
    const double dph  = 2.0 * pi / nph;
    // theta: same step as phi, a band of about +-0.5 rad around the equator (odd number of rows)
    const int    hth  = max(2, int(lround(0.5 / dph)));  // equator = row hth
    const int    nth  = 2 * hth + 1;
    const double dth  = dph;
    const double th0  = 0.5 * pi - hth * dth;

    const PointGrid pts = makegrid({nr, nth, nph}, {r0, th0, 0.0}, {dr, dth, dph});   // theta: a band around the equator
    const int P = nr * nth * nph;
    cout << "grid: " << nr << " x " << nth << " x " << nph << " = " << P << " points" << endl;

    // boundary conditions: r and theta faces let the waves leave (at the r_s limit the light speed is 0, so that
    // face just stays put), phi is PERIODIC (full 360 degrees)
    const int bc[3][2] = { { BC_OUTGOING, BC_OUTGOING },
                           { BC_OUTGOING, BC_OUTGOING },
                           { BC_PERIODIC, BC_PERIODIC } };

    // ---- random pulse centre: radius in the middle 60 % of the range, phi anywhere,
    //      theta close to the equator (so the equatorial plane we draw goes through the pulse) ----
    random_device rd;
    mt19937 rng(rd());
    uniform_real_distribution<double> runif(0.0, 1.0);

    const double rc  = r0 + (0.2 + 0.6 * runif(rng)) * R;
    const double thc = 0.5 * pi + (2.0 * runif(rng) - 1.0) * 0.3 * hth * dth;
    const double phc = 2.0 * pi * runif(rng);
    cout << "random pulse centre: r = " << rc / r_s << " r_s,  th = " << thc
         << " rad,  ph = " << phc << " rad" << endl;

    pc_spherical = true;                                  // the grid is (r, theta, phi)
    pc_x = rc * sin(thc) * cos(phc);                      // centre of the pulse in Cartesian coordinates
    pc_y = rc * sin(thc) * sin(phc);
    pc_z = rc * cos(thc);
    pc_w = 0.1 * R;                                       // width of the pulse

    // ---- the unknowns: A_mu(t=0) and d_t A_mu(t=0) from example_potential, a round blob around the centre ----
    const fvec Apulse = initialPotentialSpherical(profile[ipulse], +1.0);
    AGrid A, V;
    fillPotential(pts, Apulse, A, V);

    // ---- the black hole's own field (Kerr-Newman): static A_t(x) in component 0 (kept fixed by systemA), plus the
    //      static magnetic part A_ph in the unknowns.  The plots below show the PERTURBATION (field minus the
    //      background-only run), so you see the pulse, not the Coulomb field. ----
    AGrid Ab, Vb;                                         // the background alone, evolved side by side (see below)
    if (ikn == 1) {
        fvec Abg(4, constant(0.0));
        Abg[0] = kn_At();
        for (int i = 1; i < 4; ++i) Abg[i] = kn_potential(i);
        fillPotential(pts, Abg, Ab, Vb);
        addInto(A, Ab);                                   // pulse + background
        addInto(V, Vb);
        cout << "own field of the black hole included (charge length r_Q = " << sqrt(r_Q2) / r_s << " r_s, spin a = "
             << a_spin / r_s << " r_s)" << endl;
    }
    // Perturbation = field minus the background-only run.  The background alone is evolved with the same scheme, so the
    // small drift of a static solution caused by the finite differences cancels in the difference (valid in the
    // linear regime, which is where the pulse lives).
    auto pert = [&](int i0, int i1, int i2, int comp) {
        double v = A[i0][i1][i2].getvalue()[comp];
        if (ikn == 1) v -= Ab[i0][i1][i2].getvalue()[comp];
        return v;
    };

    // the equation of motion, in the form rk4A wants
    const RhsA rhs = [&](const AGrid& a, const AGrid& v, double tt, AGrid& da, AGrid& dv) {
        systemA(pts, a, v, bc, gam, g, ginv, Phi, phihat, tt, da, dv);
    };

    // ---- inverse metric at every grid point, computed once (the metric is static): used for the time step and
    //      for the norm of A in the plots ----
    const vector<array<double,16>> gi = inverseOnGrid(pts, ginv);

    // ---- time step (CFL, see helpers.h).  Near the horizon light is slow in r, so this is much larger than a bound
    //      that assumes the speed c everywhere. ----
    const double dq[3] = { dr, dth, dph };
    const double dt    = cflTimeStep(gi, dq);
    const int    steps0 = int(R / c / dt + 0.5);         // light crosses the radial range once
    const int    steps  = int(tfac * steps0 + 0.5);      // total number of steps
    const int    every  = max(1, steps0 / 40);           // about 40 frames per crossing time
    const int    nframes = steps / every + 1 + (steps % every != 0 ? 1 : 0);   // last step is always written

    fstream rich;
    rich.open("python/dataforpy.dat", ios::out);
    rich << r_s << endl;
    rich << r0 << endl;
    rich << r1 << endl;
    rich << nr << endl;
    rich << nframes << endl;
    rich << every * dt << endl;                  // time between two frames
    rich << nth << endl;
    rich << nph << endl;
    rich << hth << endl;                         // equator row
    rich.close();

    const int iphc = int(lround(phc / dph)) % nph;       // radial line through the centre of the pulse

    int frame = 0;
    auto start = chrono::steady_clock::now();
    double t = 0;
    for (int s = 0; s <= steps; ++s) {
        if (s % every == 0 || s == steps) {
            // radial line through the pulse: columns r, A_r, A_theta, A_phi (perturbation)
            fstream fich;
            fich.open("python/data/Field" + to_string(frame) + ".dat", ios::out);
            for (int i0 = 0; i0 < nr; ++i0) {
                fich << r0 + i0 * dr << " "
                     << pert(i0, hth, iphc, 1) << " " << pert(i0, hth, iphc, 2) << " " << pert(i0, hth, iphc, 3) << "\n";
            }
            fich.close();

            // Field norm on the equatorial plane, one line per (r, phi).
            // Columns: x y |A| A_0.   |A| = spatial part only (A_0 has its own column), perturbation.
            // A_0 = perturbation of A_0 (systemA does not evolve A_0, so this is 0).
            fstream plan;
            plan.open("python/data/Plane" + to_string(frame) + ".dat", ios::out);
            for (int i0 = 0; i0 < nr; ++i0)
                for (int i2 = 0; i2 < nph; ++i2) {
                    const array<double,16>& m = gi[(i0 * nth + hth) * nph + i2];
                    const double Aa[4] = { 0.0, pert(i0, hth, i2, 1), pert(i0, hth, i2, 2), pert(i0, hth, i2, 3) };
                    const double sn = normA(m, Aa);
                    const double r  = r0 + i0 * dr;
                    const double ph = i2 * dph;
                    plan << r * cos(ph) << " " << r * sin(ph) << " " << sn << " "
                         << pert(i0, hth, i2, 0) << "\n";
                }
            plan.close();
            ++frame;
            cout << "step " << s << "/" << steps << "  t = " << t << "  ("
                 << chrono::duration<double>(chrono::steady_clock::now() - start).count() << " s)" << endl;
        }
        if (s < steps) {
            rk4A(rhs, A, V, t, dt);
            if (ikn == 1) rk4A(rhs, Ab, Vb, t, dt);      // background alone, same step
            t += dt;
        }
    }
    cout << "done: " << nframes << " frames written in python/data/" << endl;
    return 0;
}
