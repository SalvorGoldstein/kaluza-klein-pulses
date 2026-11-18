// main_wormhole: a pulse starts on the r > 0 side of an ELLIS WORMHOLE and spreads out, through the throat (r = 0)
// to the r < 0 side.
//   ds^2 = c^2 dt^2 - dr^2 - (r^2 + a^2) dOmega^2,      r from -infinity to +infinity,  throat radius a = a_wh.
// The metric is regular everywhere (no horizon, no singularity at r = 0), so the radial grid simply runs from -R to +R
// and the field crosses r = 0 like any other point.  Both outer r faces (r = +R and r = -R) are outgoing-wave
// (Sommerfeld) faces.  Like main_blackhole, the grid covers the full 360 degrees of phi (periodic) and a band of
// theta around the equator (the poles are singular in these coordinates).
// (The reduced 5D equation is solved with rgsim: systemA + rk4A.)
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
    int ipulse, nr;
    double j1, tfac;

    cout << "which initial pulse? (0 = gaussian, 1 = wavepacket, 2 = soliton, 3 = chirp): ";
    cin >> ipulse;
    cout << "grid half-size R (in units of the throat radius a, for example 7: r goes from -R to +R): ";
    cin >> j1;
    cout << "number of radial points (over the whole range -R..R): ";
    cin >> nr;
    cout << "duration (1 = light crosses the radial range once, 3 = three times, ...): ";
    cin >> tfac;
    cout << "perfect!" << endl;

    if (ipulse < 0 || ipulse > 3 || nr < 5 || j1 < 2.0 || tfac <= 0.0) {
        cerr << "invalid input: need 0 <= pulse <= 3, nr >= 5, R >= 2 a, duration > 0" << endl;
        return 1;
    }

    func (*profile[4])(int) = { pulse_cartesian, wavepacket_cartesian, soliton_cartesian, chirp_cartesian };

    // ---- fixed ingredients: Ellis wormhole ----
    const met g    = makeMetric(ellis_wormhole);
    const met ginv = inverseMet(g);
    const christ gam = makeChristoffel(g, ginv);
    const func Phi    = constant(1.0);                 // h = Phi^2
    const func phihat = constant(0.0);                 // no dilaton: pure wave test

    // ---- the grid: r in [-R, R] ----
    const double R  = j1 * a_wh;
    const double r0 = -R;
    const double dr = 2.0 * R / (nr - 1);

    // phi: full circle, square cells at the middle radius |r| = R/2 (areal radius sqrt(r^2 + a^2))
    const double rho_mid = std::sqrt(0.25 * R * R + a_wh * a_wh);
    const int    nph  = max(48, int(lround(2.0 * pi * rho_mid / dr)));
    const double dph  = 2.0 * pi / nph;
    // theta: same step as phi, a band of about +-0.5 rad around the equator (odd number of rows)
    const int    hth  = max(2, int(lround(0.5 / dph)));
    const int    nth  = 2 * hth + 1;
    const double dth  = dph;

    const PointGrid pts = makegrid({nr, nth, nph}, {r0, 0.5 * pi - hth * dth, 0.0}, {dr, dth, dph});
    const int P = nr * nth * nph;
    cout << "grid: " << nr << " x " << nth << " x " << nph << " = " << P << " points" << endl;

    // r faces (r = -R and r = +R): outgoing-wave condition, theta faces too, phi PERIODIC
    const int bc[3][2] = { { BC_OUTGOING, BC_OUTGOING },
                           { BC_OUTGOING, BC_OUTGOING },
                           { BC_PERIODIC, BC_PERIODIC } };

    // ---- random pulse centre on the r > 0 side (between 50 % and 65 % of R), phi anywhere,
    //      theta close to the equator.  The width is small enough that the pulse does not touch the throat at t = 0. ----
    random_device rd;
    mt19937 rng(rd());
    uniform_real_distribution<double> runif(0.0, 1.0);

    const double rc  = (0.5 + 0.15 * runif(rng)) * R;
    const double thc = 0.5 * pi + (2.0 * runif(rng) - 1.0) * 0.3 * hth * dth;
    const double phc = 2.0 * pi * runif(rng);
    cout << "random pulse centre: r = " << rc / a_wh << " a,  th = " << thc
         << " rad,  ph = " << phc << " rad" << endl;

    pc_spherical = true;                                  // the grid is (r, theta, phi)
    pc_offset = 100.0 * R;                                // the r < 0 side is a different place, far from the pulse
    pc_x = rc * sin(thc) * cos(phc);                      // centre of the pulse in Cartesian coordinates
    pc_y = rc * sin(thc) * sin(phc);
    pc_z = rc * cos(thc);
    pc_w = 0.06 * (2.0 * R);                              // width of the pulse

    // ---- the unknowns: A_i(t=0) and d_t A_i(t=0) from example_potential, a round blob around the centre ----
    const fvec Apulse = initialPotentialSpherical(profile[ipulse], +1.0);
    AGrid A, V;
    fillPotential(pts, Apulse, A, V);

    const RhsA rhs = [&](const AGrid& a, const AGrid& v, double tt, AGrid& da, AGrid& dv) {
        systemA(pts, a, v, bc, gam, g, ginv, Phi, phihat, tt, da, dv);
    };

    // ---- time step (CFL, see helpers.h) ----
    const vector<array<double,16>> gi = inverseOnGrid(pts, ginv);
    const double dq[3] = { dr, dth, dph };
    const double dt    = cflTimeStep(gi, dq);
    const int    steps0 = int(2.0 * R / c / dt + 0.5);   // light crosses the radial range once
    const int    steps  = int(tfac * steps0 + 0.5);      // total number of steps
    const int    every  = max(1, steps0 / 40);           // about 40 frames per crossing time
    const int    nframes = steps / every + 1 + (steps % every != 0 ? 1 : 0);   // last step is always written

    fstream rich;
    rich.open("python/dataforpy.dat", ios::out);
    rich << a_wh << endl;
    rich << r0 << endl;
    rich << R << endl;
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
            // radial line through the pulse: columns r, A_r, A_theta, A_phi
            fstream fich;
            fich.open("python/data/Field" + to_string(frame) + ".dat", ios::out);
            for (int i0 = 0; i0 < nr; ++i0) {
                const array<double,4> a = A[i0][hth][iphc].getvalue();
                fich << r0 + i0 * dr << " " << a[1] << " " << a[2] << " " << a[3] << "\n";
            }
            fich.close();

            // field norm on the equatorial plane: columns r, phi, |A|, one line per (r, phi)
            fstream plan;
            plan.open("python/data/Plane" + to_string(frame) + ".dat", ios::out);
            for (int i0 = 0; i0 < nr; ++i0)
                for (int i2 = 0; i2 < nph; ++i2) {
                    const array<double,4> a = A[i0][hth][i2].getvalue();
                    const double Aa[4] = { 0.0, a[1], a[2], a[3] };            // A_0 = 0
                    plan << r0 + i0 * dr << " " << i2 * dph << " "
                         << normA(gi[((size_t)i0 * nth + hth) * nph + i2], Aa) << "\n";
                }
            plan.close();
            ++frame;
            cout << "step " << s << "/" << steps << "  t = " << t << "  ("
                 << chrono::duration<double>(chrono::steady_clock::now() - start).count() << " s)" << endl;
        }
        if (s < steps) { rk4A(rhs, A, V, t, dt); t += dt; }
    }
    cout << "done: " << nframes << " frames written in python/data/" << endl;
    return 0;
}
