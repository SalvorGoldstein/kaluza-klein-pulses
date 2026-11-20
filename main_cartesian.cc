// main_cartesian: a pulse starts at a chosen (or random) place of a box of FLAT space (Minkowski, Cartesian t, x, y, z)
// and spreads out.  The grid is a cube of side "size", with the same number of points on every axis.
// g, Phi and the dilaton phihat are fixed; the reduced 5D equation (rgsim: systemA + rk4A) evolves A.
// Every face of the box lets the waves leave (outgoing-wave condition).
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <random>
#include <array>
#include <cstdlib>

#include "rgsim/rgsim.h"
#include "example_metric.h"
#include "example_potential.h"
#include "helpers.h"

using namespace std; using namespace rgsim;

int main()
{
    int ipulse, icentre, nr;
    double size, tfac;

    cout << "which initial pulse? (0 = gaussian, 1 = wavepacket, 2 = soliton, 3 = chirp): ";
    cin >> ipulse;
    cout << "pulse centre? (0 = centre of the grid, 1 = random, mostly near the centre): ";
    cin >> icentre;
    cout << "grid size (in metres): ";
    cin >> size;
    cout << "number of points per axis: ";
    cin >> nr;
    cout << "duration (1 = light crosses the grid once, 3 = three times, ...): ";
    cin >> tfac;
    const int idil = askDilaton();
    cout << "perfect!" << endl;

    if (ipulse < 0 || ipulse > 3 || icentre < 0 || icentre > 1 || nr < 5 || size <= 0.0 || tfac <= 0.0) {
        cerr << "invalid input: need 0 <= pulse <= 3, centre 0 or 1, nr >= 5, size > 0, duration > 0" << endl;
        return 1;
    }

    func (*profile[4])(int) = { pulse_cartesian, wavepacket_cartesian, soliton_cartesian, chirp_cartesian };

    // ---- fixed ingredients: flat space ----
    const met g    = makeMetric(minkowski_cartesian);
    const met ginv = inverseMet(g);
    const christ gam = makeChristoffel(g, ginv);
    const func Phi    = constant(1.0);                 // h = Phi^2
    const func phihat = constant(0.0);                 // no dilaton: pure wave test

    // ---- the grid: nr points on every axis, same spacing dr.
    //      x in [0, size] (points at the cell centres, so x = 0 is never a node),
    //      y and z centred on zero ----
    const double dr = size / nr;
    const double r0 = 0.5 * dr, r1 = size - 0.5 * dr;   // first and last x
    const int    mid = nr / 2;
    const PointGrid pts = makegrid({nr, nr, nr}, {r0, -mid * dr, -mid * dr}, {dr, dr, dr});
    const int P = nr * nr * nr;
    cout << "grid: " << nr << " x " << nr << " x " << nr << " = " << P << " points" << endl;

    // no wrap-around: every face lets the waves leave (outgoing-wave condition)
    const int bc[3][2] = { { BC_OUTGOING, BC_OUTGOING },
                           { BC_OUTGOING, BC_OUTGOING },
                           { BC_OUTGOING, BC_OUTGOING } };

    // ---- the pulse: centre of the grid, or a random place (normal law around the centre,
    //      spread size/12, never further than size/4 from it) ----
    if (const char* e = getenv("PC_AMP")) pc_amp = atof(e);
    pc_x = 0.5 * size;  pc_y = 0.0;  pc_z = 0.0;         // centre of the grid
    pc_w = size / 10.0;                                  // width of the pulse
    if (icentre == 1) {
        mt19937 gen(random_device{}());
        normal_distribution<double> nd(0.0, size / 12.0);
        auto shift = [&]() { return max(-0.25 * size, min(0.25 * size, nd(gen))); };
        pc_x += shift();  pc_y += shift();  pc_z += shift();
    }
    cout << "pulse centre: x = " << pc_x << " m, y = " << pc_y << " m, z = " << pc_z
         << " m, width = " << pc_w << " m" << endl;

    // the slices written to disk go through the centre of the pulse
    const int iz = max(1, min(nr - 2, int(lround(pc_z / dr + mid))));
    const int iy = max(1, min(nr - 2, int(lround(pc_y / dr + mid))));

    // ---- the unknowns: A_i(t=0) and d_t A_i(t=0) from example_potential, outgoing from the centre ----
    const fvec Apulse = initialPotentialCartesian(profile[ipulse], +1.0);
    AGrid A, V;
    fillPotential(pts, Apulse, A, V);

    const RhsA rhs = [&](const AGrid& a, const AGrid& v, double tt, AGrid& da, AGrid& dv) {
        systemA(pts, a, v, bc, gam, g, ginv, Phi, phihat, tt, da, dv);
    };
    SGrid phi = makeScalarGrid(pts), Pi = makeScalarGrid(pts);          // dilaton and d_t dilaton (start at 0)
    const RhsAphi rhsD = [&](const AGrid& a, const AGrid& v, const SGrid& p, const SGrid& q, double tt,
                             AGrid& da, AGrid& dv, SGrid& dp, SGrid& dq) {
        systemAphi(pts, a, v, p, q, bc, gam, g, ginv, Phi, tt, da, dv, dp, dq);
    };
    const vector<array<double,16>> gi = inverseOnGrid(pts, ginv);

    // ---- time loop (CFL: dt < dr / c, same spacing on every axis) ----
    const double dt    = 0.4 * dr / c;
    const int    steps0 = int(size / c / dt + 0.5);      // light crosses the grid once
    const int    steps  = int(tfac * steps0 + 0.5);      // total number of steps
    const int    every  = max(1, steps0 / 40);           // about 40 frames per crossing time
    const int    nframes = steps / every + 1 + (steps % every != 0 ? 1 : 0);   // last step is always written

    fstream rich;
    rich.open("python/dataforpy.dat", ios::out);
    rich << size << endl;
    rich << r0 << endl;
    rich << r1 << endl;
    rich << nr << endl;
    rich << nframes << endl;
    rich << every * dt << endl;                  // time between two frames
    rich.close();

    int frame = 0;
    auto start = chrono::steady_clock::now();
    double t = 0;
    for (int s = 0; s <= steps; ++s) {
        if (s % every == 0 || s == steps) {
            // the line through the centre, along x: columns x, A_x, A_y, A_z
            fstream fich;
            fich.open("python/data/Field" + to_string(frame) + ".dat", ios::out);
            for (int i0 = 0; i0 < nr; ++i0) {
                const array<double,4> a = A[i0][iy][iz].getvalue();
                fich << r0 + i0 * dr << " " << a[1] << " " << a[2] << " " << a[3] << "\n";
            }
            fich.close();

            // field norm on the plane z = pc_z: columns x, y, |A|
            fstream plan;
            plan.open("python/data/Plane" + to_string(frame) + ".dat", ios::out);
            for (int i0 = 0; i0 < nr; ++i0)
                for (int i1 = 0; i1 < nr; ++i1) {
                    const array<double,4> a = A[i0][i1][iz].getvalue();
                    const double Aa[4] = { 0.0, a[1], a[2], a[3] };            // A_0 = 0
                    plan << r0 + i0 * dr << " " << (i1 - mid) * dr << " "
                         << normA(gi[((size_t)i0 * nr + i1) * nr + iz], Aa);
                    if (idil) plan << " " << phi[i0][i1][iz];                  // column 4: the dilaton
                    plan << "\n";
                }
            plan.close();
            ++frame;
            cout << "step " << s << "/" << steps << "  t = " << t << "  ("
                 << chrono::duration<double>(chrono::steady_clock::now() - start).count() << " s)" << endl;
        }
        if (s < steps) {
            if (idil) rk4Aphi(rhsD, A, V, phi, Pi, t, dt); else rk4A(rhs, A, V, t, dt);
            t += dt;
        }
    }
    cout << "done: " << nframes << " frames written in python/data/" << endl;
    return 0;
}
