#pragma once
#include <cmath>
#include "rgsim/rgsim.h"
#include "example_metric.h"          // r_s, c, pi

using namespace rgsim;
using namespace rgsim::constants;

// parameters shared by the examples (lengths in meters)
inline constexpr double r_0   = 8.0 * r_s;     // pulse center
inline constexpr double w_0   = 1.0 * r_s;     // pulse width
inline constexpr double k_0   = 2.0 * pi / (2.0 * r_s);   // wave number

// ---- placement of the Cartesian pulses (set by main.cc, lengths in metres) ----
inline double pc_x = 0.0, pc_y = 0.0, pc_z = 0.0;   // centre of the pulse
inline double pc_w   = 1.0;                         // width of the pulse
inline double pc_amp = 1e-5;                        // size of the field (keep it small: linear regime)
inline double pc_offset = 0.0;                      // two-sided spaces (wormhole): the r < 0 side is placed this far away
                                                    //        in z, so a pulse centred at r > 0 does not show up at r < 0
inline bool   pc_spherical = false;                 // false: grid coordinates are (x, y, z)
                                                    // true : grid coordinates are (r, theta, phi); the centre
                                                    //        (pc_x, pc_y, pc_z) is then given in Cartesian coordinates

// A_i(t=0, r). Same idea as schwarzschild(i, j): index = coordinate (0=t, 1=r, 2=th, 3=ph)
func pulse(int i);          // gaussian bump in A_phi
func wavepacket(int i);     // cos(k r) * gaussian in A_phi
func circular(int i);       // circularly polarized packet (A_th, A_phi in quadrature)
func double_pulse(int i);   // two separated bumps, opposite signs
func soliton(int i);        // sech profile in A_th
func chirp(int i);          // packet whose wavelength shrinks with r

// ---- Cartesian versions: coordinates (0=t, 1=x, 2=y, 3=z), A_x, A_y, A_z ----
// Blobs centred on (pc_x, pc_y, pc_z) with width pc_w, as functions of the distance r to that centre.
// No r_0 and no r_s: everything is set by pc_x, pc_y, pc_z, pc_w.
func pulse_cartesian(int i);          // gaussian bump in A_y
func wavepacket_cartesian(int i);     // cos(k r) * gaussian in A_y, k = pi / pc_w
func circular_cartesian(int i);       // circularly polarized packet (A_z, A_y in quadrature)
func double_pulse_cartesian(int i);   // narrow central bump and an opposite shell of radius 2 pc_w
func soliton_cartesian(int i);        // sech profile in A_z
func chirp_cartesian(int i);          // packet whose wavelength shrinks with r

// ---- the own electromagnetic field of a Kerr-Newman black hole ----
// Its potential (geometric form, sigma = r^2 + a^2 cos^2(th), r_Q = charge length):
//     A = -(r_Q r / sigma) * ( c dt - a sin^2(th) dphi )     ->  A_t = -c r_Q r / sigma ,  A_ph = r_Q r a sin^2(th) / sigma
// It is static.  A_t is kept as a fixed background function (field.hpp: A0), and the evolved unknowns start
// from the static spatial part A_ph (A_r = A_th = 0), so the background stays put by itself.
// kn_scale multiplies the whole field (1 = natural size in these units).
inline double kn_scale = 1.0;
func kn_At();                         // A_t(r, theta)
func kn_potential(int i);             // i = 1, 2, 3 (r, theta, phi): static spatial part (only A_phi is non-zero)

// d_t A_i(t=0) so that the profile travels: dir = +1 outward, -1 inward
// (uses d_t f = -dir * c * d_r f, i.e. f(r - dir*c*t), flat-space approximation)
func moving(const func& f, double dir);

// same as moving, but in Cartesian coordinates around the pulse centre: d_r f = ((x - x_c) . grad f) / r.
// The result is divided by r again and multiplied by pc_amp * pc_w^2, so the field has the size pc_amp
// whatever the grid size is (a field of order 1 makes the nonlinear term of the equation blow up).
func moving_cartesian(const func& f, double dir);

// builds the whole initial potential the way field.hpp expects it:
// A_0 = 0, A_i = moving(profile(i), dir).  Use it directly in makeInitialState.
inline fvec initialPotential(func (*profile)(int), double dir)
{
    fvec A(4, constant(0.0));
    for (int i = 1; i < 4; ++i)
        A[i] = moving(profile(i), dir);
    return A;
}

// Cartesian version of initialPotential: A_0 = 0, A_i = moving_cartesian(profile(i), dir), i = x, y, z.
inline fvec initialPotentialCartesian(func (*profile)(int), double dir)
{
    fvec A(4, constant(0.0));
    for (int i = 1; i < 4; ++i)
        A[i] = moving_cartesian(profile(i), dir);
    return A;
}

// Same for a spherical grid (r, theta, phi) with pc_spherical = true: the blob is round in real space, so it
// is seamless across phi = 0 / 2 pi.  The components are the coordinate ones (A_r, A_th, A_ph): they are
// multiplied by 1, r and r sin(theta) so that the physical size of the field is pc_amp everywhere.
inline fvec initialPotentialSpherical(func (*profile)(int), double dir)
{
    fvec A(4, constant(0.0));
    for (int i = 1; i < 4; ++i) {
        func m = moving_cartesian(profile(i), dir);
        A[i] = [m, i](const std::array<dual,DIM>& x) {
            dual s = 1.0;
            if (i == 2) s = x[1];
            if (i == 3) s = x[1] * sin(x[2]);
            dual res = m(x) * s;
            return res;
        };
    }
    return A;
}