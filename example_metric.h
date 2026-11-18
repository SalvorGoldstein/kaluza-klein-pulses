#pragma once
#include <cmath>
#include "rgsim/rgsim.h"

using namespace rgsim;
using namespace rgsim::constants;   // G, c, eps0, pi

// parameters shared by the examples (SI units)
inline constexpr double M_bh    = 5.972e30;                          // black hole mass
inline constexpr double r_s     = 2.0 * G * M_bh / (c * c);          // Schwarzschild radius
inline constexpr double r_earth = 6378137.0;
inline constexpr double r_moon  = 3.844e8;
inline constexpr double Q    = 0.5 * M_bh * std::sqrt(4 * pi * eps0 * G);   // charge
inline constexpr double J_bh    = G * M_bh * M_bh / c;               // angular momentum
inline constexpr double a_spin  = J_bh / (M_bh * c);                 // spin length a = J/(Mc)
inline constexpr double r_Q2    = G * Q * Q / (4 * pi * eps0 * c * c * c * c);
inline constexpr double a_wh    = r_s;                               // wormhole throat size

// g_{mu nu}(t, r, th, ph), signature (+,-,-,-), index = coordinate (0=t, 1=r, 2=th, 3=ph)
func minkowski(int i, int j);            // flat space, spherical coordinates
func schwarzschild(int i, int j);
func reissner_nordstrom(int i, int j);   // charged, no spin
func kerr(int i, int j);                 // spin, no charge
func kerr_newman(int i, int j);          // spin and charge
func ellis_wormhole(int i, int j);

// Cartesian versions, same index convention but coordinates (0=t, 1=x, 2=y, 3=z)
// (the wormhole throat is at r = sqrt(x^2+y^2+z^2) = 0, so keep the grid away from the origin)
func minkowski_cartesian(int i, int j);   // flat space
func ellis_cartesian(int i, int j);       // Ellis wormhole, same metric as ellis_wormhole

// builds the whole 4x4 metric from one of the functions above
inline met makeMetric(func (*m)(int, int))
{
    met g(4, fvec(4));
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            g[i][j] = m(i, j);
    return g;
}