#include "example_metric.h"
#include <utility>            // swap

using namespace std;

func minkowski(int i, int j)
{
    if (i > j) swap(i, j);

    if (i == 0 && j == 0) return constant(c * c);
    if (i == 1 && j == 1) return constant(-1.0);
    if (i == 2 && j == 2)
        return [](const array<dual,DIM>& x) {
            dual r = x[1];
            dual res = -r * r;
            return res;
        };
    if (i == 3 && j == 3)
        return [](const array<dual,DIM>& x) {
            dual r = x[1], th = x[2];
            dual res = -r * r * sin(th) * sin(th);
            return res;
        };
    return constant(0.0);
}

func schwarzschild(int i, int j)
{
    if (i > j) swap(i, j);

    if (i == 0 && j == 0)
        return [](const array<dual,DIM>& x) {
            dual r = x[1];
            dual res = c * c * (1.0 - r_s / r);
            return res;
        };
    if (i == 1 && j == 1)
        return [](const array<dual,DIM>& x) {
            dual r = x[1];
            dual res = -1.0 / (1.0 - r_s / r);
            return res;
        };
    if (i == 2 && j == 2)
        return [](const array<dual,DIM>& x) {
            dual r = x[1];
            dual res = -r * r;
            return res;
        };
    if (i == 3 && j == 3)
        return [](const array<dual,DIM>& x) {
            dual r = x[1], th = x[2];
            dual res = -r * r * sin(th) * sin(th);
            return res;
        };
    return constant(0.0);
}

func reissner_nordstrom(int i, int j)
{
    if (i > j) swap(i, j);

    // f = 1 - r_s/r + r_Q^2/r^2
    if (i == 0 && j == 0)
        return [](const array<dual,DIM>& x) {
            dual r = x[1];
            dual f = 1.0 - r_s / r + r_Q2 / (r * r);
            dual res = c * c * f;
            return res;
        };
    if (i == 1 && j == 1)
        return [](const array<dual,DIM>& x) {
            dual r = x[1];
            dual f = 1.0 - r_s / r + r_Q2 / (r * r);
            dual res = -1.0 / f;
            return res;
        };
    if (i == 2 && j == 2)
        return [](const array<dual,DIM>& x) {
            dual r = x[1];
            dual res = -r * r;
            return res;
        };
    if (i == 3 && j == 3)
        return [](const array<dual,DIM>& x) {
            dual r = x[1], th = x[2];
            dual res = -r * r * sin(th) * sin(th);
            return res;
        };
    return constant(0.0);
}

func kerr(int i, int j)
{
    if (i > j) swap(i, j);

    // sigma = r^2 + a^2 cos^2(th), delta = r^2 - r_s r + a^2
    if (i == 0 && j == 0)
        return [](const array<dual,DIM>& x) {
            dual r = x[1], th = x[2];
            dual sigma = r * r + a_spin * a_spin * cos(th) * cos(th);
            dual res = c * c * (1.0 - r_s * r / sigma);
            return res;
        };
    if (i == 0 && j == 3)
        return [](const array<dual,DIM>& x) {
            dual r = x[1], th = x[2];
            dual sigma = r * r + a_spin * a_spin * cos(th) * cos(th);
            dual res = c * r_s * r * a_spin * sin(th) * sin(th) / sigma;
            return res;
        };
    if (i == 1 && j == 1)
        return [](const array<dual,DIM>& x) {
            dual r = x[1], th = x[2];
            dual sigma = r * r + a_spin * a_spin * cos(th) * cos(th);
            dual delta = r * r - r_s * r + a_spin * a_spin;
            dual res = -sigma / delta;
            return res;
        };
    if (i == 2 && j == 2)
        return [](const array<dual,DIM>& x) {
            dual r = x[1], th = x[2];
            dual res = -(r * r + a_spin * a_spin * cos(th) * cos(th));
            return res;
        };
    if (i == 3 && j == 3)
        return [](const array<dual,DIM>& x) {
            dual r = x[1], th = x[2];
            dual sigma = r * r + a_spin * a_spin * cos(th) * cos(th);
            dual res = -(r * r + a_spin * a_spin
                         + r_s * r * a_spin * a_spin * sin(th) * sin(th) / sigma)
                       * sin(th) * sin(th);
            return res;
        };
    return constant(0.0);
}

func kerr_newman(int i, int j)
{
    if (i > j) swap(i, j);

    // sigma = r^2 + a^2 cos^2(th), delta = r^2 - r_s r + a^2 + r_Q^2, e = r_s r - r_Q^2
    if (i == 0 && j == 0)
        return [](const array<dual,DIM>& x) {
            dual r = x[1], th = x[2];
            dual sigma = r * r + a_spin * a_spin * cos(th) * cos(th);
            dual e = r_s * r - r_Q2;
            dual res = c * c * (1.0 - e / sigma);
            return res;
        };
    if (i == 0 && j == 3)
        return [](const array<dual,DIM>& x) {
            dual r = x[1], th = x[2];
            dual sigma = r * r + a_spin * a_spin * cos(th) * cos(th);
            dual e = r_s * r - r_Q2;
            dual res = c * e * a_spin * sin(th) * sin(th) / sigma;
            return res;
        };
    if (i == 1 && j == 1)
        return [](const array<dual,DIM>& x) {
            dual r = x[1], th = x[2];
            dual sigma = r * r + a_spin * a_spin * cos(th) * cos(th);
            dual delta = r * r - r_s * r + a_spin * a_spin + r_Q2;
            dual res = -sigma / delta;
            return res;
        };
    if (i == 2 && j == 2)
        return [](const array<dual,DIM>& x) {
            dual r = x[1], th = x[2];
            dual res = -(r * r + a_spin * a_spin * cos(th) * cos(th));
            return res;
        };
    if (i == 3 && j == 3)
        return [](const array<dual,DIM>& x) {
            dual r = x[1], th = x[2];
            dual sigma = r * r + a_spin * a_spin * cos(th) * cos(th);
            dual e = r_s * r - r_Q2;
            dual res = -(r * r + a_spin * a_spin
                         + e * a_spin * a_spin * sin(th) * sin(th) / sigma)
                       * sin(th) * sin(th);
            return res;
        };
    return constant(0.0);
}

func ellis_wormhole(int i, int j)
{
    if (i > j) swap(i, j);

    if (i == 0 && j == 0) return constant(c * c);
    if (i == 1 && j == 1) return constant(-1.0);
    if (i == 2 && j == 2)
        return [](const array<dual,DIM>& x) {
            dual r = x[1];
            dual res = -(r * r + a_wh * a_wh);
            return res;
        };
    if (i == 3 && j == 3)
        return [](const array<dual,DIM>& x) {
            dual r = x[1], th = x[2];
            dual res = -(r * r + a_wh * a_wh) * sin(th) * sin(th);
            return res;
        };
    return constant(0.0);
}

// ---- Cartesian coordinates (t, x, y, z) ------------------------------------------------

func minkowski_cartesian(int i, int j)
{
    if (i > j) swap(i, j);

    if (i == 0 && j == 0) return constant(c * c);
    if (i == j)           return constant(-1.0);   // (1,1), (2,2), (3,3)
    return constant(0.0);
}

// Ellis wormhole: ds^2 = c^2 dt^2 - dr^2 - (r^2 + a^2) dOmega^2 written in x, y, z.
// With dOmega^2 = (delta_ij - n_i n_j) dx^i dx^j / r^2 this gives
//   g_ij = -[ (1 + a^2/r^2) delta_ij - a^2 x_i x_j / r^4 ]      (i, j = 1..3)
func ellis_cartesian(int i, int j)
{
    if (i > j) swap(i, j);

    if (i == 0 && j == 0) return constant(c * c);
    if (i >= 1 && j <= 3)
        return [i, j](const array<dual,DIM>& x) {
            dual r2 = x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
            dual res = a_wh * a_wh * x[i] * x[j] / (r2 * r2);              // + a^2 x_i x_j / r^4
            if (i == j) res = res - (1.0 + a_wh * a_wh / r2);              // - (1 + a^2/r^2) delta_ij
            return res;
        };
    return constant(0.0);
}
