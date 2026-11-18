#include "example_potential.h"

using namespace std;

// gaussian envelope centred on r_0 with width w (default w_0)
static dual envelope(dual r, double w = w_0)
{
    dual u = (r - r_0) / w;
    dual res = exp(-u * u);
    return res;
}

func pulse(int i)
{
    if (i == 3)
        return [](const array<dual,DIM>& x) {
            dual r = x[1];
            dual res = envelope(r);
            return res;
        };
    return constant(0.0);
}

func wavepacket(int i)
{
    if (i == 3)
        return [](const array<dual,DIM>& x) {
            dual r = x[1];
            dual res = cos(k_0 * (r - r_0)) * envelope(r);
            return res;
        };
    return constant(0.0);
}

func circular(int i)
{
    if (i == 2)
        return [](const array<dual,DIM>& x) {
            dual r = x[1];
            dual res = cos(k_0 * (r - r_0)) * envelope(r);
            return res;
        };
    if (i == 3)
        return [](const array<dual,DIM>& x) {
            dual r = x[1];
            dual res = sin(k_0 * (r - r_0)) * envelope(r);
            return res;
        };
    return constant(0.0);
}

func double_pulse(int i)
{
    if (i == 3)
        return [](const array<dual,DIM>& x) {
            dual r = x[1];
            dual res = envelope(r, 0.5 * w_0);            // narrow bump at r_0
            dual u2 = (r - 1.3 * r_0) / w_0;
            res = res - exp(-u2 * u2);                    // opposite bump further out
            return res;
        };
    return constant(0.0);
}

func soliton(int i)
{
    if (i == 2)
        return [](const array<dual,DIM>& x) {
            dual r = x[1];
            dual u = (r - r_0) / w_0;
            dual res = 1.0 / cosh(u);
            return res;
        };
    return constant(0.0);
}

func chirp(int i)
{
    if (i == 3)
        return [](const array<dual,DIM>& x) {
            dual r = x[1];
            dual xr = (r - r_0) / r_s;
            dual res = cos(k_0 * r_s * xr * (1.0 + 0.2 * xr)) * envelope(r, 2.0 * w_0);
            return res;
        };
    return constant(0.0);
}

func moving(const func& f, double dir)
{
    return [f, dir](const array<dual,DIM>& x) {
        dual res = -dir * c * partial(f, x, 1);
        return res;
    };
}

// ---- Cartesian versions: blobs centred on (pc_x, pc_y, pc_z) -----------------------------

// distance to the pulse centre (a tiny softening avoids 0/0 when a grid point sits exactly on the centre).
// With pc_spherical the grid coordinates (r, theta, phi) are first turned into x, y, z.
static dual radius(const array<dual,DIM>& x)
{
    const double eps = 1e-6 * pc_w;
    dual X = x[1], Y = x[2], Z = x[3];
    if (pc_spherical) {
        X = x[1] * sin(x[2]) * cos(x[3]);
        Y = x[1] * sin(x[2]) * sin(x[3]);
        Z = x[1] * cos(x[2]);
        if (pc_offset > 0.0 && x[1] < 0.0) Z = Z + pc_offset;   // r < 0: other side of the wormhole, far from the pulse
    }
    dual dx = X - pc_x, dy = Y - pc_y, dz = Z - pc_z;
    dual res = sqrt(dx * dx + dy * dy + dz * dz + eps * eps);
    return res;
}

// gaussian of width w around the centre
static dual bump(dual r, double w)
{
    dual u = r / w;
    dual res = exp(-u * u);
    return res;
}

func pulse_cartesian(int i)
{
    if (i == 2)
        return [](const array<dual,DIM>& x) {
            dual r = radius(x);
            dual res = bump(r, pc_w);
            return res;
        };
    return constant(0.0);
}

func wavepacket_cartesian(int i)
{
    if (i == 2)
        return [](const array<dual,DIM>& x) {
            dual r = radius(x);
            dual res = cos(pi / pc_w * r) * bump(r, pc_w);
            return res;
        };
    return constant(0.0);
}

func circular_cartesian(int i)
{
    if (i == 3)
        return [](const array<dual,DIM>& x) {
            dual r = radius(x);
            dual res = cos(pi / pc_w * r) * bump(r, pc_w);
            return res;
        };
    if (i == 2)
        return [](const array<dual,DIM>& x) {
            dual r = radius(x);
            dual res = sin(pi / pc_w * r) * bump(r, pc_w);
            return res;
        };
    return constant(0.0);
}

func double_pulse_cartesian(int i)
{
    if (i == 2)
        return [](const array<dual,DIM>& x) {
            dual r = radius(x);
            dual u2 = (r - 2.0 * pc_w) / pc_w;
            dual res = bump(r, 0.5 * pc_w) - exp(-u2 * u2);   // narrow bump, opposite shell further out
            return res;
        };
    return constant(0.0);
}

func soliton_cartesian(int i)
{
    if (i == 3)
        return [](const array<dual,DIM>& x) {
            dual r = radius(x);
            dual res = 1.0 / cosh(r / pc_w);
            return res;
        };
    return constant(0.0);
}

func chirp_cartesian(int i)
{
    if (i == 2)
        return [](const array<dual,DIM>& x) {
            dual r = radius(x);
            dual xr = r / pc_w;
            dual res = cos(pi * xr * (1.0 + 0.2 * xr)) * bump(r, 2.0 * pc_w);
            return res;
        };
    return constant(0.0);
}

// f depends on x only through the distance d, so d_q f = f'(d) d_q d for every coordinate q:
// f'(d) = sum_q d_q f d_q d / sum_q (d_q d)^2   (in Cartesian coordinates this is (x - x_c) . grad f / d)
func moving_cartesian(const func& f, double dir)
{
    return [f, dir](const array<dual,DIM>& x) {
        func rad = [](const array<dual,DIM>& y) { return radius(y); };
        dual d = radius(x);
        dual num = 0.0, den = 0.0;
        for (int q = 1; q < 4; ++q) {
            dual dd = partial(rad, x, q);
            dual df = partial(f, x, q);
            num = num + df * dd;
            den = den + dd * dd;
        }
        dual res = -dir * pc_amp * pc_w * pc_w * (num / den) / d;
        return res;
    };
}

// ---- own field of the Kerr-Newman black hole (see example_potential.h) ----
func kn_At()
{
    return [](const array<dual,DIM>& x) {
        dual r = x[1], th = x[2];
        dual sigma = r * r + a_spin * a_spin * cos(th) * cos(th);
        dual res = -kn_scale * c * sqrt(r_Q2) * r / sigma;
        return res;
    };
}

func kn_potential(int i)
{
    if (i == 3)
        return [](const array<dual,DIM>& x) {            // A_ph
            dual r = x[1], th = x[2];
            dual sigma = r * r + a_spin * a_spin * cos(th) * cos(th);
            dual res = kn_scale * sqrt(r_Q2) * r * a_spin * sin(th) * sin(th) / sigma;
            return res;
        };
    return constant(0.0);
}
