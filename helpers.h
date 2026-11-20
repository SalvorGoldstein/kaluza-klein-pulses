// Small helpers shared by the three main programs (main_blackhole.cc, main_cartesian.cc, main_wormhole.cc).
// They only use the rgsim library; include this file after "rgsim/rgsim.h".
#pragma once
#include <array>
#include <vector>
#include "rgsim/rgsim.h"

// Grid of points with nn[k] points on axis k, first point q0[k], step dq[k].
// (Same layout as makegridspherical / makegridcarte in field.h, but with any start and step.)
inline rgsim::PointGrid makegrid(std::array<int,3> nn, std::array<double,3> q0, std::array<double,3> dq)
{
    rgsim::PointGrid pts(nn[0], std::vector<std::vector<rgsim::Point>>(nn[1], std::vector<rgsim::Point>(nn[2])));
    for (int i0 = 0; i0 < nn[0]; ++i0)
        for (int i1 = 0; i1 < nn[1]; ++i1)
            for (int i2 = 0; i2 < nn[2]; ++i2)
                pts[i0][i1][i2].setcoord({ q0[0] + i0 * dq[0], q0[1] + i1 * dq[1], q0[2] + i2 * dq[2] });
    return pts;
}

// Coordinates of a grid point written the way the metric functions want them: x = (t, x1, x2, x3, extra = 0).
inline std::array<rgsim::dual,DIM> pointX(const rgsim::PointGrid& pts, int i0, int i1, int i2, double t = 0.0)
{
    std::array<rgsim::dual,DIM> x;
    for (int a = 0; a < DIM; ++a) x[a] = 0.0;
    const std::array<double,3> q = pts[i0][i1][i2].getcoord();
    x[0] = t; x[1] = q[0]; x[2] = q[1]; x[3] = q[2];
    return x;
}

// The 4x4x4 Christoffel symbols of the metric g (ginv = its inverse), as functions of the coordinates.
inline rgsim::christ makeChristoffel(const rgsim::met& g, const rgsim::met& ginv)
{
    rgsim::christ gam(4, std::vector<std::vector<rgsim::func>>(4, std::vector<rgsim::func>(4)));
    for (int l = 0; l < 4; ++l)
        for (int m = 0; m < 4; ++m)
            for (int n = 0; n < 4; ++n)
                gam[l][m][n] = rgsim::christoffel4D(l, m, n, g, ginv);
    return gam;
}

// A_mu(t=0) and d_t A_mu(t=0) on the grid, from the four functions Af[0..3].
// (A_0 is stored too; systemA does not evolve it.)
inline void fillPotential(const rgsim::PointGrid& pts, const rgsim::fvec& Af, rgsim::AGrid& A, rgsim::AGrid& V)
{
    const int n0 = pts.size(), n1 = pts[0].size(), n2 = pts[0][0].size();
    A = rgsim::AGrid(n0, std::vector<std::vector<rgsim::VectorPotential>>(n1, std::vector<rgsim::VectorPotential>(n2)));
    V = A;
    for (int i0 = 0; i0 < n0; ++i0)
        for (int i1 = 0; i1 < n1; ++i1)
            for (int i2 = 0; i2 < n2; ++i2) {
                const std::array<rgsim::dual,DIM> x = pointX(pts, i0, i1, i2);
                std::array<double,4> a, v;
                for (int m = 0; m < 4; ++m) {
                    a[m] = Af[m](x).val;
                    v[m] = rgsim::partial(Af[m], x, 0).val;
                }
                A[i0][i1][i2].setvalue(a);
                V[i0][i1][i2].setvalue(v);
            }
}

// a += b, point by point
inline void addInto(rgsim::AGrid& a, const rgsim::AGrid& b)
{
    for (size_t i0 = 0; i0 < a.size(); ++i0)
        for (size_t i1 = 0; i1 < a[0].size(); ++i1)
            for (size_t i2 = 0; i2 < a[0][0].size(); ++i2) {
                std::array<double,4> x = a[i0][i1][i2].getvalue();
                const std::array<double,4> y = b[i0][i1][i2].getvalue();
                for (int m = 0; m < 4; ++m) x[m] += y[m];
                a[i0][i1][i2].setvalue(x);
            }
}

// Inverse metric at every grid point (static metric), stored as gi[(i0*n1 + i1)*n2 + i2][4*a + b].
// Used for the time step and for the norm of A in the plots.
inline std::vector<std::array<double,16>> inverseOnGrid(const rgsim::PointGrid& pts, const rgsim::met& ginv)
{
    const int n0 = pts.size(), n1 = pts[0].size(), n2 = pts[0][0].size();
    std::vector<std::array<double,16>> gi((size_t)n0 * n1 * n2);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i0 = 0; i0 < n0; ++i0)
        for (int i1 = 0; i1 < n1; ++i1)
            for (int i2 = 0; i2 < n2; ++i2) {
                const std::array<rgsim::dual,DIM> x = pointX(pts, i0, i1, i2);
                std::array<double,16>& m = gi[((size_t)i0 * n1 + i1) * n2 + i2];
                for (int a = 0; a < 4; ++a)
                    for (int b = 0; b < 4; ++b) m[4 * a + b] = ginv[a][b](x).val;
            }
    return gi;
}

// |A| = sqrt(|g^{ab} A_a A_b|) from the inverse metric m (one entry of inverseOnGrid) and A_a.
inline double normA(const std::array<double,16>& m, const double Aa[4])
{
    double s = 0.0;
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b) s += m[4 * a + b] * Aa[a] * Aa[b];
    return std::sqrt(std::fabs(s));
}

// Largest allowed time step over the whole grid (CFL).  Along axis i a wave moves at most at the speed
//     c_i = ( |g^{0i}| + sqrt( (g^{0i})^2 - g^{00} g^{ii} ) ) / g^{00}
// (the g^{0i} term is the frame dragging of a rotating hole) and dt must be smaller than
// 1 / sqrt(sum_i c_i^2 / dq_i^2) at every point.  safety = 0.7 gives the same margin as 0.4 dq / c in flat space.
inline double cflTimeStep(const std::vector<std::array<double,16>>& gi, const double dq[3], double safety = 0.7)
{
    double w2max = 0.0;
    for (size_t p = 0; p < gi.size(); ++p) {
        double w2 = 0.0;
        for (int i = 0; i < 3; ++i) {
            const double g00 = gi[p][0], g0i = gi[p][i + 1], gii = gi[p][5 * (i + 1)];
            const double ci  = (std::fabs(g0i) + std::sqrt(g0i * g0i - g00 * gii)) / g00;
            w2 += ci * ci / (dq[i] * dq[i]);
        }
        w2max = std::max(w2max, w2);
    }
    return safety / std::sqrt(w2max);
}

// ---- dilaton (see rgsim/system.h) ----
// Zero scalar grid with the same shape as the point grid.
inline rgsim::SGrid makeScalarGrid(const rgsim::PointGrid& pts, double v = 0.0)
{
    return rgsim::SGrid(pts.size(), std::vector<std::vector<double>>(pts[0].size(), std::vector<double>(pts[0][0].size(), v)));
}

// Optional last questions of the three programs: with the dilaton the scalar field is evolved too (stable for much
// larger pulses). Missing answers (old input files) mean "no dilaton".
inline int askDilaton()
{
    int idil = 0;
    std::cout << "dilaton? (0 = no: frozen scalar, old equation; 1 = yes: dynamical dilaton, stable) [0]: ";
    if (!(std::cin >> idil)) idil = 0;
    if (idil) {
        double a = rgsim::dilatonCoupling;
        std::cout << "dilaton coupling a (Kaluza-Klein value sqrt(3) = 1.732; smaller = weaker coupling, more stable) [1.732]: ";
        if (std::cin >> a && a > 0.0) rgsim::dilatonCoupling = a;
    }
    return idil;
}
