#include "system.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include "Geodiff.h"
#include "funcoperator.h"

namespace rgsim {

// ======================================================================
// Three small helper functions (only used in this file, hence "static")
// ======================================================================

// 1) The position of the grid point (i0,i1,i2) at time t, written the way the metric functions want it:
//    x[0] = time, x[1..3] = the three coordinates, x[4] = the extra dimension (always 0).
static std::array<dual,DIM> pointX(const PointGrid& pts, int i0, int i1, int i2, double t)
{
    std::array<dual,DIM> x;
    for (int a = 0; a < DIM; a++) x[a] = 0.0;
    std::array<double,3> c = pts[i0][i1][i2].getcoord();
    x[0] = t;
    x[1] = c[0];
    x[2] = c[1];
    x[3] = c[2];
    return x;
}

// 2) Component c (0..3) of the grid G at the point whose three indices are idx[0], idx[1], idx[2].
static double val(const AGrid& G, const int idx[3], int c)
{
    std::array<double,4> v = G[idx[0]][idx[1]][idx[2]].getvalue();
    return v[c];
}

// 3) Derivative along the OUTWARD direction on a border (second order, one-sided).
//    f0 = value on the border, f1 = next point inside, f2 = the one after, h = spacing.
static double dOutward(double f0, double f1, double f2, double h)
{
    return (3.0 * f0 - 4.0 * f1 + f2) / (2.0 * h);
}


// ======================================================================
// Metric data cache
// ======================================================================
// The inverse metric, the Christoffel symbols, h and box only depend on the position (and on time if the metric
// depends on time). Computing them through the autodiff functions is slow, so we do it ONCE per grid point and
// keep the numbers. If the metric depends on t, set  metricIsStatic = false  and they are rebuilt at every call.
// If you change the metric, Phi, phihat or the grid, call resetMetricCache().
bool metricIsStatic = true;
bool metricIsHomogeneous = false;

namespace {
struct PointData {
    double gi[4][4];       // inverse metric
    double Gam[4][4][4];   // Christoffel symbols
    double Gc[4];          // Gc^l = g^{ar} Gam^l_{ar}
    double h, Phil;        // h = Phi^2 and Phi
    double box;            // g^{ab} d_a d_b phihat
    double dlnh[4];        // d_a ln h  (used by the dilaton equations)
};
std::vector<PointData> cache;
int    cacheN[3] = {0, 0, 0};
double cacheT    = 0.0;
bool   cacheOk   = false;
}

void resetMetricCache() { cacheOk = false; }

static void fillPointData(PointData& pd, const std::array<dual,DIM>& x,
                          const christ& gam, const met& ginv, const func& Phi, const func& phihat)
{
    for (int a = 0; a < 4; a++)
        for (int b = 0; b < 4; b++)
            pd.gi[a][b] = ginv[a][b](x).val;

    for (int l = 0; l < 4; l++)
        for (int m = 0; m < 4; m++)
            for (int nu = 0; nu < 4; nu++)
                pd.Gam[l][m][nu] = gam[l][m][nu](x).val;

    for (int l = 0; l < 4; l++) {
        pd.Gc[l] = 0.0;
        for (int a = 0; a < 4; a++)
            for (int r = 0; r < 4; r++)
                pd.Gc[l] += pd.gi[a][r] * pd.Gam[l][a][r];
    }

    pd.Phil = Phi(x).val;
    for (int a = 0; a < 4; a++) {
        func Phi_ = Phi;
        pd.dlnh[a] = 2.0 * partial(Phi_, x, a).val / pd.Phil;
    }
    pd.h = pd.Phil * pd.Phil;

    pd.box = 0.0;
    for (int b = 0; b < 4; b++) {
        func dphihat_b = [&](const std::array<dual,DIM>& y) { return partial(phihat, y, b); };
        for (int a = 0; a < 4; a++)
            pd.box += pd.gi[a][b] * partialFD(dphihat_b, x, a).val;
    }
}

// ======================================================================
// The equation of motion
// ======================================================================
double dilatonCoupling = std::sqrt(3.0);

static void systemImpl(const PointGrid& pts, const AGrid& A, const AGrid& V,
             const SGrid* phiP, const SGrid* PiP,
             const int bc[3][2],
             const christ& gam, const met& g, const met& ginv,
             const func& Phi, const func& phihat, double t,
             AGrid& dA, AGrid& dV, SGrid* dphiP, SGrid* dPiP)
{
    (void)g;
    const bool dil = (phiP != nullptr);
    const double acoup = dilatonCoupling;    // the equation only needs ginv and gam, g is not used here

    // ---------------- Step 1: sizes, periodic axes, spacing ----------------
    int n[3];                                   // number of points on each axis
    n[0] = A.size();
    n[1] = A[0].size();
    n[2] = A[0][0].size();
    if (n[0] < 3 || n[1] < 3 || n[2] < 3) {
        std::cout << "systemA: you need at least 3 points on each axis" << std::endl;
        return;
    }

    bool periodic[3];                           // true if the axis wraps around
    for (int k = 0; k < 3; k++) {
        bool lowerIsPeriodic = (bc[k][0] == BC_PERIODIC);
        bool upperIsPeriodic = (bc[k][1] == BC_PERIODIC);
        if (lowerIsPeriodic != upperIsPeriodic) {
            std::cout << "systemA: periodic must be set on both faces of an axis" << std::endl;
            return;
        }
        periodic[k] = lowerIsPeriodic;
    }

    double dq[3];                               // spacing on each axis (the grid must be uniform)
    dq[0] = pts[1][0][0].getcoord()[0] - pts[0][0][0].getcoord()[0];
    dq[1] = pts[0][1][0].getcoord()[1] - pts[0][0][0].getcoord()[1];
    dq[2] = pts[0][0][1].getcoord()[2] - pts[0][0][0].getcoord()[2];

    dA = A;                                     // gives dA and dV the right size
    dV = A;
    if (dil) { *dphiP = *phiP; *dPiP = *phiP; }

    // ---------------- Step 1b: metric data at every grid point (cached) ----------------
    if (!cacheOk || (!metricIsStatic && cacheT != t) ||
        cacheN[0] != n[0] || cacheN[1] != n[1] || cacheN[2] != n[2]) {
        cache.resize((size_t)n[0] * n[1] * n[2]);
        if (metricIsHomogeneous) {
            // same metric data at every point (e.g. flat FLRW): evaluate once, copy everywhere (cheap even if g depends on t)
            PointData one;
            fillPointData(one, pointX(pts, 0, 0, 0, t), gam, ginv, Phi, phihat);
            for (size_t p = 0; p < cache.size(); p++) cache[p] = one;
        } else {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i0 = 0; i0 < n[0]; i0++)
            for (int i1 = 0; i1 < n[1]; i1++)
                for (int i2 = 0; i2 < n[2]; i2++)
                    fillPointData(cache[((size_t)i0 * n[1] + i1) * n[2] + i2],
                                  pointX(pts, i0, i1, i2, t), gam, ginv, Phi, phihat);
        }
        cacheN[0] = n[0]; cacheN[1] = n[1]; cacheN[2] = n[2];
        cacheT = t;
        cacheOk = true;
    }

    // ---------------- Step 2: points that are NOT on an open border ----------------
    // Here we use the equation of motion.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i0 = 0; i0 < n[0]; i0++) {
        for (int i1 = 0; i1 < n[1]; i1++) {
            for (int i2 = 0; i2 < n[2]; i2++) {

                int ii[3] = {i0, i1, i2};

                // is this point on an open border? then skip it, step 3 will handle it
                bool onBorder = false;
                for (int k = 0; k < 3; k++) {
                    if (!periodic[k] && (ii[k] == 0 || ii[k] == n[k] - 1)) onBorder = true;
                }
                if (onBorder) continue;

                // index of the next and of the previous point on each axis (wrapping if periodic)
                int up[3], down[3];
                for (int k = 0; k < 3; k++) {
                    up[k]   = ii[k] + 1;
                    down[k] = ii[k] - 1;
                    if (periodic[k]) {
                        if (up[k] == n[k]) up[k] = 0;
                        if (down[k] < 0)   down[k] = n[k] - 1;
                    }
                }

                // ---- 2a: the metric quantities at this point (from the cache) ----
                const PointData& pd = cache[((size_t)i0 * n[1] + i1) * n[2] + i2];
                const double (&gi)[4][4]      = pd.gi;
                const double (&Gam)[4][4][4]  = pd.Gam;
                const double (&Gc)[4]         = pd.Gc;
                const double Phil = pd.Phil;
                const double h    = pd.h;
                const double box  = pd.box;

                // ---- 2b: A and its derivatives at this point ----
                // Am[c]       = A_c
                // dAm[a][c]   = d_a A_c         (a = 0 is time, a = 1,2,3 are the space axes)
                // dd[a][b][c] = d_a d_b A_c
                double Am[4];
                double dAm[4][4];
                double dd[4][4][4];
                for (int a = 0; a < 4; a++) {
                    Am[a] = 0.0;
                    for (int b = 0; b < 4; b++) {
                        dAm[a][b] = 0.0;
                        for (int c = 0; c < 4; c++) dd[a][b][c] = 0.0;
                    }
                }

                for (int c = 0; c < 4; c++) {
                    Am[c]     = val(A, ii, c);
                    dAm[0][c] = val(V, ii, c);              // d_t A = V

                    // derivatives along one space axis k (it is the coordinate k+1)
                    for (int k = 0; k < 3; k++) {
                        int after[3]  = {i0, i1, i2};       // the point after, along axis k
                        int before[3] = {i0, i1, i2};       // the point before, along axis k
                        after[k]  = up[k];
                        before[k] = down[k];

                        double Aafter  = val(A, after, c);
                        double Abefore = val(A, before, c);
                        double Vafter  = val(V, after, c);
                        double Vbefore = val(V, before, c);

                        dAm[k+1][c]     = (Aafter - Abefore) / (2.0 * dq[k]);
                        dd[k+1][k+1][c] = (Aafter - 2.0 * Am[c] + Abefore) / (dq[k] * dq[k]);
                        dd[0][k+1][c]   = (Vafter - Vbefore) / (2.0 * dq[k]);
                        dd[k+1][0][c]   = dd[0][k+1][c];
                    }

                    // mixed derivatives d_k d_l A (two different space axes)
                    for (int k = 0; k < 3; k++) {
                        for (int l = k + 1; l < 3; l++) {
                            int pp[3] = {i0, i1, i2};       // after k, after l
                            int pm[3] = {i0, i1, i2};       // after k, before l
                            int mp[3] = {i0, i1, i2};       // before k, after l
                            int mm[3] = {i0, i1, i2};       // before k, before l
                            pp[k] = up[k];    pp[l] = up[l];
                            pm[k] = up[k];    pm[l] = down[l];
                            mp[k] = down[k];  mp[l] = up[l];
                            mm[k] = down[k];  mm[l] = down[l];

                            double mixed = (val(A, pp, c) - val(A, pm, c) - val(A, mp, c) + val(A, mm, c))
                                           / (4.0 * dq[k] * dq[l]);
                            dd[k+1][l+1][c] = mixed;
                            dd[l+1][k+1][c] = mixed;
                        }
                    }
                }

                // ---- 2c: field strength F_{rs} = d_r A_s - d_s A_r, and W^a_l = g^{ab} F_{bl} ----
                double F[4][4], W[4][4];
                for (int r = 0; r < 4; r++)
                    for (int s = 0; s < 4; s++)
                        F[r][s] = dAm[r][s] - dAm[s][r];

                for (int a = 0; a < 4; a++)
                    for (int l = 0; l < 4; l++) {
                        W[a][l] = 0.0;
                        for (int b = 0; b < 4; b++) W[a][l] += gi[a][b] * F[b][l];
                    }

                double FF = 0.0;                            // F_{rs} F^{rs}
                for (int a = 0; a < 4; a++)
                    for (int s = 0; s < 4; s++)
                        FF -= W[a][s] * W[s][a];

                // ---- 2c': dilaton and its derivatives (only in dilaton mode) ----
                double ph = 0.0, dp[4] = {0,0,0,0}, ddp[4][4];
                for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) ddp[a][b] = 0.0;
                if (dil) {
                    const SGrid& P = *phiP; const SGrid& Q = *PiP;
                    ph = P[i0][i1][i2];
                    dp[0] = Q[i0][i1][i2];
                    for (int k = 0; k < 3; k++) {
                        int after[3] = {i0, i1, i2}, before[3] = {i0, i1, i2};
                        after[k] = up[k]; before[k] = down[k];
                        double pa = P[after[0]][after[1]][after[2]], pb = P[before[0]][before[1]][before[2]];
                        double qa = Q[after[0]][after[1]][after[2]], qb = Q[before[0]][before[1]][before[2]];
                        dp[k+1] = (pa - pb) / (2.0 * dq[k]);
                        ddp[k+1][k+1] = (pa - 2.0 * ph + pb) / (dq[k] * dq[k]);
                        ddp[0][k+1] = ddp[k+1][0] = (qa - qb) / (2.0 * dq[k]);
                    }
                    for (int k = 0; k < 3; k++) for (int l = k + 1; l < 3; l++) {
                        int pp[3] = {i0,i1,i2}, pm[3] = {i0,i1,i2}, mp[3] = {i0,i1,i2}, mm[3] = {i0,i1,i2};
                        pp[k]=up[k];   pp[l]=up[l];
                        pm[k]=up[k];   pm[l]=down[l];
                        mp[k]=down[k]; mp[l]=up[l];
                        mm[k]=down[k]; mm[l]=down[l];
                        double mixed = (P[pp[0]][pp[1]][pp[2]] - P[pm[0]][pm[1]][pm[2]]
                                      - P[mp[0]][mp[1]][mp[2]] + P[mm[0]][mm[1]][mm[2]]) / (4.0 * dq[k] * dq[l]);
                        ddp[k+1][l+1] = ddp[l+1][k+1] = mixed;
                    }
                }

                if (dil) {
                    // d_a ln w, with w = h e^{-a phi}
                    double dlnw[4];
                    for (int a = 0; a < 4; a++) dlnw[a] = pd.dlnh[a] - acoup * dp[a];
                    std::array<double,4> rateA, rateV;
                    for (int c = 0; c < 4; c++) { rateA[c] = dAm[0][c]; rateV[c] = 0.0; }
                    for (int mu = 1; mu < 4; mu++) {
                        double s = 0.0;                 // nabla^a F_{a mu} without the d_t d_t A term
                        for (int a = 0; a < 4; a++)
                            for (int r = 0; r < 4; r++)
                                s += gi[a][r] * (dd[a][r][mu] - dd[a][mu][r]);
                        for (int l = 0; l < 4; l++) {
                            s -= Gc[l] * F[l][mu];
                            for (int a = 0; a < 4; a++) s -= Gam[l][a][mu] * W[a][l];
                        }
                        double J = 0.0;                 // (d^a ln w) F_{a mu}
                        for (int a = 0; a < 4; a++) J += dlnw[a] * W[a][mu];
                        rateV[mu] = -(s + J) / gi[0][0];
                    }
                    // box phi = g^{ab} d_a d_b phi - Gc^c d_c phi = (a/4) h e^{-a phi} F^2
                    double rest = 0.0;
                    for (int a = 0; a < 4; a++)
                        for (int b = 0; b < 4; b++)
                            if (!(a == 0 && b == 0)) rest += gi[a][b] * ddp[a][b];
                    for (int c = 0; c < 4; c++) rest -= Gc[c] * dp[c];
                    double src = 0.25 * acoup * h * std::exp(-acoup * ph) * FF;
                    (*dphiP)[i0][i1][i2] = dp[0];
                    (*dPiP)[i0][i1][i2]  = (src - rest) / gi[0][0];
                    dA[i0][i1][i2].setvalue(rateA);
                    dV[i0][i1][i2].setvalue(rateV);
                    continue;
                }

                // ---- 2d: the equation  c0 * (d_t d_t A_mu) + (h/2) * s + A_mu * kA = 0 ----
                std::array<double,4> rateA;                 // d_t A  = V
                std::array<double,4> rateV;                 // d_t V  = d_t d_t A
                for (int c = 0; c < 4; c++) {
                    rateA[c] = dAm[0][c];
                    rateV[c] = 0.0;                         // A_0 is not evolved, so d_t V_0 = 0
                }

                double c0 = 0.5 * h * gi[0][0];             // coefficient of d_t d_t A
                double kA = 0.25 * h * h * FF - Phil * box;

                if (std::fabs(c0) > 1e-300) {
                    for (int mu = 1; mu < 4; mu++) {
                        double s = 0.0;                     // everything except the d_t d_t A term
                        for (int a = 0; a < 4; a++)
                            for (int r = 0; r < 4; r++)
                                s += gi[a][r] * (dd[a][r][mu] - dd[a][mu][r]);
                        for (int l = 0; l < 4; l++) {
                            s -= Gc[l] * F[l][mu];
                            for (int a = 0; a < 4; a++)
                                s -= Gam[l][a][mu] * W[a][l];
                        }
                        rateV[mu] = -(0.5 * h * s + Am[mu] * kA) / c0;
                    }
                }

                dA[i0][i1][i2].setvalue(rateA);
                dV[i0][i1][i2].setvalue(rateV);
            }
        }
    }

    // ---------------- Step 3: points ON an open border ----------------
    // These points do not use the equation: their rate comes from the boundary condition.
    //   Dirichlet : the value stays the same            -> rate = 0
    //   Neumann   : the value follows the point inside  -> rate = rate of that point
    //   Outgoing  : only a wave leaving the box exists  -> d_t f = - c * d_f/d_outward
    //
    // A point can be on 1 border (a face), 2 borders (an edge) or 3 (a corner).
    // We do the faces first, then the edges, then the corners, so that a Neumann point
    // always copies a point whose rate is already final.
    // If several borders meet:  Dirichlet wins, then Neumann, then outgoing (the outgoing parts add up).
    for (int howMany = 1; howMany <= 3; howMany++) {
        for (int i0 = 0; i0 < n[0]; i0++) {
            for (int i1 = 0; i1 < n[1]; i1++) {
                for (int i2 = 0; i2 < n[2]; i2++) {

                    int ii[3] = {i0, i1, i2};

                    // on which axes is this point on an open border?
                    int borderAxis[3];
                    int count = 0;
                    for (int k = 0; k < 3; k++) {
                        if (!periodic[k] && (ii[k] == 0 || ii[k] == n[k] - 1)) {
                            borderAxis[count] = k;
                            count++;
                        }
                    }
                    if (count != howMany) continue;

                    // which rule applies?
                    bool dirichlet = false;
                    int neumannAxis = -1;
                    for (int q = 0; q < count; q++) {
                        int k = borderAxis[q];
                        int side = (ii[k] == 0) ? 0 : 1;        // 0 = lower face, 1 = upper face
                        if (bc[k][side] == BC_DIRICHLET) dirichlet = true;
                        if (bc[k][side] == BC_NEUMANN && neumannAxis < 0) neumannAxis = k;
                    }

                    // component 0 (A_0) is not evolved: d_t A_0 = V_0 and d_t V_0 = 0
                    std::array<double,4> rateA = {val(V, ii, 0), 0.0, 0.0, 0.0};
                    std::array<double,4> rateV = {0.0, 0.0, 0.0, 0.0};

                    if (dirichlet) {
                        // nothing to do: the rates of components 1..3 stay 0

                    } else if (neumannAxis >= 0) {
                        int k = neumannAxis;
                        int inside[3] = {i0, i1, i2};           // the point just inside
                        inside[k] = (ii[k] == 0) ? 1 : n[k] - 2;
                        std::array<double,4> insideA = dA[inside[0]][inside[1]][inside[2]].getvalue();
                        std::array<double,4> insideV = dV[inside[0]][inside[1]][inside[2]].getvalue();
                        for (int c = 1; c < 4; c++) {
                            rateA[c] = insideA[c];
                            rateV[c] = insideV[c];
                        }

                    } else {
                        // outgoing wave: needs the inverse metric at this point
                        const double (&gi)[4][4] = cache[((size_t)i0 * n[1] + i1) * n[2] + i2].gi;

                        for (int q = 0; q < count; q++) {
                            int k = borderAxis[q];

                            // speed of light along axis k (the coordinate k+1)
                            double speed = std::sqrt(std::max(0.0, -gi[k+1][k+1] / gi[0][0]));

                            // the next two points going inside
                            int step = (ii[k] == 0) ? 1 : -1;
                            int in1[3] = {i0, i1, i2};
                            int in2[3] = {i0, i1, i2};
                            in1[k] = ii[k] + step;
                            in2[k] = ii[k] + 2 * step;

                            for (int c = 1; c < 4; c++) {
                                rateA[c] -= speed * dOutward(val(A, ii, c), val(A, in1, c), val(A, in2, c), dq[k]);
                                rateV[c] -= speed * dOutward(val(V, ii, c), val(V, in1, c), val(V, in2, c), dq[k]);
                            }
                        }
                    }

                    dA[i0][i1][i2].setvalue(rateA);
                    dV[i0][i1][i2].setvalue(rateV);

                    if (dil) {
                        double rp = 0.0, rq = 0.0;              // rates of phi and Pi on the border
                        if (dirichlet) {
                        } else if (neumannAxis >= 0) {
                            int k = neumannAxis;
                            int inside[3] = {i0, i1, i2};
                            inside[k] = (ii[k] == 0) ? 1 : n[k] - 2;
                            rp = (*dphiP)[inside[0]][inside[1]][inside[2]];
                            rq = (*dPiP)[inside[0]][inside[1]][inside[2]];
                        } else {
                            const double (&gi)[4][4] = cache[((size_t)i0 * n[1] + i1) * n[2] + i2].gi;
                            for (int q = 0; q < count; q++) {
                                int k = borderAxis[q];
                                double speed = std::sqrt(std::max(0.0, -gi[k+1][k+1] / gi[0][0]));
                                int step = (ii[k] == 0) ? 1 : -1;
                                int in1[3] = {i0, i1, i2}, in2[3] = {i0, i1, i2};
                                in1[k] = ii[k] + step;  in2[k] = ii[k] + 2 * step;
                                const SGrid& P = *phiP; const SGrid& Q = *PiP;
                                rp -= speed * dOutward(P[i0][i1][i2], P[in1[0]][in1[1]][in1[2]], P[in2[0]][in2[1]][in2[2]], dq[k]);
                                rq -= speed * dOutward(Q[i0][i1][i2], Q[in1[0]][in1[1]][in1[2]], Q[in2[0]][in2[1]][in2[2]], dq[k]);
                            }
                        }
                        (*dphiP)[i0][i1][i2] = rp;
                        (*dPiP)[i0][i1][i2]  = rq;
                    }
                }
            }
        }
    }
}

void systemA(const PointGrid& pts, const AGrid& A, const AGrid& V, const int bc[3][2],
             const christ& gam, const met& g, const met& ginv,
             const func& Phi, const func& phihat, double t, AGrid& dA, AGrid& dV)
{
    systemImpl(pts, A, V, nullptr, nullptr, bc, gam, g, ginv, Phi, phihat, t, dA, dV, nullptr, nullptr);
}

void systemAphi(const PointGrid& pts, const AGrid& A, const AGrid& V, const SGrid& phi, const SGrid& Pi,
                const int bc[3][2], const christ& gam, const met& g, const met& ginv, const func& Phi, double t,
                AGrid& dA, AGrid& dV, SGrid& dphi, SGrid& dPi)
{
    systemImpl(pts, A, V, &phi, &Pi, bc, gam, g, ginv, Phi, constant(0.0), t, dA, dV, &dphi, &dPi);
}

}