#include "field_solver.h"

namespace rgsim {

// ---------------------------------------------------------------- globals
int  evolveA0 = 0;
func A0 = constant(0.0);

std::vector<PointData> pointData;
int    pdGridSize = -1;
bool   pdBuilt = false, pdAnalysed = false, pdTimeDep = false;
std::array<bool,3> pdDep = {true, true, true};
double pdTime = 0.0;
met    pdInvg;

// ---------------------------------------------------------------- the grid
int Grid::size() const { return n[0] * n[1] * n[2]; }
int Grid::index(int i0, int i1, int i2) const { return (i0 * n[1] + i1) * n[2] + i2; }

std::array<dual,DIM> Grid::coords(int i0, int i1, int i2, double t) const {
    std::array<dual,DIM> x;
    for (int a = 0; a < DIM; ++a) x[a] = 0.0;                                      // extra dimension = 0
    x[0] = t;
    x[1] = q0[0] + i0 * dq[0];
    x[2] = q0[1] + i1 * dq[1];
    x[3] = q0[2] + i2 * dq[2];
    return x;
}

// State at t = 0 from the functions A[1..3](x): A_j and d_t A_j; A_0 from the function A0.
state makeInitialState(const Grid& G, const fvec& A)
{
    const int P = G.size();
    state Q(NB * P, 0.0);
    for (int i0 = 0; i0 < G.n[0]; ++i0)
      for (int i1 = 0; i1 < G.n[1]; ++i1)
        for (int i2 = 0; i2 < G.n[2]; ++i2) {
            auto x = G.coords(i0, i1, i2);
            int p = G.index(i0, i1, i2);
            for (int j = 0; j < 3; ++j) {
                Q[j * P + p]       = A[j + 1](x).val;                // A_j at t = 0
                Q[(3 + j) * P + p] = partial(A[j + 1], x, 0).val;    // dA_j/dt at t = 0
            }
            Q[6 * P + p] = A0(x).val;                            // A_0 at t = 0 (evolved if evolveA0 >= 1)
        }
    return Q;
}

// data at grid point p (use this, not pointData[p])
const PointData& pointAt(int p)
{
    const int n1 = grid.n[1], n2 = grid.n[2];
    const int i2 = p % n2, i1 = (p / n2) % n1, i0 = p / (n1 * n2);
    const int m1 = pdDep[1] ? n1 : 1, m2 = pdDep[2] ? n2 : 1;
    return pointData[((pdDep[0] ? i0 : 0) * m1 + (pdDep[1] ? i1 : 0)) * m2 + (pdDep[2] ? i2 : 0)];
}

// true if g, Phi, phihat or A0 has a non-zero derivative along one of the coordinates kFirst..kLast,
// tested at the grid corners and centre
bool dependsOn(const Grid& G, double t, int kFirst, int kLast)
{
    std::vector<std::array<dual,DIM>> pts;
    for (int i0 : {0, G.n[0] - 1})
      for (int i1 : {0, G.n[1] - 1})
        for (int i2 : {0, G.n[2] - 1})
            pts.push_back(G.coords(i0, i1, i2, t));
    pts.push_back(G.coords(G.n[0] / 2, G.n[1] / 2, G.n[2] / 2, t));

    for (const auto& x : pts)
        for (int k = kFirst; k <= kLast; ++k) {
            if (partial(Phi, x, k).val != 0.0 || partial(phihat, x, k).val != 0.0 || partial(A0, x, k).val != 0.0) return true;
            for (int a = 0; a < 4; ++a)
                for (int b = a; b < 4; ++b)
                    if (partial(g[a][b], x, k).val != 0.0) return true;
        }
    return false;
}

void ensurePointData(double t)
{
    const Grid& G = grid;
    if (pdGridSize != G.size()) {                       // new grid: start again
        pdGridSize = G.size();
        pdBuilt = pdAnalysed = pdTimeDep = false;
    }
    if (!pdAnalysed) {
        pdInvg = inverseMet(g);
        for (int k = 0; k < 3; ++k) pdDep[k] = dependsOn(G, t, k + 1, k + 1);
        pdAnalysed = true;
    }
    if (!pdTimeDep) pdTimeDep = dependsOn(G, t, 0, 0);
    if (pdBuilt && (!pdTimeDep || t == pdTime)) return;

    const func hF = Phi * Phi;
    auto fill = [&](PointData& pd, const std::array<dual,DIM>& x) {
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b)
                pd.ginv[a][b] = pdInvg[a][b](x).val;

        for (int l = 0; l < 4; ++l)
            for (int mu = 0; mu < 4; ++mu)
                for (int nu = mu; nu < 4; ++nu) {
                    dual sum = 0.0;
                    for (int m = 0; m < 4; ++m)
                        sum += pdInvg[l][m](x) * 0.5 *
                               ( partial(g[m][nu], x, mu) + partial(g[m][mu], x, nu) - partial(g[mu][nu], x, m) );
                    pd.Gam[l][mu][nu] = pd.Gam[l][nu][mu] = sum.val;
                }
        {
            Eigen::Matrix4d M;
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b) M(a, b) = g[a][b](x).val;
            const double sg = std::sqrt(std::fabs(M.determinant()));
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b) pd.U[a][b] = sg * pd.ginv[a][b];
        }
        for (int l = 0; l < 4; ++l) {
            double sum = 0.0;
            for (int a = 0; a < 4; ++a)
                for (int r = 0; r < 4; ++r) sum += pd.ginv[a][r] * pd.Gam[l][a][r];
            pd.Gc[l] = sum;
        }

        pd.h    = hF(x).val;
        pd.Phil = Phi(x).val;

        double box = 0.0;
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b) {
                func dphb = [b](const std::array<dual,DIM>& y) { return partial(phihat, y, b); };
                box += pd.ginv[a][b] * partialFD(dphb, x, a).val;
            }
        pd.box = box;

        for (int a = 0; a < 4; ++a) {
            func dA = [a](const std::array<dual,DIM>& y) { return partial(A0, y, a); };
            pd.dA0[a] = dA(x).val;
            for (int b = 0; b < 4; ++b) pd.ddA0[a][b] = partial(dA, x, b).val;
        }
    };

    const int m0 = pdDep[0] ? G.n[0] : 1, m1 = pdDep[1] ? G.n[1] : 1, m2 = pdDep[2] ? G.n[2] : 1;
    pointData.assign(m0 * m1 * m2, PointData());
    for (int i0 = 0; i0 < m0; ++i0)
      for (int i1 = 0; i1 < m1; ++i1)
        for (int i2 = 0; i2 < m2; ++i2)
            fill(pointData[(i0 * m1 + i1) * m2 + i2],
                 G.coords(pdDep[0] ? i0 : G.n[0] / 2, pdDep[1] ? i1 : G.n[1] / 2, pdDep[2] ? i2 : G.n[2] / 2, t));
    pdBuilt = true;
    pdTime  = t;
}

// ---------------------------------------------------------------- the equation of motion
// dQ = dQ/dt at time t.
void rhsField(const state& Q, double t, state& dQ)
{
    dQ.assign(Q.size(), 0.0);
    const Grid& G = grid;
    const int Pt = G.size();

    ensurePointData(t);

    const int n0 = G.n[0], n1 = G.n[1], n2 = G.n[2];
    const int st0 = n1 * n2, st1 = n2;               // flat-index step along x0, x1 (x2 has step 1)
    const double idq0 = 1.0 / G.dq[0], idq1 = 1.0 / G.dq[1], idq2 = 1.0 / G.dq[2];
    const double sq[3] = { idq0 * idq0, idq1 * idq1, idq2 * idq2 };                        // 1/dq^2
    const double mx[3] = { 0.25 * idq0 * idq1, 0.25 * idq0 * idq2, 0.25 * idq1 * idq2 };   // 1/(4 dq dq')
    const bool per[3] = { G.periodic[0], G.periodic[1], G.periodic[2] };

    // points on the faces of a non-periodic axis are not evolved here (see the outgoing-wave condition below)
    const int lo0 = per[0] ? 0 : 1, hi0 = per[0] ? n0 : n0 - 1;
    const int lo1 = per[1] ? 0 : 1, hi1 = per[1] ? n1 : n1 - 1;
    const int lo2 = per[2] ? 0 : 1, hi2 = per[2] ? n2 : n2 - 1;

    const double* __restrict q = Q.data();
    double* __restrict out = dQ.data();
    const PointData* pdp = pointData.data();

    // ---- Lorenz gauge in conservation form:  d_mu ( U^{mu nu} A_nu ) = 0,  U = sqrt(-g) g^{mu nu}  (static metric),
    //         U^{00} d_t A_0 = - U^{0i} v_i - d_i Flux^i,   Flux^i = U^{i0} A_0 + U^{ij} A_j.
    //      d_t A_0 is computed at EVERY point (first pass) because the second pass needs d_k d_t A_0.
    //      The divergence is taken on the flux itself (not expanded with Christoffel symbols): this keeps the discrete
    //      operator symmetric, which is what makes it stable where the light speed goes to 0 (near the horizon).
    //      On the faces of a non-periodic axis the derivative is one-sided (second order). ----
    static std::vector<double> rate0, flux;
    if (evolveA0 >= 1) {
        rate0.assign(Pt, 0.0);
        flux.assign(3 * Pt, 0.0);
        const double* a0q = q + 6 * Pt;
        const int nnn[3] = { n0, n1, n2 }, sss[3] = { st0, st1, 1 };
        const double idqq[3] = { idq0, idq1, idq2 };
        #pragma omp parallel for schedule(static)
        for (int p = 0; p < Pt; ++p) {
            const PointData& pd = pointAt(p);
            for (int i = 1; i < 4; ++i)
                flux[(i - 1) * Pt + p] = pd.U[i][0] * a0q[p] + pd.U[i][1] * q[p] + pd.U[i][2] * q[Pt + p] + pd.U[i][3] * q[2 * Pt + p];
        }
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i0 = 0; i0 < n0; ++i0)
          for (int i1 = 0; i1 < n1; ++i1)
            for (int i2 = 0; i2 < n2; ++i2) {
                const int p = (i0 * n1 + i1) * n2 + i2;
                const int ii[3] = { i0, i1, i2 };
                const PointData& pd = pointAt(p);
                if (std::abs(pd.U[0][0]) < 1e-300) continue;
                auto d1 = [&](const double* a, int k) {        // d_k a: central (wrapping if periodic), one-sided on a face
                    const int st = sss[k], nk = nnn[k], ik = ii[k];
                    if (!per[k] && ik == 0)      return (-3.0 * a[p] + 4.0 * a[p + st] - a[p + 2 * st]) * 0.5 * idqq[k];
                    if (!per[k] && ik == nk - 1) return ( 3.0 * a[p] - 4.0 * a[p - st] + a[p - 2 * st]) * 0.5 * idqq[k];
                    const int up = ik < nk - 1 ? st : -(nk - 1) * st, dn = ik > 0 ? -st : (nk - 1) * st;
                    return (a[p + up] - a[p + dn]) * 0.5 * idqq[k];
                };
                double rhs = 0.0;
                for (int i = 1; i < 4; ++i) rhs += pd.U[0][i] * q[(2 + i) * Pt + p] + d1(flux.data() + (i - 1) * Pt, i - 1);
                rate0[p] = -rhs / pd.U[0][0];
            }
    }
    const bool d0 = pdDep[0], d1 = pdDep[1], d2 = pdDep[2];
    const int  pm1 = d1 ? n1 : 1, pm2 = d2 ? n2 : 1, pinc2 = d2 ? 1 : 0;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i0 = lo0; i0 < hi0; ++i0)
      for (int i1 = lo1; i1 < hi1; ++i1) {
        // neighbour offsets (a face point is only reached on a periodic axis, where the stencil wraps around)
        const int    a0p = i0 < n0 - 1 ? st0 : -(n0 - 1) * st0, a0m = i0 > 0 ? -st0 : (n0 - 1) * st0;
        const int    a1p = i1 < n1 - 1 ? st1 : -(n1 - 1) * st1, a1m = i1 > 0 ? -st1 : (n1 - 1) * st1;
        const int    row = i0 * st0 + i1 * st1;
        const int    pbase = ((d0 ? i0 : 0) * pm1 + (d1 ? i1 : 0)) * pm2;

        for (int i2 = lo2; i2 < hi2; ++i2) {
            const int    p   = row + i2;
            const int    a2p = i2 < n2 - 1 ? 1 : -(n2 - 1), a2m = i2 > 0 ? -1 : n2 - 1;
            const int    ap[3] = { a0p, a1p, a2p }, am[3] = { a0m, a1m, a2m };
            const double f[3]  = { 0.5 * idq0, 0.5 * idq1, 0.5 * idq2 };

            const PointData& pd = pdp[pbase + i2 * pinc2];
            const double (&ginv)[4][4]   = pd.ginv;
            const double (&Gam)[4][4][4] = pd.Gam;

            // ---- A_i, v_i = d_t A_i and their derivatives: dA[a][i] = d_a A_i, dd[a][b][i] = d_a d_b A_i.
            //      Index 0 = time, 1..3 = space.  dd[0][0][i] (the acceleration) is the unknown and stays 0 here. ----
            double A[4] = {0, 0, 0, 0};
            double dA[4][4] = {};
            double dd[4][4][4] = {};
            for (int c = 0; c < 3; ++c) {
                const double* a = q + c * Pt;        // A_{c+1}
                const double* v = q + (3 + c) * Pt;  // d_t A_{c+1}
                const int i = c + 1;
                const double ac = a[p];
                A[i]     = ac;
                dA[0][i] = v[p];
                for (int k = 0; k < 3; ++k) {
                    dA[k+1][i]       = (a[p + ap[k]] - a[p + am[k]]) * f[k];
                    dd[0][k+1][i]    = dd[k+1][0][i] = (v[p + ap[k]] - v[p + am[k]]) * f[k];
                    dd[k+1][k+1][i]  = (a[p + ap[k]] - 2.0 * ac + a[p + am[k]]) * sq[k];
                }
                dd[1][2][i] = dd[2][1][i] = (a[p + a0p + a1p] - a[p + a0p + a1m] - a[p + a0m + a1p] + a[p + a0m + a1m]) * mx[0];
                dd[1][3][i] = dd[3][1][i] = (a[p + a0p + a2p] - a[p + a0p + a2m] - a[p + a0m + a2p] + a[p + a0m + a2m]) * mx[1];
                dd[2][3][i] = dd[3][2][i] = (a[p + a1p + a2p] - a[p + a1p + a2m] - a[p + a1m + a2p] + a[p + a1m + a2m]) * mx[2];
            }
            if (evolveA0 == 2) {
                // A_0 from the state (block 6); d_t A_0 and d_k d_t A_0 from the Lorenz-gauge rate computed above
                const double* a = q + 6 * Pt;
                A[0] = a[p];
                dA[0][0] = rate0[p];
                for (int k = 0; k < 3; ++k) {
                    dA[k+1][0]       = (a[p + ap[k]] - a[p + am[k]]) * f[k];
                    dd[0][k+1][0]    = dd[k+1][0][0] = (rate0[p + ap[k]] - rate0[p + am[k]]) * f[k];
                    dd[k+1][k+1][0]  = (a[p + ap[k]] - 2.0 * a[p] + a[p + am[k]]) * sq[k];
                }
                dd[1][2][0] = dd[2][1][0] = (a[p + a0p + a1p] - a[p + a0p + a1m] - a[p + a0m + a1p] + a[p + a0m + a1m]) * mx[0];
                dd[1][3][0] = dd[3][1][0] = (a[p + a0p + a2p] - a[p + a0p + a2m] - a[p + a0m + a2p] + a[p + a0m + a2m]) * mx[1];
                dd[2][3][0] = dd[3][2][0] = (a[p + a1p + a2p] - a[p + a1p + a2m] - a[p + a1m + a2p] + a[p + a1m + a2m]) * mx[2];
            } else {
                // the prescribed A_0
                for (int a = 0; a < 4; ++a) {
                    dA[a][0] = pd.dA0[a];
                    for (int b = 0; b < 4; ++b) dd[a][b][0] = pd.ddA0[a][b];
                }
            }

            // ---- F_{rs} = d_r A_s - d_s A_r,  W^a_l = g^{ab} F_{bl},  FF = F_{rs}F^{rs} = -tr(W W) ----
            double F[4][4], W[4][4];
            for (int r = 0; r < 4; ++r)
                for (int s = 0; s < 4; ++s)
                    F[r][s] = dA[r][s] - dA[s][r];
            double FF = 0.0;
            for (int a = 0; a < 4; ++a)
                for (int l = 0; l < 4; ++l)
                    W[a][l] = ginv[a][0] * F[0][l] + ginv[a][1] * F[1][l] + ginv[a][2] * F[2][l] + ginv[a][3] * F[3][l];
            for (int a = 0; a < 4; ++a)
                for (int s = 0; s < 4; ++s)
                    FF -= W[a][s] * W[s][a];

            const double h = pd.h, Phil = pd.Phil, box = pd.box;
            const double c0 = 0.5 * h * ginv[0][0];  // coefficient of d_t^2 A_j in the equation

            for (int j = 1; j < 4; ++j) out[(j-1) * Pt + p] = dA[0][j];     // dA_j/dt = v_j
            if (evolveA0 >= 1) out[6 * Pt + p] = rate0[p];                        // dA_0/dt (Lorenz gauge)
            if (std::abs(c0) < 1e-300) continue;     // acceleration stays 0

            // ---- the equation for mu = 1..3:  c0 d_t^2 A_mu + (h/2) s + A_mu kA = 0,
            //      s = everything else (d_t^2 A set to 0 in dd) ----
            const double ic0 = -1.0 / c0;
            const double kA  = 0.25 * h * h * FF - Phil * box;
            for (int mu = 1; mu < 4; ++mu) {
                double s = 0.0;
                for (int a = 0; a < 4; ++a)
                    for (int r = 0; r < 4; ++r) s += ginv[a][r] * (dd[a][r][mu] - dd[a][mu][r]);
                for (int l = 0; l < 4; ++l) {
                    s -= pd.Gc[l] * F[l][mu];
                    for (int a = 0; a < 4; ++a) s -= Gam[l][a][mu] * W[a][l];
                }
                out[(2 + mu) * Pt + p] = (0.5 * h * s + A[mu] * kA) * ic0;  // d(v_mu)/dt = d_t^2 A_mu
            }
        }
      }

    // ---- outgoing-wave (Sommerfeld) condition on the faces of the non-periodic axes ----
    // Only a wave leaving the box exists there:  d_t f + s c_k d_k f = 0,  s = +1 on the upper face, -1 on the lower,
    // c_k = sqrt(-g^{kk}/g^{00}) = local coordinate speed of light along axis k (0 at a horizon: the face stays put).
    // Applied to every component of the state (to state - background if a background is given).  On an edge or
    // corner the conditions of the axes involved add up.  d_k f is one-sided, second order.
    const int    nn[3]  = { n0, n1, n2 };
    const int    stp[3] = { st0, st1, 1 };
    const double idq[3] = { idq0, idq1, idq2 };
    for (int i0 = 0; i0 < n0; ++i0)
      for (int i1 = 0; i1 < n1; ++i1)
        for (int i2 = 0; i2 < n2; ++i2) {
            const int ii[3] = { i0, i1, i2 };
            bool onFace = false;
            for (int k = 0; k < 3; ++k) if (!per[k] && (ii[k] == 0 || ii[k] == nn[k] - 1)) onFace = true;
            if (!onFace) continue;
            const int p = (i0 * n1 + i1) * n2 + i2;
            const PointData& pd = pointAt(p);
            for (int comp = 0; comp < (evolveA0 >= 1 ? NB : 6); ++comp) {
                const double* fq = Q.data() + comp * Pt;
                const double* b0 = G.bg ? G.bg->data() + comp * Pt : nullptr;
                auto f = [&](int idx) { return b0 ? fq[idx] - b0[idx] : fq[idx]; };   // perturbation
                double rate = 0.0;
                for (int k = 0; k < 3; ++k) {
                    if (per[k] || nn[k] < 3 || (ii[k] != 0 && ii[k] != nn[k] - 1)) continue;
                    const double ck  = std::sqrt(std::max(0.0, -pd.ginv[k+1][k+1] / pd.ginv[0][0]));
                    const int    sgn = ii[k] == 0 ? -1 : +1;           // outward direction along axis k
                    const int    st  = stp[k] * (-sgn);                // step towards the interior
                    const double dout = -sgn * (-3.0 * f(p) + 4.0 * f(p + st) - f(p + 2 * st)) * 0.5 * idq[k];   // derivative along +k
                    rate -= sgn * ck * dout;                           // d_t f = - s c d_k f
                }
                dQ[comp * Pt + p] = rate;
            }
        }
}

}  // namespace rgsim
