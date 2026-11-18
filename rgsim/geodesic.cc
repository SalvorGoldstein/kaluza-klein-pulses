#include "geodesic.h"

namespace rgsim {

double solveU0(const std::array<double,DIM>& pos, const spatial& us,
               const met& g, double kappa)
{
    std::array<dual,DIM> x;
    for (int a = 0; a < DIM; ++a) x[a] = pos[a];

    double gm[DIM][DIM];
    for (int a = 0; a < DIM; ++a)
        for (int b = 0; b < DIM; ++b)
            gm[a][b] = g[a][b](x).val;

    double u[DIM];
    u[0] = 0.0;
    for (int i = 1; i < DIM; ++i) u[i] = us[i - 1];

    double A = gm[0][0];
    double B = 0.0;
    double C = -kappa;
    for (int i = 1; i < DIM; ++i) {
        B += 2.0 * gm[0][i] * u[i];
        for (int j = 1; j < DIM; ++j)
            C += gm[i][j] * u[i] * u[j];
    }

    double D = B * B - 4.0 * A * C;
    if (D < 0.0) return NAN;
    return (-B + std::sqrt(D)) / (2.0 * A);
}

double timelikeU(const std::array<double,DIM>& pos, const spatial& us, const met& g)
{
    return solveU0(pos, us, g, constants::c * constants::c);
}

double lightlikeU(const std::array<double,DIM>& pos, const spatial& us, const met& g)
{
    double u0 = solveU0(pos, us, g, 0.0);
    if (std::isnan(u0))
        std::cerr << "No lightlike solution for k^0 at this point/direction!" << std::endl;
    return u0;
}

void geo(double (&q)[2*DIM], double tau, double (&qp)[2*DIM], int n, const christ& Gam)
{
    double u[DIM];
    std::array<dual,DIM> x;
    for (int i = 0; i < DIM; ++i) {
        x[i] = q[i];
        u[i] = q[DIM + i];
    }

    for (int i = 0; i < DIM; ++i)
        qp[i] = u[i];

    for (int i = 0; i < DIM; ++i) {
        double sum = 0.0;
        for (int k = 0; k < DIM; ++k)
            for (int l = 0; l < DIM; ++l) {
                dual gam = Gam[i][k][l](x);
                sum += gam.val * u[k] * u[l];
            }
        qp[DIM + i] = -sum;
    }
}



spatial filled(double v)
{
    spatial s;
    s.fill(v);
    return s;
}

void geodesics(const spatial& xs, const spatial& us,
               int steps, double dtau,
               const std::string& filename,
               const met& g, const christ& Gam,
               const rhs_fn& sd,
               bool type,
               const spatial& xmax,
               const spatial& xmin)
{
    const int n = 2 * DIM;
    std::ofstream fich(filename);
    fich << std::scientific << std::setprecision(10);

    double q[2*DIM] = {};
    std::array<double,DIM> pos = {};
    for (int i = 1; i < DIM; ++i) {
        q[i]       = xs[i - 1];
        q[DIM + i] = us[i - 1];
        pos[i]     = xs[i - 1];
    }

    double u0 = type ? timelikeU(pos, us, g) : lightlikeU(pos, us, g);
    if (!std::isfinite(u0)) {
        std::cerr << "invalid initial velocity" << std::endl;
        return;
    }
    q[DIM] = u0;

    double tau = 0.0;
    for (int step = 0; step < steps; ++step) {
        bool stop = false;
        for (int i = 1; i < DIM; ++i)
            if (!std::isfinite(q[i]) || q[i] < xmin[i - 1] || q[i] > xmax[i - 1]) { stop = true; break; }
        if (stop) break;

        for (int i = 0; i < DIM; ++i)
            fich << q[i] << (i + 1 < DIM ? " " : "\n");
        rk4(sd, q, tau, dtau, n, Gam);
        tau += dtau;
    }
}

}
