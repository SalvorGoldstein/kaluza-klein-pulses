
#include "systemsolver.h"
 
namespace rgsim {
 
// RK4 in words. We want to go from y (at time t) to y at time t+dt, with  d_t y = f(y, t).
//    k1 = f(y,            t)
//    k2 = f(y + dt/2 * k1, t + dt/2)
//    k3 = f(y + dt/2 * k2, t + dt/2)
//    k4 = f(y + dt   * k3, t + dt)
//    y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
// Here y is the pair (A, V), and f is the function "rhs" that gives (dA, dV).
//
// To be fast we do two things:
//   1) we do NOT create new grids at each step: the work grids below are created once and reused,
//   2) we do not keep k1..k4: after each stage we add  (weight * k)  into one running sum "acc",
//      and in the same loop we prepare the input of the next stage. So the grids are read only once per stage.
void rk4A(const RhsA& rhs, AGrid& A, AGrid& V, double t, double dt)
{
    const int n0 = A.size();
    const int n1 = A[0].size();
    const int n2 = A[0][0].size();
 
    // work grids: k (the slope of the current stage), acc (the running sum), tmp (input of the next stage)
    static AGrid kA, kV, accA, accV, tmpA, tmpV;
    if (kA.empty() || (int)kA.size() != n0 || (int)kA[0].size() != n1 || (int)kA[0][0].size() != n2) {
        kA = A;   kV = A;               // a copy gives the right shape
        accA = A; accV = A;
        tmpA = A; tmpV = A;
    }
 
    const double tStage[4] = { t, t + 0.5 * dt, t + 0.5 * dt, t + dt };   // time of each stage
    const double step[4]   = { 0.5 * dt, 0.5 * dt, dt, 0.0 };             // input of next stage = y + step * k
    const double weight[4] = { 1.0, 2.0, 2.0, 1.0 };                      // weight of each k in the final sum
 
    for (int s = 0; s < 4; s++) {
 
        // the slope k of this stage (stage 0 starts from y itself, the others from tmp)
        if (s == 0) rhs(A, V, tStage[s], kA, kV);
        else        rhs(tmpA, tmpV, tStage[s], kA, kV);
 
        // one pass over the grid: update the sum, and prepare the next stage (or finish)
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i0 = 0; i0 < n0; i0++) {
            for (int i1 = 0; i1 < n1; i1++) {
                for (int i2 = 0; i2 < n2; i2++) {
 
                    std::array<double,4> ka = kA[i0][i1][i2].getvalue();
                    std::array<double,4> kv = kV[i0][i1][i2].getvalue();
                    std::array<double,4> sa, sv;            // running sums at this point
 
                    if (s == 0) {
                        for (int c = 0; c < 4; c++) { sa[c] = weight[s] * ka[c]; sv[c] = weight[s] * kv[c]; }
                    } else {
                        std::array<double,4> oldA = accA[i0][i1][i2].getvalue();
                        std::array<double,4> oldV = accV[i0][i1][i2].getvalue();
                        for (int c = 0; c < 4; c++) {
                            sa[c] = oldA[c] + weight[s] * ka[c];
                            sv[c] = oldV[c] + weight[s] * kv[c];
                        }
                    }
 
                    std::array<double,4> ya = A[i0][i1][i2].getvalue();
                    std::array<double,4> yv = V[i0][i1][i2].getvalue();
                    std::array<double,4> na, nv;
 
                    if (s < 3) {
                        accA[i0][i1][i2].setvalue(sa);
                        accV[i0][i1][i2].setvalue(sv);
                        for (int c = 0; c < 4; c++) {
                            na[c] = ya[c] + step[s] * ka[c];
                            nv[c] = yv[c] + step[s] * kv[c];
                        }
                        tmpA[i0][i1][i2].setvalue(na);      // input of the next stage
                        tmpV[i0][i1][i2].setvalue(nv);
                    } else {
                        for (int c = 0; c < 4; c++) {
                            na[c] = ya[c] + dt / 6.0 * sa[c];
                            nv[c] = yv[c] + dt / 6.0 * sv[c];
                        }
                        A[i0][i1][i2].setvalue(na);         // final result, written in place
                        V[i0][i1][i2].setvalue(nv);
                    }
                }
            }
        }
    }
}
 

// RK4 for (A, V, phi, Pi). Same scheme as rk4A, written with one pass per stage over all four grids.
void rk4Aphi(const RhsAphi& rhs, AGrid& A, AGrid& V, SGrid& phi, SGrid& Pi, double t, double dt)
{
    const int n0 = A.size(), n1 = A[0].size(), n2 = A[0][0].size();
    static AGrid kA, kV, accA, accV, tmpA, tmpV;
    static SGrid kp, kq, accp, accq, tmpp, tmpq;
    if (kA.empty() || (int)kA.size() != n0 || (int)kA[0].size() != n1 || (int)kA[0][0].size() != n2) {
        kA = accA = tmpA = A;  kV = accV = tmpV = A;
        kp = accp = tmpp = phi;  kq = accq = tmpq = phi;
    }
    const double tStage[4] = { t, t + 0.5 * dt, t + 0.5 * dt, t + dt };
    const double step[4]   = { 0.5 * dt, 0.5 * dt, dt, 0.0 };
    const double weight[4] = { 1.0, 2.0, 2.0, 1.0 };

    for (int s = 0; s < 4; s++) {
        if (s == 0) rhs(A, V, phi, Pi, tStage[s], kA, kV, kp, kq);
        else        rhs(tmpA, tmpV, tmpp, tmpq, tStage[s], kA, kV, kp, kq);

        #pragma omp parallel for collapse(2) schedule(static)
        for (int i0 = 0; i0 < n0; i0++)
        for (int i1 = 0; i1 < n1; i1++)
        for (int i2 = 0; i2 < n2; i2++) {
            std::array<double,4> ka = kA[i0][i1][i2].getvalue(), kv = kV[i0][i1][i2].getvalue();
            std::array<double,4> sa, sv;
            double sp, sq;
            const double kpv = kp[i0][i1][i2], kqv = kq[i0][i1][i2];
            if (s == 0) {
                for (int c = 0; c < 4; c++) { sa[c] = ka[c]; sv[c] = kv[c]; }
                sp = kpv; sq = kqv;
            } else {
                std::array<double,4> oa = accA[i0][i1][i2].getvalue(), ov = accV[i0][i1][i2].getvalue();
                for (int c = 0; c < 4; c++) { sa[c] = oa[c] + weight[s] * ka[c]; sv[c] = ov[c] + weight[s] * kv[c]; }
                sp = accp[i0][i1][i2] + weight[s] * kpv;
                sq = accq[i0][i1][i2] + weight[s] * kqv;
            }
            std::array<double,4> ya = A[i0][i1][i2].getvalue(), yv = V[i0][i1][i2].getvalue();
            std::array<double,4> na, nv;
            const double yp = phi[i0][i1][i2], yq = Pi[i0][i1][i2];
            if (s < 3) {
                accA[i0][i1][i2].setvalue(sa); accV[i0][i1][i2].setvalue(sv);
                accp[i0][i1][i2] = sp; accq[i0][i1][i2] = sq;
                for (int c = 0; c < 4; c++) { na[c] = ya[c] + step[s] * ka[c]; nv[c] = yv[c] + step[s] * kv[c]; }
                tmpA[i0][i1][i2].setvalue(na); tmpV[i0][i1][i2].setvalue(nv);
                tmpp[i0][i1][i2] = yp + step[s] * kpv;
                tmpq[i0][i1][i2] = yq + step[s] * kqv;
            } else {
                for (int c = 0; c < 4; c++) { na[c] = ya[c] + dt / 6.0 * sa[c]; nv[c] = yv[c] + dt / 6.0 * sv[c]; }
                A[i0][i1][i2].setvalue(na); V[i0][i1][i2].setvalue(nv);
                phi[i0][i1][i2] = yp + dt / 6.0 * sp;
                Pi[i0][i1][i2]  = yq + dt / 6.0 * sq;
            }
        }
    }
}

}
