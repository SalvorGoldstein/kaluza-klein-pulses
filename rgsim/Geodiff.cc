#include "Geodiff.h"

namespace rgsim {

dual partial(const func& fn,
             std::array<dual,DIM> x,
             int coordinate)
{
    for (int a = 0; a < DIM; ++a) x[a].grad = 0.0;
    x[coordinate].grad = 1.0;
    dual y = fn(x);
    dual res = y.grad;
    return res;
}

dual partialFD(const func& fn, std::array<dual,DIM> x, int coordinate)
{
    for (int a = 0; a < DIM; ++a) x[a].grad = 0.0;
    double x0 = x[coordinate].val;
    double h  = 1e-5 * (x0 != 0.0 ? std::abs(x0) : 1.0);

    x[coordinate].val = x0 + h;  double fp = fn(x).val;
    x[coordinate].val = x0 - h;  double fm = fn(x).val;

    dual res = (fp - fm) / (2.0 * h);
    return res;
}

func christoffel(int i, int k, int l,
                 const met& g, const met& invg)
{
    return [i, k, l, g, invg](const std::array<dual,DIM>& x)
    {
        dual sum = 0.0;
        for (int m = 0; m < DIM; ++m) {
            const dual dg_mk_dl = partial(g[m][k], x, l);
            const dual dg_ml_dk = partial(g[m][l], x, k);
            const dual dg_kl_dm = partial(g[k][l], x, m);

            sum += invg[i][m](x)
                 * (dg_mk_dl + dg_ml_dk - dg_kl_dm);
        }
        dual res = 0.5 * sum;
        return res;
    };
}

func christoffel4D(int i, int k, int l,
                   const met& g, const met& invg)
{
    return [i, k, l, g, invg](const std::array<dual,DIM>& x)
    {
        dual sum = 0.0;
        for (int m = 0; m < D; ++m) {
            const dual dg_mk_dl = partial(g[m][k], x, l);
            const dual dg_ml_dk = partial(g[m][l], x, k);
            const dual dg_kl_dm = partial(g[k][l], x, m);

            sum += invg[i][m](x)
                 * (dg_mk_dl + dg_ml_dk - dg_kl_dm);
        }
        dual res = 0.5 * sum;
        return res;
    };
}


func Ricci(int k, int m, const christ& gam)
{
    return [k, m, gam](const std::array<dual,DIM>& x)
    {
        dual sum = 0.0;
        for (int i = 0; i < DIM; ++i) {
            sum += partialFD(gam[i][k][m], x, i) - partialFD(gam[i][i][k], x, m);
            for (int n = 0; n < DIM; ++n)
                sum += gam[i][i][n](x) * gam[n][k][m](x) - gam[i][m][n](x) * gam[n][i][k](x);
        }
        dual res = sum;
        return res;
    };
}

func RicciScalar(const christ& gam, const met& invg)
{
    std::vector<std::vector<func>> Rc(DIM, std::vector<func>(DIM));
    for (int k = 0; k < DIM; ++k)
        for (int m = 0; m < DIM; ++m)
            Rc[k][m] = Ricci(k, m, gam);

    return [Rc, invg](const std::array<dual,DIM>& x)
    {
        dual sum = 0.0;
        for (int k = 0; k < DIM; ++k)
            for (int m = 0; m < DIM; ++m)
                sum += invg[k][m](x) * Rc[k][m](x);
        dual res = sum;
        return res;
    };
}

}
