#include "metricoperator.h"

namespace rgsim {

met minorOf(const met& m, int row, int col)
{
    int n = m.size();
    met s(n - 1, std::vector<func>(n - 1));
    for (int i = 0, a = 0; i < n; ++i) {
        if (i == row) continue;
        for (int j = 0, b = 0; j < n; ++j) {
            if (j == col) continue;
            s[a][b++] = m[i][j];
        }
        ++a;
    }
    return s;
}

func determinant(const met& m)
{
    int n = m.size();
    if (n == 1) return m[0][0];
    func sum = constant(0.0);
    for (int j = 0; j < n; ++j) {
        func term = m[0][j] * determinant(minorOf(m, 0, j));
        if (j % 2 == 0) sum = sum + term;
        else            sum = sum - term;
    }
    return sum;
}

met inverseMet(const met& g)
{
    int n = g.size();
    func det = determinant(g);
    met inv(n, std::vector<func>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            func cof = determinant(minorOf(g, j, i));
            if ((i + j) % 2 == 1) cof = constant(0.0) - cof;
            inv[i][j] = cof / det;
        }
    return inv;
}

func dist(const met& g, const met& ref)
{
    return [g, ref](const std::array<dual,DIM>& x)
    {
        dual sum = 0.0;
        for (int a = 0; a < DIM; ++a)
            for (int b = 0; b < DIM; ++b) {
                double w = (a == 0 ? 1.0 / constants::c : 1.0)
                         * (b == 0 ? 1.0 / constants::c : 1.0);
                dual h = g[a][b](x) - ref[a][b](x);
                sum += (w * h) * (w * h);
            }
        dual res = sqrt(sum);
        return res;
    };
}

}
