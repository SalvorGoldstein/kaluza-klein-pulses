#include "field.h"
#include "metricoperator.h"
#include <cmath>
#include <iostream>

namespace rgsim {

void Point::setcoord(std::array<double,3> x1) { x = x1; }

std::array<double,3> Point::getcoord() const { return x; }

void Point::aff() const {
    std::cout << x[0] << " " << x[1] << " " << x[2] << std::endl;
}

double Point::dist() const {
    return std::sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
}


void VectorPotential::aff() const {
    std::cout << A[0] << " " << A[1] << " " << A[2] << " " << A[3] << std::endl;
}

double VectorPotential::norm2(const met& g, const Point& p, double t) const {
    const met ginv = inverseMet(g);

    std::array<dual,DIM> x;
    for (int a = 0; a < DIM; ++a) x[a] = 0.0;      // extra dimension = 0
    const std::array<double,3> c = p.getcoord();
    x[0] = t;
    x[1] = c[0];
    x[2] = c[1];
    x[3] = c[2];

    double s = 0.0;
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b)
            s += ginv[a][b](x).val * A[a] * A[b];
    return s;
}

double VectorPotential::norm(const met& g, const Point& p, double t) const {
    return std::sqrt(std::fabs(norm2(g, p, t)));
}

PointGrid makegridspherical(std::array<int,3> n, double Re, double Rb) {
    const double pi = std::acos(-1.0);
    const double eps    = 0.01 * pi;                  // distance from the poles
    double dtheta = (pi - 2*eps) / (n[1] - 1);
    double dR     = (Re - Rb) / (n[0] - 1);   // n[0] points, both Rb and Re included
    double dphi   = 2 * pi / n[2];            // phi wraps around: last point != first

    PointGrid Grid(n[0], std::vector<std::vector<Point>>(n[1], std::vector<Point>(n[2])));
    for (int i0 = 0; i0 < n[0]; i0++)
        for (int i1 = 0; i1 < n[1]; i1++)
            for (int i2 = 0; i2 < n[2]; i2++)
                Grid[i0][i1][i2].setcoord({Rb + i0*dR, eps + i1*dtheta, i2*dphi});
    return Grid;
}

PointGrid makegridcarte(std::array<int,3> n, std::array<double,3> L) {
    double dx = L[0] / n[0];
    double dy = L[1] / n[1];
    double dz = L[2] / n[2];

    PointGrid Grid(n[0], std::vector<std::vector<Point>>(n[1], std::vector<Point>(n[2])));
    for (int i0 = 0; i0 < n[0]; i0++)
        for (int i1 = 0; i1 < n[1]; i1++)
            for (int i2 = 0; i2 < n[2]; i2++)
                Grid[i0][i1][i2].setcoord({i0*dx, i1*dy, i2*dz});
    return Grid;
}

}  // namespace rgsim