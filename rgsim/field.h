#pragma once
#include <array>
#include <vector>
#include "types.h"

namespace rgsim {

class Point {
public:
    void setcoord(std::array<double,3> x1);
    std::array<double,3> getcoord() const;
    void aff() const;
    double dist() const;
private:
    std::array<double,3> x{};
};


class VectorPotential {
public:
    void setvalue(std::array<double,4> A1) { A = A1; }
    std::array<double,4> getvalue() const { return A; }
    void aff() const;
    double norm2(const met& g, const Point& p, double t = 0.0) const;   // g^{mu nu} A_mu A_nu (signed)
    double norm(const met& g, const Point& p, double t = 0.0) const;    // sqrt(|norm2|)
private:
    std::array<double,4> A{};
};

using PointGrid = std::vector<std::vector<std::vector<Point>>>;
using AGrid     = std::vector<std::vector<std::vector<VectorPotential>>>;

PointGrid makegridspherical(std::array<int,3> n, double Re, double Rb);
PointGrid makegridcarte(std::array<int,3> n, std::array<double,3> L);

}