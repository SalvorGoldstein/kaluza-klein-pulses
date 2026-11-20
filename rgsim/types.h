#pragma once

#include <vector>
#include <array>
#include <functional>
#include <limits>
#include <cmath>
#include<array>
#include <autodiff/forward/dual.hpp>

constexpr int DIM = 5;
constexpr int D   = 4;            
constexpr int E   = DIM - D;      
constexpr int NS  = D - 1; 

namespace rgsim {

using autodiff::dual;

using func    = std::function<dual(const std::array<dual,DIM>&)>;
using met     = std::vector<std::vector<func>>;
using christ  = std::vector<std::vector<std::vector<func>>>;
using fvec  = std::vector<func>;



using rhs_fn = std::function<void(double (&)[2*DIM], double, double (&)[2*DIM], int,
                                  const christ&)>;

namespace constants {
inline constexpr double G    = 6.67430e-11;
inline constexpr double c    = 299792458.0;
inline constexpr double eps0 = 8.8541878128e-12;
inline constexpr double pi   = 3.14159265358979323846;
inline const double infinity = std::numeric_limits<double>::infinity();
inline const double minusinfinity = std::numeric_limits<double>::lowest();

class Point{
  public : 
  std::array<double,3> coord;
  void setcoord(std::array<double,3> x);
  double getcoord();
  double dist();
};
}

}