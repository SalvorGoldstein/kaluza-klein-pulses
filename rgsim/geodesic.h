#pragma once

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include "types.h"
#include "systemsolver.h"

namespace rgsim {

double solveU0(const std::array<double,DIM>& pos, const spatial& us,
               const met& g, double kappa);

double timelikeU(const std::array<double,DIM>& pos, const spatial& us, const met& g);

double lightlikeU(const std::array<double,DIM>& pos, const spatial& us, const met& g);

void geo(double (&q)[2*DIM], double tau, double (&qp)[2*DIM], int n, const christ& Gam);

spatial filled(double v);

void geodesics(const spatial& xs, const spatial& us,
               int steps, double dtau,
               const std::string& filename,
               const met& g, const christ& Gam,
               const rhs_fn& sd = geo,
               bool type = true,
               const spatial& xmax = filled(constants::infinity),
               const spatial& xmin = filled(constants::minusinfinity));

}
