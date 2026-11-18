#pragma once

#include <array>
#include <cmath>
#include <vector>
#include "types.h"
#include "funcoperator.h"

namespace rgsim {

met minorOf(const met& m, int row, int col);
func determinant(const met& m);
met inverseMet(const met& g);
func dist(const met& g, const met& ref);

}
