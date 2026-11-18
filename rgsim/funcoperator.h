#pragma once

#include <array>
#include "types.h"

namespace rgsim {

func constant(double v);
func operator+(func a, func b);
func operator-(func a, func b);
func operator*(func a, func b);
func operator/(func a, func b);

}
