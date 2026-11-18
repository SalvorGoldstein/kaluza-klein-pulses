#pragma once

#include <array>
#include <cmath>
#include <vector>
#include "types.h"

namespace rgsim {

dual partial(const func& fn,
             std::array<dual,DIM> x,
             int coordinate);

dual partialFD(const func& fn, std::array<dual,DIM> x, int coordinate);

func christoffel(int i, int k, int l,
                 const met& g, const met& invg);

func christoffel4D(int i, int k, int l,
                   const met& g, const met& invg);

func Ricci(int k, int m, const christ& gam);

func RicciScalar(const christ& gam, const met& invg);

}
