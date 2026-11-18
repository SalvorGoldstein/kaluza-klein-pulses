#include "funcoperator.h"

namespace rgsim {

func constant(double v)
{
    return [v](const std::array<dual,DIM>&) { dual res = v; return res; };
}

func operator+(func a, func b)
{
    return [a, b](const std::array<dual,DIM>& x)
    { dual p = a(x); dual q = b(x); dual res = p + q; return res; };
}
func operator-(func a, func b)
{
    return [a, b](const std::array<dual,DIM>& x)
    { dual p = a(x); dual q = b(x); dual res = p - q; return res; };
}
func operator*(func a, func b)
{
    return [a, b](const std::array<dual,DIM>& x)
    { dual p = a(x); dual q = b(x); dual res = p * q; return res; };
}
func operator/(func a, func b)
{
    return [a, b](const std::array<dual,DIM>& x)
    { dual p = a(x); dual q = b(x); dual res = p / q; return res; };
}

}
