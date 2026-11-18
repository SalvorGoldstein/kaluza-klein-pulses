# rgsim: electromagnetic pulses in curved space

A small C++ library (`rgsim/`) and three example programs that evolve the vector potential `A_mu`
on a fixed curved 4D metric, using a reduced 5D (Kaluza-Klein-like) field equation. A pulse is sent through
the space and a Python script turns the result into an animation.

## The three programs

| Program | Space | Coordinates | Animation |
|---------|-------|-------------|-----------|
| `main_blackhole.cc` | Kerr-Newman black hole (optionally with its own electromagnetic field) | spherical (r, theta, phi) | `python/showBH.py` |
| `main_cartesian.cc` | flat space (Minkowski) in a box | Cartesian (x, y, z) | `python/showA.py` |
| `main_wormhole.cc` | Ellis wormhole, the pulse crosses the throat | spherical (r, theta, phi), r from -R to +R | `python/showWH.py` |

## Layout

| Path | Content |
|------|---------|
| `rgsim/` | the library (see below) |
| `example_metric.*` | metrics: Minkowski, Schwarzschild, Reissner-Nordstrom, Kerr, Kerr-Newman, Ellis wormhole (spherical and Cartesian versions) |
| `example_potential.*` | initial pulses (gaussian, wavepacket, soliton, chirp, ...) and the black hole's own field |
| `helpers.h` | small functions shared by the three programs (grid, initial state, time step, ...) |
| `main_*.cc` | the three programs |
| `python/` | the animation scripts |
| `rgsim/third_party/` | autodiff and Eigen (headers only) |

## The library

- **Functions of the coordinates** (`types.h`, `funcoperator.*`, `metricoperator.*`): a metric is a 4x4 table of
  `func`, i.e. functions of `(t, x1, x2, x3, x4)` written with `autodiff` dual numbers. `inverseMet` gives the
  inverse metric.
- **Differential geometry** (`Geodiff.*`): partial derivatives by automatic differentiation, Christoffel symbols,
  Ricci tensor and scalar.
- **Grid** (`field.*`): `Point`, `VectorPotential` (4 numbers `A_0..A_3`), `PointGrid`, `AGrid`.
- **Equation of motion** (`system.*`): `systemA` gives `d_t A` and `d_t V` (with `V = d_t A`) on the whole grid,
  with finite differences in space and boundary conditions per face (Dirichlet, Neumann, periodic, outgoing wave).
  `A_0` is not evolved.
- **Time stepping** (`systemsolver.*`): `rk4A` makes one Runge-Kutta 4 step on `(A, V)`.
- `einstein.*` (the 5D metric `g_hat`), `geodesic.*` (geodesics).

`field_solver.*` is the previous solver (state vector instead of grids). It is kept in the folder but is not part
of the build any more. `geodesic.cc` is not built either: it still calls an `rk4` for plain arrays that no
longer exists.

## Build and run

You need `g++` (C++17, OpenMP) and, for the animations, Python 3 with `numpy` and `matplotlib` (`ffmpeg` is
optional: without it a `.gif` is written instead of an `.mp4`).

```
make               # builds the three programs
make blackhole     # runs main_blackhole, then makes python/blackhole.mp4
make cartesian     # runs main_cartesian, then makes python/field.mp4
make wormhole      # runs main_wormhole,  then makes python/wormhole.mp4
make clean
```

Each program asks for its parameters (initial pulse, size of the grid, number of points, duration, ...).
It writes `python/dataforpy.dat`, `python/data/Field*.dat` and `python/data/Plane*.dat`, which the Python script reads.

Speed: `systemA` computes the metric data (inverse metric, Christoffel symbols, `h`, `box`) once per grid point
and keeps it, so the autodiff functions are not called again at every Runge-Kutta stage, and the loop over the
grid points runs in parallel (OpenMP). This assumes the metric does not depend on `t` (true for all the
examples). If yours does, set `metricIsStatic = false` (`rgsim/system.h`); call `resetMetricCache()` if you change the metric
or the grid. On one core, a grid of 150 000 points needs about 45 s for the one-time setup and then about 0.3 s per time step.

## Credits

Claude (Anthropic) wrote the RK4 part (`rk4A`, `systemsolver.*`) and the system part (`systemA`, `system.*`) of the
program, and optimised both of them.
