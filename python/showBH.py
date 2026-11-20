import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                            # no window: the video is only saved
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

base = os.path.dirname(os.path.abspath(__file__)) + "/"
d = np.loadtxt(base + "dataforpy.dat")
r_s, r0, r1 = d[0], d[1], d[2]
nr, n = int(d[3]), int(d[4])
dtf = d[5]                                       # time between frames
nph = int(d[7])                                  # points on the full circle
k = 1e3                                          # metres -> km

# each Plane file (written by main_blackhole5d): one line per (r, phi) on the equator, columns x y |A| A_0
#   |A| = norm of the spatial part of the pulse, A_0 = change of the scalar potential A_0 (see below)
def load(i):
    P = np.loadtxt(base + "data/Plane" + str(i) + ".dat")
    P = P.reshape(nr, nph, P.shape[1])
    return np.concatenate([P, P[:, :1]], axis=1)  # repeat the first angle to close the circle

planes = [load(i) for i in range(n)]
X = planes[0][:, :, 0] / k
Y = planes[0][:, :, 1] / k
Z = [P[:, :, 2] for P in planes]
has_A0 = planes[0].shape[2] > 3
Z0 = [P[:, :, 3] for P in planes] if has_A0 else None          # scalar potential A_0

# A frame with nan/inf means the simulation blew up (numerical instability): stop the animation just before it,
# otherwise one huge value ruins the colour scale (and, with new matplotlib versions, crashes the drawing).
ref = max(z.max() for z in Z[:max(5, len(Z) // 10)] if np.isfinite(z).all())   # size of the field at the start
good = [i for i, z in enumerate(Z) if np.isfinite(z).all() and z.max() < 4.0 * ref]   # runaway = 4x the early maximum
nbad = len(Z) - len(good)
if nbad:
    last = good[-1] if good else 0
    print("WARNING: the simulation became unstable after frame %d of %d (nan/inf or runaway values): "
          "showing the first %d frames only." % (last, n, last + 1))
    Z = Z[:last + 1]
    n = len(Z)
# colour scale: the largest value of the frames that are kept
vmax = max(z.max() for z in Z)

# extra panels: columns 3.. of the Plane files.  With the new library there is only one: A_0.
# systemA does not evolve A_0 (its rate is 0), so this column is 0 everywhere and the panel is skipped below;
# it appears automatically if A_0 is ever evolved.  (The old psi = ln Phi and dilaton columns no longer exist.)
extra = []
names = ["A_0 (scalar potential)", "dilaton"]
ncols_file = planes[0].shape[2]
for c in range(3, ncols_file):
    Zc = [P[:, :, c] for P in planes][:n]
    m = max(max(np.abs(z).max() for z in Zc), 0.0)
    if m > 0.0:                                   # skip a panel that is identically zero (e.g. A_0 = 0)
        extra.append((names[c - 3] if c - 3 < len(names) else "column %d" % c, Zc, m))

ncol = 1 + len(extra)
fig, axs = plt.subplots(1, ncol, figsize=(7 * ncol, 6), constrained_layout=True, squeeze=False)
axs = axs[0]
ax = axs[0]
ax.set_facecolor("black")                        # outside the simulated ring: nothing is shown
mesh = ax.pcolormesh(X, Y, Z[0], shading="gouraud", cmap="plasma",
                     norm=PowerNorm(0.5, vmin=0, vmax=vmax, clip=True))
fig.colorbar(mesh, ax=ax, label="|A|  (the pulse)")
ax.set_title("norm of A")
meshes = [mesh]
extraZ = []
for j, (name, Zc, m) in enumerate(extra):      # (j, not k: k is the metres -> km factor used below)
    axk = axs[j + 1]
    axk.set_facecolor("black")
    mk = axk.pcolormesh(X, Y, Zc[0], shading="gouraud", cmap="coolwarm", vmin=-m, vmax=m)   # same symmetric range for all frames
    fig.colorbar(mk, ax=axk, label=name)
    axk.set_title(name)
    meshes.append(mk)
    extraZ.append(Zc)

for a_ in axs:
    a_.add_patch(Circle((0, 0), r_s / k, fill=False, ec="white", ls="--", lw=1, zorder=3))   # horizon
    a_.set_aspect("equal")
    a_.set_xlim(-r1 / k, r1 / k)
    a_.set_ylim(-r1 / k, r1 / k)
    a_.set_xlabel("x (km)")
    a_.set_ylabel("y (km)")
sup = fig.suptitle("")

def update(i):
    meshes[0].set_array(Z[i])                    # keep the 2D shape (a flattened array breaks gouraud shading in recent matplotlib)
    for j, Zc in enumerate(extraZ):
        meshes[j + 1].set_array(Zc[i])
    sup.set_text("t = %.3e s   max|A| = %.2e   (%d/%d)" % (i * dtf, Z[i].max(), i + 1, n))
    return meshes

anim = FuncAnimation(fig, update, frames=n, interval=50)

try:                                             # mp4 if ffmpeg is installed
    anim.save(base + "blackhole.mp4", writer=FFMpegWriter(fps=20))
    print("saved " + base + "blackhole.mp4")
except Exception:
    anim.save(base + "blackhole.gif", writer=PillowWriter(fps=20))
    print("saved " + base + "blackhole.gif")
plt.close()