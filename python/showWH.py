# Animation for main_wormhole: |A| in the equatorial plane, drawn on the two sides of an Ellis wormhole.
# Left: the r > 0 side, right: the r < 0 side.  Each side is drawn in areal radius rho = sqrt(r^2 + a^2),
# so the throat (r = 0) is the circle rho = a (grey disc), and the field crosses it from one panel to the other.
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
a, r0, R = d[0], d[1], d[2]
nr, n = int(d[3]), int(d[4])
dtf = d[5]                                       # time between frames
nph = int(d[7])                                  # points on the full circle
k = 1e3                                          # metres -> km

# each Plane file: one line per (r, phi) on the equator, columns r phi |A|
def load(i):
    P = np.loadtxt(base + "data/Plane" + str(i) + ".dat")
    P = P.reshape(nr, nph, P.shape[1])           # columns r phi |A| [dilaton]
    return np.concatenate([P, P[:, :1]], axis=1)  # repeat the first angle to close the circle

planes = [load(i) for i in range(n)]
r   = planes[0][:, :, 0]
ph  = planes[0][:, :, 1]
rho = np.sqrt(r ** 2 + a ** 2)                   # areal radius
X = rho * np.cos(ph) / k
Y = rho * np.sin(ph) / k
Z = [P[:, :, 2] for P in planes]
has_dil = planes[0].shape[2] > 3
Zd = [P[:, :, 3] for P in planes] if has_dil else None

# nan/inf or runaway values = the simulation blew up: keep only the frames before it
ref = max(z.max() for z in Z[:max(5, len(Z) // 10)] if np.isfinite(z).all())
good = [i for i, z in enumerate(Z) if np.isfinite(z).all() and z.max() < 4.0 * ref]
if len(good) < len(Z):
    last = good[-1] if good else 0
    print("WARNING: the simulation became unstable after frame %d of %d: showing the first %d frames only."
          % (last, n, last + 1))
    n = last + 1
    Z = Z[:n]
    Zd = Zd[:n] if has_dil else None
vmax = max(z.max() for z in Z)

ipos = np.where(r[:, 0] >= 0)[0]                 # rows with r >= 0
ineg = np.where(r[:, 0] <= 0)[0]                 # rows with r <= 0
side = ((ipos, "r > 0"), (ineg, "r < 0"))
lim = np.sqrt(R ** 2 + a ** 2) / k

nrow = 2 if has_dil else 1
fig, axg = plt.subplots(nrow, 2, figsize=(11, 5.2 * nrow), constrained_layout=True, squeeze=False)
axs = axg[0]
meshes = []
for ax, (rows, title) in zip(axs, side):
    ax.set_facecolor("lightgrey")                # the throat disc shows through where nothing is plotted
    m = ax.pcolormesh(X[rows], Y[rows], Z[0][rows], shading="gouraud", cmap="plasma",
                      norm=PowerNorm(0.5, vmin=0, vmax=vmax))
    ax.add_patch(Circle((0, 0), a / k, fc="lightgrey", ec="white", lw=1, zorder=3))      # throat r = 0
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title(title)
    meshes.append(m)
ax.figure.colorbar(meshes[1], ax=axs, label="|A|")
# outside the simulated disc the background is black
for ax in axs:
    ax.set_facecolor("black")
meshesd = []
if has_dil:                                      # second row: the dilaton, same geometry
    md_ = max(max(np.abs(z).max() for z in Zd), 1e-300)
    for ax, (rows, title) in zip(axg[1], side):
        ax.set_facecolor("black")
        m = ax.pcolormesh(X[rows], Y[rows], Zd[0][rows], shading="gouraud", cmap="coolwarm", vmin=-md_, vmax=md_)
        ax.add_patch(Circle((0, 0), a / k, fc="lightgrey", ec="white", lw=1, zorder=3))
        ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)"); ax.set_title("dilaton, " + title)
        meshesd.append(m)
    ax.figure.colorbar(meshesd[1], ax=list(axg[1]), label="dilaton")
sup = fig.suptitle("")

def update(i):
    for m, (rows, _) in zip(meshes, side):
        m.set_array(Z[i][rows])                  # keep the 2D shape (a flattened array breaks gouraud shading)
    for m, (rows, _) in zip(meshesd, side):
        m.set_array(Zd[i][rows])
    sup.set_text("|A|   t = %.3e s   max = %.2e   (%d/%d)" % (i * dtf, Z[i].max(), i + 1, n))
    return meshes

anim = FuncAnimation(fig, update, frames=n, interval=50)

try:                                             # mp4 if ffmpeg is installed
    anim.save(base + "wormhole.mp4", writer=FFMpegWriter(fps=20))
    print("saved " + base + "wormhole.mp4")
except Exception:
    anim.save(base + "wormhole.gif", writer=PillowWriter(fps=20))
    print("saved " + base + "wormhole.gif")
plt.close()
