# Animation for main_cartesian: |A| on the plane z = pulse centre (and, if the dilaton was on, the dilaton phi).
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                            # no window: the video is only saved
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

base = os.path.dirname(os.path.abspath(__file__)) + "/"
d = np.loadtxt(base + "dataforpy.dat")
nr, n = int(d[3]), int(d[4])
dtf = d[5] if len(d) > 5 else None               # time between frames
k, u = (1e3, "km") if d[2] > 1e3 else (1.0, "m")

# Plane files: columns x, y, |A|  [, dilaton]
raw = [np.loadtxt(base + "data/Plane" + str(i) + ".dat") for i in range(n)]
X = raw[0][:, 0].reshape(nr, nr) / k
Y = raw[0][:, 1].reshape(nr, nr) / k
Z = [P[:, 2].reshape(nr, nr) for P in raw]
has_dil = raw[0].shape[1] > 3
Zd = [P[:, 3].reshape(nr, nr) for P in raw] if has_dil else None

# A frame with nan/inf or a runaway value means the simulation blew up: keep only the frames before it,
# otherwise one huge value ruins the colour scale (and crashes the drawing in recent matplotlib).
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

ncol = 2 if has_dil else 1
fig, axs = plt.subplots(1, ncol, figsize=(7 * ncol, 6), constrained_layout=True, squeeze=False)
axs = axs[0]
mesh = axs[0].pcolormesh(X, Y, Z[0], shading='gouraud', cmap='plasma', norm=PowerNorm(0.5, vmin=0, vmax=vmax))
fig.colorbar(mesh, ax=axs[0], label="|A|")
meshes = [mesh]
if has_dil:
    m = max(np.abs(z).max() for z in Zd)
    m = m if m > 0 else 1.0
    md = axs[1].pcolormesh(X, Y, Zd[0], shading='gouraud', cmap='coolwarm', vmin=-m, vmax=m)
    fig.colorbar(md, ax=axs[1], label="dilaton")
    axs[1].set_title("dilaton")
    meshes.append(md)
for ax in axs:
    ax.set_aspect('equal')
    ax.set_xlabel("x (" + u + ")")
    ax.set_ylabel("y (" + u + ")")
sup = fig.suptitle("")

def update(i):
    meshes[0].set_array(Z[i])                    # keep the 2D shape (a flattened array breaks gouraud shading)
    if has_dil:
        meshes[1].set_array(Zd[i])
    t = "   t = %.3e s" % (i * dtf) if dtf else ""
    sup.set_text("|A|" + t + "   max = %.2e   (%d/%d)" % (Z[i].max(), i + 1, n))
    return meshes

anim = FuncAnimation(fig, update, frames=n, interval=50)

try:                                             # mp4 if ffmpeg is installed
    anim.save(base + "field.mp4", writer=FFMpegWriter(fps=20))
    print("saved " + base + "field.mp4")
except Exception:
    anim.save(base + "field.gif", writer=PillowWriter(fps=20))
    print("saved " + base + "field.gif")
plt.close()
