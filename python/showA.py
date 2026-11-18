import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

base = os.path.dirname(os.path.abspath(__file__)) + "/"
d = np.loadtxt(base + "dataforpy.dat")
nr, n = int(d[3]), int(d[4])
dtf = d[5] if len(d) > 5 else None               # time between frames
k, u = (1e3, "km") if d[2] > 1e3 else (1.0, "m")

planes = [np.loadtxt(base + "data/Plane" + str(i) + ".dat") for i in range(n)]
X = planes[0][:, 0].reshape(nr, nr) / k
Y = planes[0][:, 1].reshape(nr, nr) / k
Z = [P[:, 2].reshape(nr, nr) for P in planes]
vmax = max(z.max() for z in Z)

fig, ax = plt.subplots(figsize=(7, 6))
mesh = ax.pcolormesh(X, Y, Z[0], shading='gouraud', cmap='plasma',
                     norm=PowerNorm(0.5, vmin=0, vmax=vmax))
fig.colorbar(mesh, label="|A|")
ax.set_aspect('equal')
ax.set_xlabel("x (" + u + ")")
ax.set_ylabel("y (" + u + ")")

def update(i):
    mesh.set_array(Z[i].ravel())
    t = "   t = %.3e s" % (i * dtf) if dtf else ""
    ax.set_title("|A|" + t + "   max = %.2e   (%d/%d)" % (Z[i].max(), i + 1, n))
    return mesh,

anim = FuncAnimation(fig, update, frames=n, interval=50)

try:                                             # mp4 if ffmpeg is installed
    anim.save(base + "field.mp4", writer=FFMpegWriter(fps=20))
    print("saved " + base + "field.mp4")
except Exception:
    anim.save(base + "field.gif", writer=PillowWriter(fps=20))
    print("saved " + base + "field.gif")
plt.close()