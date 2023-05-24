import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cm
from flare import plt as flareplt
import h5py
import sys
plt.rcParams['axes.grid'] = True


# Define the snapshot strings
snaps = [str(i).zfill(4) for i in range(0, 20)]

slopes = {}
odens = {}
zs = {}

# Set up redshift norm
norm = cm.Normalize(vmin=2, vmax=15)
cmap = plt.get_cmap('plasma', len(snaps))

# Loop over snapshots
prev_grid = None
prev_time = None
for snap in snaps:

    print(snap)

    # Define path to file
    metafile = "overdensity_L2800N5040_DMO_snap%s.hdf5" % snap
    path = "/cosma7/data/dp004/FLARES/FLARES-2/Parent/" \
        "overdensity_gridding/L2800N5040/DMO/snap_" + snap + "/" + metafile

    # Open file
    hdf = h5py.File(path, "r")

    mean_density = hdf["Parent"].attrs["Mean_Density"]
    z = hdf["Parent"].attrs["Redshift"]
    grid = hdf["Parent_Grid"][...]

    hdf.close()

    # Compute the logarithmic slope of overdensity
    if prev_grid is not None:

        delta = grid - prev_grid

        slopes[snap] = delta.flatten()
        odens[snap] = grid.flatten()
        zs[snap] = z

    prev_grid = grid


fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy()

for snap in zs.keys():

    H, bins = np.histogram(slopes[snap], bins=50)
    bin_cents = (bins[1:] + bins[:-1]) / 2

    ax.plot(bin_cents, H, color=cmap(norm(z)))

ax2 = fig.add_axes([0.95, 0.1, 0.015, 0.8])
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm)
cb1.set_label("$z$")

ax.set_xlabel("$\Delta_B - \Delta_A$")
ax.set_ylabel("$N$")

fig.savefig("plots/delta_overdensity_z_L2800N5040_DMO.png",
            bbox_inches="tight")
plt.close(fig)
    
