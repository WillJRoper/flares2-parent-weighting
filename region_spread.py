import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cm
from flare import plt as flareplt
import h5py
import sys
plt.rcParams['axes.grid'] = True


# Get the commandline argument for which snapshot
num = int(sys.argv[1])

# Get the simulation "tag"
sim_tag = sys.argv[2]

# Get the simulation "type"
sim_type = sys.argv[3]

# Define kernel width
ini_kernel_width = int(sys.argv[4])

# Extract the snapshot string
snaps = [str(i).zfill(4) for i in range(0, 21)]
snap = snaps[num]

# Define output paths
metafile = "overdensity_" + sim_tag + "_" + sim_type + "_snap%s.hdf5" % snap
outdir = "/cosma7/data/dp004/FLARES/FLARES-2/Parent/" \
        "overdensity_gridding/" + sim_tag + "/" + sim_type + "/snap_" + snap + "/"

# Define path to file
file = "smoothed_" + metafile.split(".")[
   0] + "_kernel%d.hdf5" % k
path = outdir + file

# Open file
hdf = h5py.File(path, "r")

grid = hdf["Region_Overdensity"][...]
grid_std = hdf["Region_Overdensity_Stdev"][...]

hdf.close()

fig = plt.figure()
ax = fig.add_subplot(111)

im = ax.hexbin(np.log10(grid), grid_std,
               norm=cm.LogNorm,
               gridsize=100, mincnt=1,
               linewidths=0.2, cmap='viridis')

ax.set_xlabel("$\log_{10}(1 + \delta)$")
ax.set_ylabel("$\sigma(1 + \delta)$")

cbar = fig.colorbar(im)
cbar.set_label("$N$")

fig.savefig("plots/log_region_hist_" + snap + ".png",
            bbox_inches="tight")

plt.close()
