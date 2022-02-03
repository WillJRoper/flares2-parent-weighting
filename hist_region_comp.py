import numpy as np
import matplotlib.pyplot as plt
from flare import plt as flareplt
import h5py
import sys
plt.rcParams['axes.grid'] = True


# Get the commandline argument for which snapshot
num = int(sys.argv[1])

# Extract the snapshot string
snaps = [str(i).zfill(4) for i in range(0, 21)]
zs = [15, 12.26, 10.38, 9.51, 8.7, 7.95, 7.26, 6.63, 6.04, 5.50, 5.0, 4.75,
      4.5, 4.25, 2.0, 3.75, 3.5, 3.25, 3.0, 2.95, 2.90, 2.85, 2.8, 2.75,
      2.7, 2.65, 2.6, 2.55, 2.5]
snap = snaps[num]
z = zs[num]

# Set up bins
step = 0.1
bin_edges = np.arange(0.00001, 15 + step, step)
bin_cents = (bin_edges[:-1] + bin_edges[1:]) / 2
step = 0.05
log_bin_edges = np.arange(-1.0, 2.0 + step, step)
log_bin_cents = (log_bin_edges[:-1] + log_bin_edges[1:]) / 2

hists = {}

for sim_tag in ["L2800N5040", ]:
       for sim_type in ["HYDRO", "DMO"]:
           for k in [2, 25]:
               hists[sim_tag + "_" + sim_type + "_" + str(k) + "Mpc"] = {}

               print(sim_tag, sim_type, k)

               if k == 2:

                   # Define path to file
                   metafile = "overdensity_" + sim_tag + "_" + sim_type + "_snap%s.hdf5" % snap
                   path = "/cosma7/data/dp004/FLARES/FLARES-2/Parent/" \
                         "overdensity_gridding/" + sim_tag + "/" + sim_type + "/snap_" + snap + "/" + metafile

                   # Open file
                   hdf = h5py.File(path, "r")

                   grid = hdf["Parent_Grid"][...]

                   # Get counts for this cell
                   hists[sim_tag + "_" + sim_type + "_" + str(k) + "Mpc"]["log_H"], _ = np.histogram(np.log10(grid), bins=log_bin_edges)

                   hdf.close()

               else:

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

                   hdf.close()

                   # Get counts for this cell
                   hists[sim_tag + "_" + sim_type + "_" + str(k) + "Mpc"]["log_H"], _ = np.histogram(np.log10(grid), bins=log_bin_edges)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy()

for sim_tag in ["L2800N5040", ]:
       for sim_type in ["HYDRO", "DMO"]:
           for k in [2, 25]:
               if k == 2:
                  ax.plot(log_bin_cents,
                          hists[sim_tag + "_" + sim_type + "_"
                                + str(k) + "Mpc"]["log_H"],
                          label=sim_tag + "_" + sim_type + "_%dMpc" % k)
               else:
                   ax.plot(log_bin_cents,
                           hists[sim_tag + "_" + sim_type + "_"
                                 + str(k) + "Mpc"]["log_H"], linestyle="--",
                           label=sim_tag + "_" + sim_type + "_%dMpc" % k)

ax.set_xlabel("$\log_{10}(1 + \delta)$")
ax.set_ylabel("$N$")

ax.text(0.95, 0.05, f'$z={z}$',
        bbox=dict(boxstyle="round,pad=0.3", fc='w',
                  ec="k", lw=1, alpha=0.8),
        transform=ax.transAxes,
        horizontalalignment='right')

ax.legend(loc="upper right")

fig.savefig("plots/log_region_hist_" + snap + ".png",
            bbox_inches="tight")

plt.close()
