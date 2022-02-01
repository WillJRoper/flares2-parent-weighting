import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cm
import matplotlib as mpl
import seaborn as sns
import h5py
import sys


sns.set_context("paper")
sns.set_style('whitegrid')


# Get the simulation "tag"
sim_tag = sys.argv[1]

# Get the simulation "type"
sim_type = sys.argv[2]

# Extract the snapshot string
snaps = [str(i).zfill(4) for i in range(0, 21)]

# Set up bins
step = 0.1
bin_edges = np.arange(0.00001, 15 + step, step)
bin_cents = (bin_edges[:-1] + bin_edges[1:]) / 2
step = 0.05
log_bin_edges = np.arange(-1.0, 1.0 + step, step)
log_bin_cents = (log_bin_edges[:-1] + log_bin_edges[1:]) / 2

# Set up redshift norm
norm = cm.Normalize(vmin=2, vmax=15)
cmap = cm.get_cmap('plasma', len(snaps))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy()

fig_log = plt.figure()
ax_log = fig_log.add_subplot(111)
ax_log.semilogy()

for snap in snaps:

       # Define path to file
       metafile = "overdensity_" + sim_tag + "_" + sim_type + "_snap%s.hdf5" % snap
       path = "/cosma7/data/dp004/FLARES/FLARES-2/Parent/" \
              "overdensity_gridding/" + sim_tag + "/" + sim_type + "/snap_" + snap + "/" + metafile

       # Open file
       hdf = h5py.File(path, "r")

       mean_density = hdf["Parent"].attrs["Mean_Density"]
       z = hdf["Parent"].attrs["Redshift"]
       grid = hdf["Parent_Grid"][...]

       # Get counts for this cell
       H, _ = np.histogram(grid, bins=bin_edges)

       # Get counts for this cell
       log_H, _ = np.histogram(np.log10(grid), bins=log_bin_edges)

       # Plot this snapshot
       ax.plot(bin_cents, H, color=cmap(norm(z)))
       ax_log.plot(log_bin_cents, log_H, color=cmap(norm(z)))

ax.set_xlabel("$1 + \delta$")
ax.set_ylabel("$N$")
ax_log.set_xlabel("$\log_{10}(1 + \delta)$")
ax_log.set_ylabel("$N$")

ax2 = fig.add_axes([0.95, 0.1, 0.015, 0.8])
cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                                norm=norm)

fig.savefig(
       "plots/overden_hist_" + sim_tag + "_" + sim_type + ".png",
       bbox_inches="tight")
plt.close(fig)

ax2_log = fig_log.add_axes([0.95, 0.1, 0.015, 0.8])
cb1 = mpl.colorbar.ColorbarBase(ax2_log, cmap=cmap,
                                norm=norm)

fig_log.savefig("plots/log_overden_hist_" + sim_tag + "_" + sim_type + ".png",
                bbox_inches="tight")
plt.close(fig_log)
