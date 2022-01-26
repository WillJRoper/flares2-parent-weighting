import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import sys


sns.set_context("paper")
sns.set_style('whitegrid')

# Get the commandline argument for which snapshot
num = int(sys.argv[1])

# Extract the snapshot string
snaps = [str(i).zfill(4) for i in range(0, 19)]
snap = snaps[num]

# Set up bins
step = 0.1
bin_edges = np.arange(0.00001, 15 + step, step)
bin_cents = (bin_edges[:-1] + bin_edges[1:]) / 2
step = 0.05
log_bin_edges = np.arange(-1.0, 1.0 + step, step)
log_bin_cents = (log_bin_edges[:-1] + log_bin_edges[1:]) / 2

# Define path to file
metafile = "overdensity_L2800N5040_HYDRO_snap%s.hdf5" % snap
path = "/cosma7/data/dp004/FLARES/FLARES-2/Parent/" \
       "overdensity_gridding/L2800N5040/HYDRO/snap_" + snap + "/" + metafile

# Open file
hdf = h5py.File(path, "r")

mean_density = hdf["Parent"].attrs["Mean_Density"]

# Set up array to store counts
H_tot = np.zeros_like(bin_cents)
log_H_tot = np.zeros_like(log_bin_cents)

# Loop over cells
for key in hdf.keys():

    # Skip meta data
    if key in ["Parent", "Delta_grid"]:
        continue

    print(key, np.min(hdf[key]["grid"][...]) * mean_density + mean_density,
          np.max(hdf[key]["grid"][...]) * mean_density + mean_density)

    # Get counts for this cell
    H, _ = np.histogram(hdf[key]["grid"][...], bins=bin_edges)

    # Add counts to main array
    H_tot += H

    # Get counts for this cell
    H, _ = np.histogram(np.log10(hdf[key]["grid"][...]), bins=log_bin_edges)

    # Add counts to main array
    log_H_tot += H

fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy()

ax.plot(bin_cents, H_tot, label="L2800N5040_HYDRO_2Mpc")

ax.set_xlabel("$1 + \delta$")
ax.set_ylabel("$N$")

fig.savefig("plots/overdensity_hist_" + snap + ".png", bbox_inches="tight")

plt.close()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy()

ax.plot(log_bin_cents, log_H_tot, label="L2800N5040_HYDRO_2Mpc")

ax.set_xlabel("$\log_{10}(1 + \delta)$")
ax.set_ylabel("$N$")

fig.savefig("plots/log_overdensity_hist_" + snap + ".png", bbox_inches="tight")

plt.close()
