import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys


# Get the commandline argument for which snapshot
num = int(sys.argv[1])

# Extract the snapshot string
snaps = [str(i).zfill(4) for i in range(0, 19)]
snap = snaps[num]

# Set up bins
bin_edges = np.linspace(-1, 1, 100)
bin_cents = (bin_edges[:-1] + bin_edges[1:]) / 2

# Define path to file
metafile = "overdensity_L2800N5040_HYDRO_snap%s.hdf5" % snap
path = "/cosma7/data/dp004/FLARES/FLARES-2/Parent/" \
       "overdensity_gridding/L2800N5040/HYDRO/snap_" + snap + "/" + metafile

# Open file
hdf = h5py.File(path, "r")

# Set up array to store counts
H_tot = np.zeros_like(bin_cents)

# Loop over cells
for key in hdf.keys():

    # Skip meta data
    if key in ["Parent", "Delta_grid"]:
        continue

    print(key, np.min(hdf[key]["grid"][...]), np.max(hdf[key]["grid"][...]))

    # Get counts for this cell
    H, _ = np.histogram(hdf[key]["grid"][...], bins=bin_edges)

    # Add counts to main array
    H_tot += H

fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy()

ax.plot(bin_cents, H_tot, label="L2800N5040_HYDRO_2Mpc")

ax.set_xlabel("$\delta$")
ax.set_ylabel("$N$")

fig.savefig("plots/overdensity_hist_" + snap + ".png", bbox_inches="tight")

plt.close()
