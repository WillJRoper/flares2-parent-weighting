import sys

import numpy as np
import h5py
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt



# Set the seed
np.random.seed(42)

# Define all snapshot codes
snaps = [str(i).zfill(4) for i in range(0, 23)]

# Define the selection snapshot
snap = sys.argv[1].zfill(4)

# Get the simulation "tag"
sim_tag = sys.argv[2]

# Get the simulation "type"
sim_type = sys.argv[3]

# Define initial kernel width
ini_kernel_width = int(sys.argv[4])

# Define the number of high and low density regions we want
nhigh, nlow = int(sys.argv[5]), int(sys.argv[6])

# Define the number of regions needed
nregions = int(sys.argv[7])

# Define output paths
metafile = "overdensity_" + sim_tag + "_" + sim_type + "_snap%s.hdf5" % snap
outdir = "/cosma7/data/dp004/FLARES/FLARES-2/Parent/" \
         "overdensity_gridding/" + sim_tag + "/" + sim_type + "/snap_" + snap + "/"

# Define path to file
file = "smoothed_" + metafile.split(".")[
    0] + "_kernel%d.hdf5" % ini_kernel_width
path = outdir + file

# Open file
hdf = h5py.File(path, "r")

kernel_width = hdf.attrs["Kernel_Width"]
half_kernel_width = kernel_width / 2
grid = hdf["Region_Overdensity"]
z = hdf.attrs["Parent_Redshift"]
# grid_std = hdf["Region_Overdensity_Stdev"][...]
centres = hdf["Region_Centres"]
sinds = hdf["Sorted_Indices"][...][::-1]

print("Selecting %d high density regions, %d low density regions, "
      "and %d random regions at z=%f" % (nhigh, nlow,
                                         nregions - nhigh - nlow, z))

# Minimum distance between regions
r = half_kernel_width[0] / np.cos(np.pi / 4) * 2

# Create lists to store the region data
region_centres = [centres[sinds[0]], ]
region_inds = [sinds[0]]

# Loop until we have nregions distinct regions
ind = 0
low_ind = 0
while len(region_inds) < nregions:

    # If we have the 50 highest overdensities and 30
    # lowest get a random region
    if len(region_inds) > nhigh + nlow:
        ind = np.random.randint(low=0, high=sinds.size)
    elif len(region_inds) > nhigh:
        low_ind -= 1
        ind = low_ind
    else:
        ind += 1

    # Get a region
    region_ind = sinds[ind]

    # Get this regions centre
    cent = centres[region_ind, :]

    # Build kd tree of current region centers
    tree = cKDTree(region_centres)

    # Is the region too close to an already selected region?
    close_regions = tree.query_ball_point(np.array([cent, ]), r=r)

    # If not we found no neighbours and can add it to the list
    if len(close_regions) == 0:
        region_inds.append(region_ind)
        region_centres.append(cent)
        print("Found %d regions", len(region_inds))

hdf.close()

# ============ Get overdensities for all outputs for these regions ============

# Set up dictionaries to store results
zs = {}
ovdens = {}

for ind in region_inds:
    zs[ind] = []
    ovdens[ind] = []
    for snap in snaps:
        # Define output paths
        metafile = "overdensity_" + sim_tag + "_" + sim_type + "_snap%s.hdf5" % snap
        outdir = "/cosma7/data/dp004/FLARES/FLARES-2/Parent/" \
                 "overdensity_gridding/" + sim_tag + "/" + sim_type + "/snap_" + snap + "/"

        # Define path to file
        file = "smoothed_" + metafile.split(".")[
            0] + "_kernel%d.hdf5" % ini_kernel_width
        path = outdir + file

        # Open file
        hdf = h5py.File(path, "r")

        # Get the current redshift
        z = hdf.attrs["Parent_Redshift"]

        # Get the overdensity for this region
        ovden = hdf["Region_Overdensity"][ind]

        # Store these values
        zs[ind].append(z)
        ovdens[ind].append(ovden)



fig = plt.figure()
ax = fig.add_subplot(111)

for i in zs:
    ax.plot(zs[i], np.log10(ovdens[i]))

ax.set_ylabel("$\log_{10}(1 + \delta)$")
ax.set_xlabel("$z$")

fig.savefig("plots/region_select_time_series" + str(ini_kernel_width) + "_"
            + sim_tag + "_" + sim_type + ".png", bbox_inches="tight")

