import sys

import numpy as np
import h5py
from scipy.spatial import cKDTree


# Set the seed
np.random.seed(42)

# Define the number of regions needed
nregions = 400

# Define the selection snapshot
snap = sys.argv[1].zfill(4)

# Get the simulation "tag"
sim_tag = sys.argv[2]

# Get the simulation "type"
sim_type = sys.argv[3]

# Define initial kernel width
ini_kernel_width = int(sys.argv[4])

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
# grid_std = hdf["Region_Overdensity_Stdev"][...]
centres = hdf["Region_Centres"]
sinds = hdf["Sorted_Indices"][::-1]

# Minimum distance between regions
r = half_kernel_width / np.cos(np.pi / 4) * 2

# Create lists to store the region data
region_centres = [centres[sinds[0]], ]
region_inds = [sinds[0]]

# Loop until we have nregions distinct regions
ind = 0
low_ind = 0
while len(region_inds) < nregions:

    # If we have the 50 highest overdensities and 30
    # lowest get a random region
    if len(region_inds) > 80:
        ind = np.random.randint(low=0, high=sinds.size)
    elif len(region_inds) > 50:
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
    close_regions = tree.query_ball_point(cent, r=r)

    # If not we found no neighbours and can add it to the list
    if len(close_regions) == 0:
        region_inds.append(region_ind)
        region_centres.append(cent)

hdf.close()
