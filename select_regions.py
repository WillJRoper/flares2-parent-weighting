import sys

import numpy as np
import h5py


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

# Create lists to store the region data


hdf.close()
