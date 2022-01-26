import numpy as np
import h5py


def smoothed_grid(snap, ini_kernel_width):

    # Define path to file
    metafile = "overdensity_L2800N5040_HYDRO_snap%s.hdf5" % snap
    path = "/cosma7/data/dp004/FLARES/FLARES-2/Parent/" \
           "overdensity_gridding/L2800N5040/HYDRO/snap_" + snap + "/" \
           + metafile

    # Open file
    hdf = h5py.File(path, "r")

    # Get simulation metadata
    boxsize = hdf["Parent"].attrs["Boxsize"]
    mean_density = hdf["Parent"].attrs["Mean_Density"]
    grid_cell_width = hdf["Delta_grid"].attrs["Cell_Width"]
    grid_cell_vol = grid_cell_width ** 3

    # Compute actual kernel width
    n_kernels = boxsize / ini_kernel_width

    # Loop over cells
    for key in hdf.keys():

        # Skip meta data
        if key in ["Parent", "Delta_grid"]:
            continue



