import sys

import numpy as np
import h5py


def get_smoothed_grid(snap, ini_kernel_width, outdir):

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
    ngrid_cells = hdf["Delta_grid"].attrs["Ncells_Total"]
    grid_cell_width = hdf["Delta_grid"].attrs["Cell_Width"]
    grid_cell_vol = grid_cell_width ** 3

    # Compute actual kernel width and volume
    cells_per_kernel = np.int32(np.ceil(ini_kernel_width / grid_cell_width[0]))
    kernel_width = cells_per_kernel * grid_cell_width

    # Print some nice things
    print("Boxsize:", boxsize)
    print("Mean Density:", mean_density)
    print("Grid Cell Width:", grid_cell_width)
    print("Kernel Width:", kernel_width)
    print("Grid cells in kernel:", cells_per_kernel)

    # Set up mass grid
    ovden_grid = np.zeros((int(ngrid_cells[0]), int(ngrid_cells[1]),
                          int(ngrid_cells[2])))

    # Loop over simulation cells to build single grid
    for key in hdf.keys():

        # Skip meta data
        if key in ["Parent", "Delta_grid"]:
            continue

        # Get this cells hdf5 group and edges
        cell_grp = hdf[key]
        edges = cell_grp.attrs["Sim_Cell_Edges"]

        # Get the ijk grid coordinates associated to this cell
        i = int(edges[0] / grid_cell_width[0])
        j = int(edges[0] / grid_cell_width[0])
        k = int(edges[0] / grid_cell_width[0])

        # Get the overdensity grid and convert to mass
        grid = cell_grp["grid"][...]

        ovden_grid[i: i + grid.shape[0],
                   j: j + grid.shape[1],
                   k: k + grid.shape[2]] = grid

    hdf.close()

    print("Created Grid")

    # Set up grid for the smoothed
    smooth_grid = np.zeros((ovden_grid.shape[0] - cells_per_kernel,
                            ovden_grid.shape[1] - cells_per_kernel,
                            ovden_grid.shape[2] - cells_per_kernel))
    smooth_vals = np.zeros(ovden_grid.shape[0] - cells_per_kernel
                           * ovden_grid.shape[1] - cells_per_kernel
                           * ovden_grid.shape[2] - cells_per_kernel)

    # Set up array to store ijks and edges
    edges = np.zeros((ovden_grid.shape[0] * ovden_grid.shape[1]
                      * ovden_grid.shape[2], 3))

    # Initialise pointers to edges of kernel in overdensity grid
    low_i = -1
    low_j = -1
    low_k = -1

    # Loop over the smoothed cells
    ind = 0
    for i in range(smooth_grid.shape[0]):
        low_i += 1
        for j in range(smooth_grid.shape[1]):
            low_j += 1
            for k in range(smooth_grid.shape[2]):
                low_k += 1

                # Get the mean of these overdensities
                ovden_kernel = np.mean(ovden_grid[low_i: low_i
                                                         + cells_per_kernel,
                                                  low_j: low_j
                                                         + cells_per_kernel,
                                                  low_k: low_k
                                                         + cells_per_kernel])
                # Store ijks and edges
                edges[ind, :] = np.array([low_i * grid_cell_width,
                                            low_j * grid_cell_width,
                                            low_k * grid_cell_width])

                # Store smoothed value in both the grid for
                # visualisation and array
                smooth_vals[ind] = ovden_kernel
                smooth_grid[i, j, k] = ovden_kernel

                # Increment index counter
                ind += 1

    # Set up the outpath
    outpath = outdir + "smoothed_" + metafile.split(".")[0] \
              + "_%d.hdf5" % ini_kernel_width

    # Write out the results of smoothing
    hdf = h5py.File(outpath, "w")
    hdf.attrs["Kernel_Width"] = kernel_width
    hdf.create_dataset("Smoothed_Grid", data=smooth_grid,
                       shape=smooth_grid.shape, dtype=smooth_grid.dtype,
                       compression="lzf")
    hdf.create_dataset("Smoothed_Overdensity", data=smooth_vals,
                       shape=smooth_vals.shape, dtype=smooth_vals.dtype,
                       compression="lzf")
    hdf.create_dataset("Smoothed_Region_Edges", data=edges,
                       shape=edges.shape, dtype=edges.dtype,
                       compression="lzf")
    hdf.close()


if __name__ == "__main__":

    # Get the commandline argument for which snapshot
    num = int(sys.argv[1])

    # Extract the snapshot string
    snaps = [str(i).zfill(4) for i in range(0, 19)]
    snap = snaps[num]

    # Define output paths
    outdir = "/cosma7/data/dp004/FLARES/FLARES-2/Parent/" \
             "overdensity_gridding/L2800N5040/HYDRO/snap_" + snap + "/"

    # Run smoothing
    ini_kernel_width = 25  # in cMpc
    get_smoothed_grid(snap, ini_kernel_width, outdir)


