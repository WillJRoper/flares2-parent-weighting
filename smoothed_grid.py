import sys

import numpy as np
import h5py
from mpi4py import MPI


# Initializations and preliminaries
comm = MPI.COMM_WORLD  # get MPI communicator object
size = comm.size  # total number of processes
rank = comm.rank  # rank of this process
status = MPI.Status()  # get MPI status object


def get_smoothed_grid(snap, ini_kernel_width, outdir, rank, size):

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
    if rank == 0:
        print("Boxsize:", boxsize)
        print("Mean Density:", mean_density)
        print("Grid Cell Width:", grid_cell_width)
        print("Kernel Width:", kernel_width)
        print("Grid cells in kernel:", cells_per_kernel)
        print("Grid cells total:", ngrid_cells)

    # Get full parent grid
    # NOTE: this has a 1 grid cell pad region
    ovden_grid = hdf["Parent_Grid"][...]

    hdf.close()

    # Set up grid for the smoothed
    smooth_grid = np.zeros((ovden_grid.shape[0] - cells_per_kernel,
                            ovden_grid.shape[1] - cells_per_kernel,
                            ovden_grid.shape[2] - cells_per_kernel))
    smooth_vals = np.zeros((ovden_grid.shape[0] - cells_per_kernel)
                           * (ovden_grid.shape[1] - cells_per_kernel)
                           * (ovden_grid.shape[2] - cells_per_kernel))

    # Set up array to store centres and edges
    edges = np.zeros((ovden_grid.shape[0] * ovden_grid.shape[1]
                      * ovden_grid.shape[2], 3))
    centres = np.zeros((ovden_grid.shape[0] * ovden_grid.shape[1]
                        * ovden_grid.shape[2], 3))

    # Initialise pointer to i edges of kernel in overdensity grid
    low_i = -1

    # Get i coordinates for this rank
    my_is = np.linspace(0, smooth_grid.shape[0], size + 1)
    print("Rank: %d has %d cells" % (rank,
                                     (my_is[rank + 1] - my_is[rank]
                                      * smooth_grid.shape[1]
                                      * smooth_grid.shape[2])))

    # Loop over the smoothed cells
    ind = 0
    for i in range(my_is[rank], my_is[rank + 1]):
        low_i += 1
        low_j = -1  # initialise j edges of kernel in overdensity grid
        for j in range(smooth_grid.shape[1]):
            low_j += 1
            low_k = -1  # initialise k edges of kernel in overdensity grid
            for k in range(smooth_grid.shape[2]):
                low_k += 1

                print(i, j, k, low_i, low_j, low_k, ind)

                # Get the mean of these overdensities
                ovden_kernel = np.mean(ovden_grid[low_i: low_i
                                                         + cells_per_kernel,
                                                  low_j: low_j
                                                         + cells_per_kernel,
                                                  low_k: low_k
                                                         + cells_per_kernel])

                # Store ijks and edges (subtracting a cell for the pad region)
                edges[ind, :] = np.array([low_i * grid_cell_width[0]
                                          - grid_cell_width[0],
                                          low_j * grid_cell_width[1]
                                          - grid_cell_width[1],
                                          low_k * grid_cell_width[2]
                                          - grid_cell_width[2]])
                centres[ind, :] = edges[ind, :] + (kernel_width / 2)

                # Store smoothed value in both the grid for
                # visualisation and array
                smooth_vals[ind] = ovden_kernel
                smooth_grid[i, j, k] = ovden_kernel

                # Increment index counter
                ind += 1

    # Set up the outpath for each rank file
    outpath = outdir + "smoothed_" + metafile.split(".")[0] \
              + "_kernel%d_rank%d.hdf5" % (ini_kernel_width, rank)

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
    hdf.create_dataset("Smoothed_Region_Centres", data=centres,
                       shape=centres.shape, dtype=centres.dtype,
                       compression="lzf")
    hdf.close()

    comm.Barrier()

    if rank == 0:

        # Set up the outpaths
        outpath = outdir + "smoothed_" + metafile.split(".")[0] \
                  + "_kernel%d.hdf5" % ini_kernel_width
        outpath0 = outdir + "smoothed_" + metafile.split(".")[0] \
                  + "_kernel%d_rank0.hdf5" % ini_kernel_width

        # Open file to combine results
        hdf = h5py.File(outpath, "w")

        # Open rank 0 file to get metadata
        hdf_rank0 = h5py.File(outpath0, "r")  # open rank 0 file
        for key in hdf_rank0.attrs.keys():
            hdf.attrs[key] = hdf_rank0.attrs[key]  # write attrs

        hdf_rank0.close()

        for other_rank in range(size):

            # Set up the outpath for each rank file
            rank_outpath = outdir + "smoothed_" + metafile.split(".")[0]\
                           + "_kernel%d_rank%d.hdf5" % (ini_kernel_width,
                                                        other_rank)

            hdf_rank0 = h5py.File(outpath, "r")  # open rank 0 file



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
    get_smoothed_grid(snap, ini_kernel_width, outdir, rank, size)


