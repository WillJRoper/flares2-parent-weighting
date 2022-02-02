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

    # Get the simulation "tag"
    sim_tag = sys.argv[2]

    # Get the simulation "type"
    sim_type = sys.argv[3]

    # Define path to file
    metafile = "overdensity_" + sim_tag + "_" + sim_type + "_snap%s.hdf5" % snap
    path = "/cosma7/data/dp004/FLARES/FLARES-2/Parent/" \
           "overdensity_gridding/" + sim_tag + "/" + sim_type + "/snap_" + snap + "/" \
           + metafile

    # Open file
    hdf = h5py.File(path, "r")

    # Get metadata
    boxsize = hdf["Parent"].attrs["Boxsize"]
    mean_density = hdf["Parent"].attrs["Mean_Density"]
    ngrid_cells = hdf["Delta_grid"].attrs["Ncells_Total"]
    grid_cell_width = hdf["Delta_grid"].attrs["Cell_Width"]

    # Compute actual kernel width 
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
    ovden_grid = hdf["Parent_Grid"][...]

    hdf.close()

    # Set up arrays for the smoothed grid
    smooth_grid = np.zeros((ovden_grid.shape[0] - cells_per_kernel,
                            ovden_grid.shape[1] - cells_per_kernel,
                            ovden_grid.shape[2] - cells_per_kernel))
    smooth_vals = np.zeros(smooth_grid[0] * smooth_grid[1] * smooth_grid[2])

    # Set up array to store centres and edges
    edges = np.zeros((smooth_vals.size, 3))
    centres = np.zeros((smooth_vals.size, 3))

    # Get the list of simulation cell indices and the associated ijk references
    all_cells = []
    i_s = []
    j_s = []
    k_s = []
    for i in range(smooth_grid.shape[0]):
        for j in range(smooth_grid.shape[1]):
            for k in range(smooth_grid.shape[2]):
                cell = (k + smooth_grid.shape[2] * (j + smooth_grid.shape[1] * i))
                all_cells.append(cell)
                i_s.append(i)
                j_s.append(j)
                k_s.append(k)

    # Find the cells and simulation ijk grid references
    # that this rank has to work on
    rank_cells = np.linspace(0, len(all_cells), size + 1, dtype=int)
    my_cells = all_cells[rank_cells[rank]: rank_cells[rank + 1]]
    my_i_s = i_s[rank_cells[rank]: rank_cells[rank + 1]]
    my_j_s = j_s[rank_cells[rank]: rank_cells[rank + 1]]
    my_k_s = k_s[rank_cells[rank]: rank_cells[rank + 1]]

    print("Rank: %d has %d cells" % (rank, len(my_cells)))

    # Loop over the smoothed cells
    smooth_cdim = smooth_grid.shape
    for i in my_i_s:
        low_i = i
        for j in my_j_s:
            low_j = j
            for k in my_k_s:
                low_k = k

                # Get the index for this smoothed grid cell
                ind = (k + smooth_cdim[2] * (j + smooth_cdim[1] * i))

                print(i, j, k, low_i, low_j, low_k,
                      low_i + cells_per_kernel, low_j + cells_per_kernel,
                      low_k + cells_per_kernel, ind)

                # Get the mean of these overdensities
                ovden_kernel = np.mean(ovden_grid[low_i: low_i
                                                         + cells_per_kernel,
                                                  low_j: low_j
                                                         + cells_per_kernel,
                                                  low_k: low_k
                                                         + cells_per_kernel])

                # Store ijks and edges
                edges[ind, :] = np.array([low_i * grid_cell_width[0],
                                          low_j * grid_cell_width[1],
                                          low_k * grid_cell_width[2]])
                centres[ind, :] = edges[ind, :] + (kernel_width / 2)

                # Store smoothed value in both the grid for
                # visualisation and array
                smooth_vals[ind] = ovden_kernel
                smooth_grid[i, j, k] = ovden_kernel

    # Set up the outpath for each rank file
    outpath = outdir + "smoothed_" + metafile.split(".")[0] \
              + "_kernel%d_rank%d.hdf5" % (ini_kernel_width, rank)

    # Write out the results of smoothing
    hdf = h5py.File(outpath, "w")
    hdf.attrs["Kernel_Width"] = kernel_width
    hdf.create_dataset("Smoothed_Grid", data=smooth_grid,
                       shape=smooth_grid.shape, dtype=smooth_grid.dtype,
                       compression="lzf")
    hdf.create_dataset("Smoothed_Array", data=smooth_vals,
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

        #  ========== Define arrays to store the collected results ==========

        # Set up arrays for the smoothed grid
        final_smooth_grid = np.zeros((ovden_grid.shape[0] - cells_per_kernel,
                                      ovden_grid.shape[1] - cells_per_kernel,
                                      ovden_grid.shape[2] - cells_per_kernel))
        final_smooth_vals = np.zeros(smooth_grid[0]
                                     * smooth_grid[1]
                                     * smooth_grid[2])

        # Set up array to store centres and edges
        final_edges = np.zeros((smooth_vals.size, 3))
        final_centres = np.zeros((smooth_vals.size, 3))

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

            hdf_rank = h5py.File(rank_outpath, "r")  # open rank 0 file

            # Combien this ranks results into the final array
            final_smooth_vals += hdf_rank["Smoothed_Array"][...]
            final_smooth_grid += hdf_rank["Smoothed_Grid"][...]
            final_edges += hdf_rank["Smoothed_Region_Edges"][...]
            final_centres += hdf_rank["Smoothed_Region_Centres"][...]

            hdf_rank.close()

        hdf.create_dataset("Smoothed_Grid",
                           data=final_smooth_grid,
                           shape=final_smooth_grid.shape,
                           dtype=final_smooth_grid.dtype,
                           compression="lzf")
        hdf.create_dataset("Smoothed_Array",
                           data=final_smooth_vals,
                           shape=final_smooth_vals.shape,
                           dtype=final_smooth_vals.dtype,
                           compression="lzf")
        hdf.create_dataset("Smoothed_Region_Edges",
                           data=final_edges,
                           shape=final_edges.shape,
                           dtype=final_edges.dtype,
                           compression="lzf")
        hdf.create_dataset("Smoothed_Region_Centres",
                           data=final_centres,
                           shape=final_centres.shape,
                           dtype=final_centres.dtype,
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
             "overdensity_gridding/" + sim_tag + "/" + sim_type + "/snap_" + snap + "/"

    # Run smoothing
    ini_kernel_width = 25  # in cMpc
    get_smoothed_grid(snap, ini_kernel_width, outdir, rank, size)


