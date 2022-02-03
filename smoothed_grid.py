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

    # Set up arrays for the region overdensities and their spread
    grid_shape = (ovden_grid.shape[0] - cells_per_kernel, 
                  ovden_grid.shape[1] - cells_per_kernel, 
                  ovden_grid.shape[2] - cells_per_kernel)
    region_vals = np.zeros((ovden_grid.shape[0] - cells_per_kernel) 
                           * (ovden_grid.shape[1] - cells_per_kernel)
                           * (ovden_grid.shape[2] - cells_per_kernel))
    region_stds = np.zeros(region_vals.size)

    # Set up array to store centres 
    centres = np.zeros((region_vals.size, 3))

    print("Created arrays with shapes:", region_vals.shape, region_stds.shape, 
          centres.shape)

    # Get the list of simulation cell indices and the associated ijk references
    i_s = np.zeros(region_vals.size, dtype=np.int16)
    j_s = np.zeros(region_vals.size, dtype=np.int16)
    k_s = np.zeros(region_vals.size, dtype=np.int16)
    ind = 0
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            for k in range(grid_shape[2]):
                i_s[ind] = i
                j_s[ind] = j
                k_s[ind] = k
                ind += 1

    # Find the cells and simulation ijk grid references
    # that this rank has to work on
    rank_cells = np.linspace(0, region_vals.size - 1, size + 1, dtype=int)

    print("Rank: %d has %d cells" % (rank,
                                     rank_cells[rank + 1] - rank_cells[rank]))

    # Loop over the smoothed cells
    for i in range(i_s[rank_cells[rank]], i_s[rank_cells[rank + 1]]):
        low_i = i
        for j in range(j_s[rank_cells[rank]], j_s[rank_cells[rank + 1]]):
            low_j = j
            for k in range(k_s[rank_cells[rank]], k_s[rank_cells[rank + 1]]):
                low_k = k

                # Get the index for this smoothed grid cell
                ind = (k + grid_shape[2] * (j + grid_shape[1] * i))

                # Get the mean of these overdensities
                ovden_kernel = np.mean(ovden_grid[low_i: low_i
                                                         + cells_per_kernel,
                                                  low_j: low_j
                                                         + cells_per_kernel,
                                                  low_k: low_k
                                                         + cells_per_kernel])
                ovden_kernel_std = np.std(ovden_grid[
                                          low_i: low_i + cells_per_kernel,
                                          low_j: low_j + cells_per_kernel,
                                          low_k: low_k + cells_per_kernel])

                # Store edges
                edges = np.array([i * grid_cell_width, 
                                  j * grid_cell_width, 
                                  k * grid_cell_width])
                centres[ind, :] = edges + (kernel_width / 2)

                # Store smoothed value in both the grid for
                # visualisation and array
                region_vals[ind] = ovden_kernel
                region_stds[ind] = ovden_kernel_std

    # Set up the outpath for each rank file
    outpath = outdir + "smoothed_" + metafile.split(".")[0] \
              + "_kernel%d_rank%d.hdf5" % (ini_kernel_width, rank)

    # Write out the results of smoothing
    hdf = h5py.File(outpath, "w")
    hdf.attrs["Kernel_Width"] = kernel_width
    hdf.create_dataset("Region_Overdensity", data=region_vals,
                       shape=region_vals.shape, dtype=region_vals.dtype,
                       compression="lzf")
    hdf.create_dataset("Region_Overdensity_Stdev", data=region_stds,
                       shape=region_stds.shape, dtype=region_stds.dtype,
                       compression="lzf")
    hdf.create_dataset("Region_Centres", data=centres,
                       shape=centres.shape, dtype=centres.dtype,
                       compression="lzf")
    hdf.close()

    comm.Barrier()

    if rank == 0:

        #  ========== Define arrays to store the collected results ==========

        # Set up arrays for the smoothed grid
        final_region_vals = np.zeros(grid_shape[0]
                                     * grid_shape[1]
                                     * grid_shape[2])
        final_region_stds = np.zeros((region_vals.size, 3))

        # Set up array to store centres
        final_centres = np.zeros((region_vals.size, 3))

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

            # Combine this rank's results into the final array
            final_region_vals += hdf_rank["Region_Overdensity"][...]
            
            final_region_stds += hdf_rank["Region_Overdensity_Stdev"][...]
            final_centres += hdf_rank["Region_Centres"][...]

            hdf_rank.close()

        hdf.create_dataset("Region_Overdensity",
                           data=final_region_vals,
                           shape=final_region_vals.shape,
                           dtype=final_region_vals.dtype,
                           compression="lzf")
        hdf.create_dataset("Region_Overdensity_Stdev",
                           data=final_region_stds,
                           shape=final_region_stds.shape,
                           dtype=final_region_stds.dtype,
                           compression="lzf")
        hdf.create_dataset("Region_Centres",
                           data=final_centres,
                           shape=final_centres.shape,
                           dtype=final_centres.dtype,
                           compression="lzf")

        hdf.close()


if __name__ == "__main__":

    # Get the commandline argument for which snapshot
    num = int(sys.argv[1])

    # Get the simulation "tag"
    sim_tag = sys.argv[2]

    # Get the simulation "type"
    sim_type = sys.argv[3]

    # Extract the snapshot string
    snaps = [str(i).zfill(4) for i in range(0, 19)]
    snap = snaps[num]

    # Define output paths
    outdir = "/cosma7/data/dp004/FLARES/FLARES-2/Parent/" \
             "overdensity_gridding/" + sim_tag + "/" + sim_type + "/snap_" + snap + "/"

    # Run smoothing
    ini_kernel_width = 25  # in cMpc
    get_smoothed_grid(snap, ini_kernel_width, outdir, rank, size)


