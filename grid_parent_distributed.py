import os
import sys

import h5py
import numpy as np
from mpi4py import MPI

# Initializations and preliminaries
comm = MPI.COMM_WORLD  # get MPI communicator object
size = comm.size  # total number of processes
rank = comm.rank  # rank of this process
status = MPI.Status()  # get MPI status object


def get_cellid(cdim, i, j, k):
    return (k + cdim[2] * (j + cdim[1] * i))


def get_ovdengrid(filepath, outpath, size, rank, target_grid_width=2.0):

    # Open HDF5 file
    hdf = h5py.File(filepath, "r")

    # Get metadata
    boxsize = hdf["Header"].attrs["BoxSize"]
    z = hdf["Header"].attrs["Redshift"]
    nparts = hdf["/PartType1/Masses"].size
    pmass = hdf["Header"].attrs["InitialMassTable"][1]
    cdim = hdf["Cells/Meta-data"].attrs["dimension"]
    ncells = hdf["/Cells/Meta-data"].attrs["nr_cells"]
    cell_width = hdf["Cells/Meta-data"].attrs["size"]

    # Calculate the mean density
    tot_mass = nparts * pmass
    mean_density = tot_mass / (boxsize[0] * boxsize[1] * boxsize[2])

    # Set up overdensity grid properties
    ovden_cdim = np.int32(cell_width / target_grid_width)
    ovden_cell_width = cell_width / ovden_cdim
    full_grid_ncells = np.int32(boxsize / ovden_cell_width)
    ovden_cell_volume = (ovden_cell_width[0] * ovden_cell_width[1]
                         * ovden_cell_width[2])

    # Print some fun stuff
    if rank == 0:
        print("Boxsize:", boxsize)
        print("Redshift:", z)
        print("Npart:", nparts)
        print("Particle Mass:", pmass)
        print("Number of simulation cells:", ncells)
        print("Mean Density:", mean_density)
        print("Sim Cell Width:", cell_width)
        print("Grid Cell Width:", ovden_cell_width)
        print("Grid Cell Volume:", ovden_cell_width[0]
              * ovden_cell_width[1] * ovden_cell_width[2])
        print("Parent Grid NCells:", full_grid_ncells + 2)

    # Get the list of simulation cell indices and the associated ijk references
    all_cells = []
    i_s = []
    j_s = []
    k_s = []
    for i in range(cdim[0]):
        for j in range(cdim[1]):
            for k in range(cdim[2]):
                cell = (k + cdim[2] * (j + cdim[1] * i))
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

    print("Rank=", rank, "- My Ncells=", len(my_cells))

    # Open HDF5 file
    hdf_out = h5py.File(outpath, "w")

    # Store some metadata about the parent box
    parent = hdf_out.create_group("Parent")
    parent.attrs["Boxsize"] = boxsize
    parent.attrs["Redshift"] = z
    parent.attrs["Npart"] = nparts
    parent.attrs["Ncells"] = cdim
    parent.attrs["Cell_Width"] = cell_width
    parent.attrs["DM_Mass"] = pmass
    parent.attrs["Mean_Density"] = mean_density

    # Store some metadata about the overdensity grid
    parent = hdf_out.create_group("Delta_grid")
    parent.attrs["Cell_Width"] = ovden_cell_width
    parent.attrs["Ncells_Total"] = full_grid_ncells
    parent.attrs["Ncells_PerSimCell"] = ovden_cdim

    # Create cells group
    cells_grp = hdf_out.create_group("Cells")

    # Loop over cells calculating the overdensity grid
    for i, j, k, my_cell in zip(my_i_s, my_j_s, my_k_s, my_cells):

        # Set up array to store this cells overdensity grid
        # with pad region of 1 cell
        mass_grid_this_cell = np.zeros((ovden_cdim[0] + 2,
                                        ovden_cdim[1] + 2,
                                        ovden_cdim[2] + 2))

        # Retrieve the offset and counts for this cell
        my_offset = hdf["/Cells/OffsetsInFile/PartType1"][my_cell]
        my_count = hdf["/Cells/Counts/PartType1"][my_cell]

        # Define the edges of this cell with pad region
        my_edges = np.array([(i * cell_width[0]) - ovden_cell_width[0],
                             (j * cell_width[1]) - ovden_cell_width[1],
                             (k * cell_width[2]) - ovden_cell_width[2]])

        if my_count > 0:
            poss = hdf["/PartType1/Coordinates"][
                   my_offset:my_offset + my_count, :] - my_edges
            masses = hdf["/PartType1/Masses"][
                     my_offset:my_offset + my_count]

            # Handle the edge case where a particle has been
            # wrapped outside the box
            poss[poss > cell_width] -= boxsize[0]
            poss[poss < -cell_width] += boxsize[0]

            # Compute overdensity grid ijk references
            ovden_ijk = np.int32(poss / ovden_cell_width)

            # Store the mass in each grid cell
            for ii, jj, kk, m in zip(ovden_ijk[:, 0], ovden_ijk[:, 1],
                                     ovden_ijk[:, 2], masses):
                mass_grid_this_cell[ii, jj, kk] += m

            # Convert the mass entries to overdensities
            # (\delta(x) = (\rho(x) - \bar{\rho}) / \bar{\rho})
            ovden_grid_this_cell = ((mass_grid_this_cell / ovden_cell_volume)
                                    / mean_density)

        else:
            ovden_grid_this_cell = mass_grid_this_cell

        # Create a group for this cell
        this_cell = cells_grp.create_group(str(i) + "_" + str(j)
                                           + "_" + str(k))
        this_cell.attrs["Sim_Cell_Index"] = my_cell
        this_cell.attrs["Sim_Cell_Edges"] = my_edges
        this_cell.create_dataset("grid", data=ovden_grid_this_cell,
                                 shape=ovden_grid_this_cell.shape,
                                 dtype=ovden_grid_this_cell.dtype,
                                 compression="lzf")

    hdf_out.close()
    hdf.close()


def create_meta_file(metafile, rankfile_dir, outfile_without_rank, size):
    # Change to the data directory to ensure relative paths work
    os.chdir(rankfile_dir)

    # Write the metadata from rank 0 file to meta file
    rank0file = (outfile_without_rank
                 + "rank%s.hdf5" % "0".zfill(4))  # get rank 0 file
    hdf_rank0 = h5py.File(rank0file, "r")  # open rank 0 file
    hdf_meta = h5py.File(metafile, "w")  # create meta file
    for root_key in ["Parent", "Delta_grid"]:  # loop over root groups
        grp = hdf_meta.create_group(root_key)
        for key in hdf_rank0[root_key].attrs.keys():
            grp.attrs[key] = hdf_rank0[root_key].attrs[key]  # write attrs

    hdf_rank0.close()

    # Get the full parent overdensity grid dimensions
    # (this has a 1 cell pad region)
    ngrid_cells = hdf_meta["Delta_grid"].attrs["Ncells_Total"]

    # Get the grid cell width
    grid_cell_width = hdf_meta["Delta_grid"].attrs["Cell_Width"]

    # Set up full grid array
    full_grid = np.zeros((ngrid_cells[0], ngrid_cells[1], ngrid_cells[2]))

    # Loop over rank files creating external links
    for other_rank in range(size):

        # Get the path to this rank
        rankfile = (outfile_without_rank
                    + "rank%s.hdf5" % str(other_rank).zfill(4))

        # Open rankfile
        hdf_rank = h5py.File(rankfile, "r")

        # Loop over groups creating external links with relative path
        for key in hdf_rank["Cells"].keys():

            hdf_meta[key] = h5py.ExternalLink(rankfile, "/Cells/" + key)

            # Get this cells hdf5 group and edges
            cell_grp = hdf_rank["Cells"][key]
            edges = cell_grp.attrs["Sim_Cell_Edges"]

            # Get the ijk grid coordinates associated to this cell
            i = int(edges[0] / grid_cell_width[0]) + 1
            j = int(edges[1] / grid_cell_width[1]) + 1
            k = int(edges[2] / grid_cell_width[2]) + 1

            # Get the overdensity grid and convert to mass
            grid = cell_grp["grid"][...]

            print(grid.shape, i, j, k, i - 1, j - 1, k - 1,
                  i + grid.shape[0], j + grid.shape[1], k + grid.shape[2])

            full_grid[i: i + grid.shape[0], j: j + grid.shape[1], k: k + grid.shape[2]] = grid

        hdf_rank.close()

    hdf_meta.create_dataset("Parent_Grid", data=full_grid, shape=full_grid,
                            dtype=full_grid.dtype, compression="lzf")

    hdf_meta.close()


if __name__ == "__main__":

    # Get the commandline argument for which snapshot
    num = int(sys.argv[1])

    # Extract the snapshot string
    snaps = [str(i).zfill(4) for i in range(0, 19)]
    snap = snaps[num]

    # Define input path
    inpath = "/cosma8/data/dp004/jlvc76/FLAMINGO/ScienceRuns/L2800N5040/" \
             "HYDRO_FIDUCIAL/snapshots/flamingo_" + snap \
             + "/flamingo_" + snap + ".hdf5"

    # Define out file name
    outfile = "overdensity_L2800N5040_HYDRO_" \
              "snap%s_rank%s.hdf5" % (snap, str(rank).zfill(4))
    metafile = "overdensity_L2800N5040_HYDRO_snap%s.hdf5" % snap
    outfile_without_rank = "overdensity_L2800N5040_HYDRO_snap%s_" % snap

    # Define output paths
    out_dir = "/cosma7/data/dp004/FLARES/FLARES-2/Parent/" \
              "overdensity_gridding/L2800N5040/HYDRO/snap_" + snap
    if not os.path.isdir(out_dir) and rank == 0:
        os.mkdir(out_dir)
    outpath = out_dir + "/" + outfile  # Combine file and path
    ini_rankpath = out_dir + "/" + outfile_without_rank  # rankless string

    # # Get the overdensity grid for this rank
    # get_ovdengrid(inpath, outpath, size, rank, target_grid_width=2.0)

    # Ensure all files are finished writing
    comm.Barrier()

    # Create the meta file now we have each individual rank file
    if rank == 0:
        create_meta_file(metafile, out_dir, outfile_without_rank, size)
