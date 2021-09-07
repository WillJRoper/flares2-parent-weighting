import numpy as np
import h5py
import sys


filepath = "/cosma8/data/dp004/jlvc76/FLAMINGO/ScienceRuns/" \
           "DMO/L3200N5760/snapshots/flamingo_0000/flamingo_0000.hdf5"

def get_and_write_ovdengrid(filepath, zoom_ncells, zoom_width, njobs, jobid):

    # Open HDF5 file
    hdf = h5py.File(filepath, "r")

    # Get metadata
    boxsize = hdf["Header"].attrs["BoxSize"]
    z = hdf["Header"].attrs["Redshift"]
    nparts = hdf["Header"].attrs["NumPart_Total"]
    dm_npart = nparts[1]
    dm_mass = hdf["PartType1/Masses"][0]

    # Read in the cell centres and size
    nr_cells = int(hdf["/Cells/Meta-data"].attrs["nr_cells"])

    # Retrieve the offset and counts
    offsets = hdf["/Cells/OffsetsInFile/PartType1"][:]
    counts = hdf["/Cells/Counts/PartType1"][:]

    # Set up overdensity grid array
    cell_width = zoom_width / zoom_ncells
    ncells = np.int32(boxsize / cell_width)
    cell_volume = cell_width**3

    mass_grid = np.zeros(ncells)

    mean_density = dm_npart * dm_mass / boxsize**3

    print("N_part:", nparts)
    print("Boxsize:", boxsize)
    print("Redshift:", z)
    print("Cell Width:", cell_width)
    print("nr_cells:", nr_cells)
    print("grid cells:", ncells)

    cell_bins = np.linspace(0, nr_cells, njobs + 1, dtype=int)
    my_cells = np.arange(cell_bins[jobid], cell_bins[jobid + 1], 1, dtype=int)

    for icell in my_cells:

        # Retrieve the offset and counts
        my_offset = offsets[icell]
        my_count = counts[icell]

        # Get the densities of the particles in this cell
        poss = hdf["/PartType1/Coordinates"][my_offset: my_offset + my_count]

        i = np.int32(poss[:, 0] / cell_width)
        j = np.int32(poss[:, 1] / cell_width)
        k = np.int32(poss[:, 2] / cell_width)

        mass_grid[i, j, k] += dm_mass

    hdf.close()

    # # Convert mass grid to overdensities
    # den_grid = mass_grid / cell_volume
    # ovden_grid = (den_grid - mean_density) / mean_density

    try:
        hdf = h5py.File("data/parent_ovden_grid_" + str(jobid) + ".hdf5", "r+")
        grid_grp = hdf["Grids"]
    except OSError as e:
        print(e)
        hdf = h5py.File("data/parent_ovden_grid_" + str(jobid) + ".hdf5", "w")
        grid_grp = hdf.create_group("Grids")

    try:
        zoom_grp = grid_grp.create_group(str(zoom_width) + "_"
                                         + str(zoom_ncells))
    except OSError as e:
        print(e)
        del grid_grp[str(zoom_width) + "_" + str(zoom_ncells)]
        zoom_grp = grid_grp.create_group(str(zoom_width) + "_"
                                         + str(zoom_ncells))

    zoom_grp.attrs["zoom_ncells"] = zoom_ncells
    zoom_grp.attrs["zoom_width"] = zoom_width
    zoom_grp.attrs["cell_width"] = cell_width
    zoom_grp.create_dataset("Mass_grid", data=mass_grid)

    hdf.close()

njobs = int(sys.argv[2])
jobid = int(sys.argv[1])

zoom_width = 25
for zoom_ncells in [16, 32, 64, 128, 256]:
    print("================== zoom width, zoom_ncells =",
          zoom_width, zoom_ncells, "==================")
    get_and_write_ovdengrid(filepath, zoom_ncells, zoom_width, njobs, jobid)





