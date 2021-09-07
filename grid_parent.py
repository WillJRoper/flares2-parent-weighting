import numpy as np
import h5py


# filepath = "/cosma8/data/dp004/jlvc76/FLAMINGO/ScienceRuns/" \
#            "DMO/L3200N5760/snapshots/flamingo_0000/flamingo_0000.hdf5"
filepath = "/Users/willroper/Documents/3D Printing/Python/ani_hydro_1379.hdf5"

# Open HDF5 file
hdf = h5py.File(filepath, "r")

# Get metadata
boxsize = hdf["Header"].attrs["BoxSize"]
z = hdf["Header"].attrs["Redshift"]
dm_mass = hdf["PartType1/Masses"][0]

# Read in the cell centres and size
nr_cells = int(hdf["/Cells/Meta-data"].attrs["nr_cells"])
centres = hdf["/Cells/Centres"][:, :]

# Set up overdensity grid array
zoom_width = 25
zoom_ncells = 32
cell_width = zoom_width / zoom_ncells
ncells = np.int32(boxsize / cell_width)

cell_volume = cell_width**3

mass_grid = np.zeros(ncells)

print("Boxsize:", boxsize)
print("Redshift:", z)

for icell in range(nr_cells):

    # Print the position of the centre of the cell of interest
    centre = hdf["/Cells/Centres"][icell, :]
    print("Centre of the cell:", centre)

    # Retrieve the offset and counts
    my_offset = hdf["/Cells/OffsetsInFile/PartType1"][icell]
    my_count = hdf["/Cells/Counts/PartType1"][icell]

    # Get the densities of the particles in this cell
    poss = hdf["/PartType1/Coordinates"][my_offset: my_offset + my_count]

    for ipart in range(poss.shape[0]):

        this_pos = poss[ipart, :]

        i = int(this_pos[0] / cell_width)
        j = int(this_pos[1] / cell_width)
        k = int(this_pos[2] / cell_width)

        mass_grid[i, j, k] += dm_mass

hdf.close()

# Convert mass grid to overdensities
ovden_grid = mass_grid / cell_width * 10**10

print(ovden_grid)
print(ovden_grid.min(), ovden_grid.max())

