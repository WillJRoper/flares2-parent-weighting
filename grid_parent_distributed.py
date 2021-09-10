import numpy as np
import h5py
import sys


filepath = "/cosma8/data/dp004/jlvc76/FLAMINGO/ScienceRuns/DMO/L3200N5760/snapshots/flamingo_0000/flamingo_0000."
# filepath = "/Users/willroper/Documents/3D Printing/Python/ani_hydro_1379.hdf5"

def get_and_write_ovdengrid(filepath, zoom_ncells, zoom_width, njobs, jobid):

    total_jobs = 1199
    job_bins = np.linspace(0, total_jobs, njobs, dtype=int)
    print(str(jobid) + ":", "I have files",
          job_bins[jobid], "-", job_bins[jobid + 1])

    for ifile in range(job_bins[jobid], job_bins[jobid + 1]):

        # Open HDF5 file
        hdf = h5py.File(filepath + str(ifile) + ".hdf5", "r")

        # Get metadata
        boxsize = hdf["Header"].attrs["BoxSize"]
        z = hdf["Header"].attrs["Redshift"]
        nparts = hdf["Header"].attrs["NumPart_Total"]
        nparts_this_file = hdf["Header"].attrs["NumPart_ThisFile"]

        # Set up overdensity grid array
        zoom_width = np.array([zoom_width, zoom_width, zoom_width])
        cell_width = zoom_width / zoom_ncells
        ncells = np.int32(boxsize / cell_width)

        bins = np.linspace(0, nparts_this_file, 100, dtype=int)
        print("N_part:", nparts)
        print("NpartThisFile:", nparts_this_file)
        print("Boxsize:", boxsize)
        print("Redshift:", z)
        print("Grid Cell Width:", cell_width)
        print("Grid NCells:", ncells)

        grid_poss = np.zeros((nparts_this_file, 3), dtype=int)

        for ibin in range(len(bins[:-1])):

            print(ibin, "of", nparts_this_file)

            # Get the densities of the particles in this cell
            poss = hdf["/PartType1/Coordinates"][bins[ibin]: bins[ibin + 1], :]

            i = np.int32(poss[:, 0] / cell_width[0])
            j = np.int32(poss[:, 1] / cell_width[1])
            k = np.int32(poss[:, 2] / cell_width[2])

            grid_poss[bins[ibin]: bins[ibin + 1], 0] = i
            grid_poss[bins[ibin]: bins[ibin + 1], 1] = j
            grid_poss[bins[ibin]: bins[ibin + 1], 2] = k

        hdf.close()

        try:
            grid_hdf = h5py.File("data/parent_ovden_grid_" + str(jobid) + ".hdf5",
                            "r+")
        except OSError as e:
            print(e)
            grid_hdf = h5py.File("data/parent_ovden_grid_" + str(jobid) + ".hdf5",
                            "w")
        try:
            zoom_grp = grid_hdf.create_group(str(zoom_width) + "_"
                                             + str(zoom_ncells))
        except OSError as e:
            print(e)
            del grid_hdf[str(zoom_width) + "_" + str(zoom_ncells)]
            zoom_grp = grid_hdf.create_group(str(zoom_width) + "_"
                                             + str(zoom_ncells))
        zoom_grp.create_dataset("grid-pos", data=grid_poss,
                                compression="gzip")
        grid_hdf.close()

njobs = int(sys.argv[2])
jobid = int(sys.argv[1])

zoom_width = 25
for zoom_ncells in [16, 32, 64, 128, 256]:
    print("================== zoom width, zoom_ncells =",
          zoom_width, zoom_ncells, "==================")
    get_and_write_ovdengrid(filepath, zoom_ncells, zoom_width, njobs, jobid)





