import numpy as np

from velociraptor import load as load_catalogue
from velociraptor.particles import load_groups
from velociraptor.swift.swift import to_swiftsimio_dataset


def read_halos_region(flamingos_dir=('/cosma8/data/dp004/flamingo/'
                                     'Runs/L2800N5040/DMO_FIDUCIAL/'),
                      snap='0019'
                     ):
    
    catalogue_name = f"vr_catalogue_{snap}"

    catalogue = load_catalogue((f"{flamingos_dir}/VR/catalogue_{snap}/"
                            f"{catalogue_name}.properties.0"))

    masses = catalogue.masses.mass_200crit
    coods = np.array([catalogue.positions.xc,
                      catalogue.positions.yc,
                      catalogue.positions.zc]).T
    # run cood filter here

    return masses, coods

# groups = load_groups((f"{flamingos_dir}/VR/catalogue_{snap}/"
#                       f"{catalogue_name}.catalog_groups.0"),
#                      catalogue=catalogue)
# 
# particles, unbound_particles = groups.extract_halo(halo_index=0)
# 
# data, mask = to_swiftsimio_dataset(
#     particles,
#     f"{flamingos_dir}/snapshots/flamingo_{snap}/{snapshot_name}.hdf5",
#     generate_extra_mask=True
# )

