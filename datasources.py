# Data source definitions.

root_path = ""

# FMI gzipped PGM files
fmi = {
    "root_path": "", # insert your data path here (relative to top-level root_path)
    "path_fmt": "%Y%m%d",
    "fn_pattern": "%Y%m%d%H%M_fmi.radar.composite.lowest_FIN_SUOMI1",
    "fn_ext": "pgm.gz",
    "importer": "fmi_pgm",
    "timestep": 5,
    "importer_kwargs": {
      "gzipped": True
    } 
}

# MeteoSwiss HDF5 files
mch = {
    "root_path": "", # insert your data path here (relative to top-level root_path)
    "path_fmt": "%Y%m%d",
    "fn_pattern": "RZC%y%j%H%M??.801",
    "fn_ext": "h5",
    "importer": "mch_hdf5",
    "timestep": 5,
    "importer_kwargs": {
        "qty": "RATE"
    }
}
