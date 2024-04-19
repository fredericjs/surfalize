import struct
import numpy as np
from .common import read_binary_layout, RawSurface

LAYOUT_HEADER = (
    ("Version", "8s", True),
    ("ManufacturerID", "10s", True),
    ("CreateDate", "12s", True),
    ("ModDate", "12s", True),
    ("NumPoints", "H", False),
    ("NumProfiles", "H", False),
    ("Xscale", "d", False),
    ("Yscale", "d", False),
    ("Zscale", "d", False),
    ("Zresolution", "d", False),
    ("CompressionType", "B", False),
    ("DataType", "B", False),
    ("ChecksumType", "B", False),
)


# Define a function to read the data section based on the DataType in the header
def read_data_section(file, header):
    num_points = header["NumPoints"]
    num_profiles = header["NumProfiles"]
    data_type = header["DataType"]

    format_map = {
        5: "h",  # INT16
        6: "I",  # UINT16
        7: "d",  # DOUBLE
        # Extend the map as needed
    }
    if data_type not in format_map:
        raise ValueError(f"Unsupported DataType in SDF file: {data_type}")

    data_format = format_map[data_type]
    item_size = struct.calcsize(data_format)
    data_size = item_size * num_points * num_profiles
    data_bytes = file.read(data_size)

    if len(data_bytes) != data_size:
        raise ValueError("Unexpected end of file or corrupt data section.")

    data = np.frombuffer(data_bytes, dtype=np.dtype(data_format))

    zscale = header["Zscale"]

    data = data * (zscale) / 10**-6
    data = data.reshape((num_profiles, num_points))
    # Scales are in meters
    step_x = header["Xscale"] / 10**-6
    step_y = header["Yscale"] / 10**-6

    return (data, step_x, step_y)


# Main function to read the SDF file
def read_binary_sdf(file_path, encoding="utf-8"):
    with open(file_path, "rb") as file:
        header = read_binary_layout(file, LAYOUT_HEADER, encoding=encoding)
        data, step_x, step_y = read_data_section(file, header)

    return RawSurface(data, step_x, step_y)
