import numpy as np
import nanosurf as nsf
from surfalize.surface import Surface

def read_nid(filename):
    nid_file = nsf.util.nid_reader.NIDFileReader()
    if not nid_file.read(filename):
        raise ValueError(f"Failed to read the NID file: {filename}")

    # Topography data
    topo_data = nid_file.data.image.forward["Z-Axis"]

    # Offset and scale Z to µm
    topo_data -= np.min(topo_data)
    topo_data *= 1e6

    # Read scan info and convert to µm
    x_info = nid_file.data_param.image_dim_info["X"]
    y_info = nid_file.data_param.image_dim_info["Y"]
    size_x = x_info['range'][0] * 1e6
    size_y = y_info['range'][0] * 1e6

    shape = topo_data.shape  # (rows, cols)
    step_x = size_x / shape[1]  # correct: X → cols
    step_y = size_y / shape[0]  # correct: Y → rows

    return Surface(topo_data, step_x=step_x, step_y=step_y)


#Testing
import glob
files = glob.glob('*.nid')
surface = read_nid(files[1])
surface.show()