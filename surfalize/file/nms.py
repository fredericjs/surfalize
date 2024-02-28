import struct
import numpy as np

from ..exceptions import CorruptedFileError

HEADER_SIZE = 3468
OFFSET_Z = 16
OFFSET_POINTS = 1368
OFFSET_SPACING = 1376


def read_nms(filepath):
    with open(filepath, 'rb') as file:
        file.seek(OFFSET_Z, 0)
        zmin, zmax = struct.unpack('<2d', file.read(16))
        file.seek(OFFSET_POINTS, 0)
        nx, ny = struct.unpack('<2I', file.read(8))
        file.seek(OFFSET_SPACING, 0)
        dx, dy = struct.unpack('<2d', file.read(16))
        file.seek(HEADER_SIZE, 0)

        data = np.fromfile(file, dtype=np.uint16, count=nx * ny)
        nonmeasured_points_mask = (data == 0)
        data = data / (2 ** 16 - 2) * (zmax - zmin) + zmax
        data[nonmeasured_points_mask] = np.nan
        data = data.reshape(ny, nx)

        step_x = dx * 1e-3
        step_y = dy * 1e-3

    return data, step_x, step_y