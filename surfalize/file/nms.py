import struct
import numpy as np
import dateutil
from .common import RawSurface, FileHandler, read_array
from datetime import datetime


HEADER_SIZE = 3468
OFFSET_Z = 16
OFFSET_DATE = 1076
OFFSET_POINTS = 1368
OFFSET_SPACING = 1376

DTYPE_HEIGHT = np.uint16
DTYPE_IMG = np.uint8

@FileHandler.register_reader(suffix='.nms')
def read_nms(filehandle, read_image_layers=False, encoding='utf-8'):
    filehandle.seek(OFFSET_Z, 0)
    zmin, zmax = struct.unpack('<2d', filehandle.read(16))
    filehandle.seek(OFFSET_DATE, 0)
    date = dateutil.parser.parse(filehandle.read(16).decode())
    filehandle.seek(OFFSET_POINTS, 0)
    nx, ny = struct.unpack('<2I', filehandle.read(8))
    filehandle.seek(OFFSET_SPACING, 0)
    dx, dy = struct.unpack('<2d', filehandle.read(16))
    filehandle.seek(HEADER_SIZE, 0)

    data = read_array(filehandle, dtype=DTYPE_HEIGHT, count=nx * ny)
    #nonmeasured_points_mask = (data == 0)
    data = data / (2 ** 16 - 2) * (zmax - zmin) + zmax
    #data[nonmeasured_points_mask] = np.nan
    data = data.reshape(ny, nx)

    step_x = dx * 1e-3
    step_y = dy * 1e-3
    image_layers = {}
    if read_image_layers:
        image_layers['Grayscale'] = read_array(filehandle, dtype=DTYPE_IMG, count=nx * ny).reshape(ny, ny)

    metadata = dict(date=date)

    return RawSurface(data, step_x, step_y, metadata=metadata, image_layers=image_layers)