import dateutil
import numpy as np
from .common import read_binary_layout, RawSurface

NON_MEASURED_VALUE = 1000001

DATE_SIZE = 128
COMMENT_SIZE = 256

LAYOUT_CALIBRATION = (
    ('yres', 'I'),
    ('xres', 'I'),
    ('N_tall', 'I'),
    ('dy_multip', 'f'),
    ('mppx', 'f'),
    ('mppy', 'f'),
    ('x_0', 'f'),
    ('y_0', 'f'),
    ('mpp_tall', 'f'),
    ('z0', 'f')
)

LAYOUT_MEASURE_CONFIG = (
   ('type', 'I'),
   ('algorithm', 'I'),
   ('method', 'I'),
   ('objective', 'I'),
   ('area', 'I'),
   ('xres_area', 'I'),
   ('yres_area', 'I'),
   ('xres', 'I'),
   ('yres', 'I'),
   ('na', 'I'),
   ('incr_z', 'd'),
   ('range', 'f'),
   ('n_planes', 'I'),
   ('tpc_umbral_F', 'I'),
   ('restore', 'b'),
   ('num_layers', 'b'),
   ('version', 'b'),
   ('config_hardware', 'b'),
   ('stack_in_num', 'b'),
   (None, 3),
   ('factorio_delmacio', 'I')
)

def read_plu(filepath, read_image_layers=False, encoding='utf-8'):
    with open(filepath, 'rb') as filehandle:
        date_block = filehandle.read(DATE_SIZE)
        timestamp = dateutil.parser.parse(date_block.decode().rstrip('\x00'))
        filehandle.seek(COMMENT_SIZE + 4, 1)
        calibration = read_binary_layout(filehandle, LAYOUT_CALIBRATION, encoding=encoding)
        measure_config = read_binary_layout(filehandle, LAYOUT_MEASURE_CONFIG, encoding=encoding)
        data_length = calibration['xres'] * calibration['yres']
        data = np.fromfile(filehandle, dtype=np.float32, count=data_length)
        image_layers = {}
        if read_image_layers:
            filehandle.seek(16, 1) # skip 16 bytes, no idea what they are doing
            img = np.fromfile(filehandle, dtype=np.uint8, count=data_length * 3)
            img = img.reshape(calibration['yres'], calibration['xres'], 3)
            if np.all((img[:, :, 0] == img[:, :, 1]) & (img[:, :, 0] == img[:, :, 2])):
                image_layers['Grayscale'] = img[:, :, 0]
            else:
                image_layers['RGB'] = img
    data = data.reshape((calibration['yres'], calibration['xres']))
    data[data == NON_MEASURED_VALUE] = np.nan

    step_x = calibration['mppx']
    step_y = calibration['mppy']

    metadata = {'timestamp': timestamp}

    return RawSurface(data, step_x, step_y, image_layers=image_layers, metadata=metadata)

