import dateutil
import numpy as np
from .common import RawSurface, Reserved, Entry, Layout

NON_MEASURED_VALUE = 1000001

DATE_SIZE = 128
COMMENT_SIZE = 256

LAYOUT_CALIBRATION = Layout(
    Entry('yres', 'I'),
    Entry('xres', 'I'),
    Entry('N_tall', 'I'),
    Entry('dy_multip', 'f'),
    Entry('mppx', 'f'),
    Entry('mppy', 'f'),
    Entry('x_0', 'f'),
    Entry('y_0', 'f'),
    Entry('mpp_tall', 'f'),
    Entry('z0', 'f')
)

LAYOUT_MEASURE_CONFIG = Layout(
   Entry('type', 'I'),
   Entry('algorithm', 'I'),
   Entry('method', 'I'),
   Entry('objective', 'I'),
   Entry('area', 'I'),
   Entry('xres_area', 'I'),
   Entry('yres_area', 'I'),
   Entry('xres', 'I'),
   Entry('yres', 'I'),
   Entry('na', 'I'),
   Entry('incr_z', 'd'),
   Entry('range', 'f'),
   Entry('n_planes', 'I'),
   Entry('tpc_umbral_F', 'I'),
   Entry('restore', 'b'),
   Entry('num_layers', 'b'),
   Entry('version', 'b'),
   Entry('config_hardware', 'b'),
   Entry('stack_in_num', 'b'),
   Reserved(3),
   Entry('factorio_delmacio', 'I')
)

def read_plu(filepath, read_image_layers=False, encoding='utf-8'):
    with open(filepath, 'rb') as filehandle:
        date_block = filehandle.read(DATE_SIZE)
        timestamp = dateutil.parser.parse(date_block.decode().rstrip('\x00'))
        filehandle.seek(COMMENT_SIZE + 4, 1)
        calibration = LAYOUT_CALIBRATION.read(filehandle, encoding=encoding)
        measure_config = LAYOUT_MEASURE_CONFIG.read(filehandle, encoding=encoding)
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

