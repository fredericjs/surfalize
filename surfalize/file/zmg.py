import numpy as np
from .common import RawSurface, Layout, Entry, Reserved

LAYOUT_HEADER = Layout(
    Reserved(85),
    Entry('res_x', 'I'),
    Entry('res_y', 'I'),
    Reserved(4),
    Entry('step_x', 'f'),
    Entry('step_y', 'f'),
    Entry('step_z', 'f'),
    Reserved(8),
    Entry('comment_size', 'I'),
    Reserved(84)
)

def read_zmg(filepath, read_image_layers=False, encoding='utf-8'):
    with open(filepath, 'rb') as filehandle:
        header = LAYOUT_HEADER.read(filehandle, encoding=encoding)
        filehandle.seek(header['comment_size'], 1)
        data_length = header['res_x'] * header['res_y']
        data = np.fromfile(filehandle, dtype=np.int16, count=data_length) * header['step_z']
    data = data.reshape((header['res_y'], header['res_x']))

    step_x = header['step_x']
    step_y = header['step_y']

    return RawSurface(data, step_x, step_y)