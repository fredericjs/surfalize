import numpy as np
from .common import read_binary_layout, RawSurface

LAYOUT_HEADER = (
    (None, 85),
    ('res_x', 'I'),
    ('res_y', 'I'),
    (None, 4),
    ('step_x', 'f'),
    ('step_y', 'f'),
    ('step_z', 'f'),
    (None, 8),
    ('comment_size', 'I'),
    (None, 84)
)

def read_zmg(filepath, read_image_layers=False, encoding='utf-8'):
    with open(filepath, 'rb') as filehandle:
        header = read_binary_layout(filehandle, LAYOUT_HEADER, encoding=encoding)
        filehandle.seek(header['comment_size'], 1)
        data_length = header['res_x'] * header['res_y']
        data = np.fromfile(filehandle, dtype=np.int16, count=data_length) * header['step_z']
    data = data.reshape((header['res_y'], header['res_x']))

    step_x = header['step_x']
    step_y = header['step_y']

    return RawSurface(data, step_x, step_y)