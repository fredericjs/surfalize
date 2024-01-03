import numpy as np
from .common import read_binary_layout

LAYOUT_HEADER = (
    (None, 85, None),
    ('res_x', 'I', False),
    ('res_y', 'I', False),
    (None, 4, None),
    ('step_x', 'f', False),
    ('step_y', 'f', False),
    ('step_z', 'f', False),
    (None, 8, None),
    ('comment_size', 'I', False),
    (None, 84, None)
)

def read_zmg(filepath):
    with open(filepath, 'rb') as filehandle:
        header = read_binary_layout(filehandle, LAYOUT_HEADER)
        filehandle.seek(header['comment_size'], 1)
        data_length = header['res_x'] * header['res_y']
        data = np.fromfile(filehandle, dtype=np.int16, count=data_length) * header['step_z']
    data = data.reshape((header['res_y'], header['res_x']))

    step_x = header['step_x']
    step_y = header['step_y']

    return (data, step_x, step_y)