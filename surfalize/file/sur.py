import struct
from .common import read_binary_layout, get_unit_conversion
from ..exceptions import CorruptedFileError
import numpy as np

# This is not fully implemented! Won't work with all SUR files.

MAGIC = 'DIGITAL SURF'
HEADER_SIZE = 512

LAYOUT_HEADER = (
    ('code', '12s', False),
    ('format', 'h', False),
    ('n_objects', 'h', False),
    ('version_number', 'h', False),
    ('studiable_type', 'h', True),
    ('name_object', '30s', True),
    ('name_operator', '30s', True),
    (None, 6, None), # Reserved
    ('non_measured_points', 'h', False),
    ('absolute_z_axis', 'h', False),
    (None, 8, None), # Reserved
    ('bits_per_point', 'h', False),
    ('min_point', 'i', False),
    ('max_point', 'i', False),
    ('n_points_per_line', 'i', False),
    ('n_lines', 'i', False),
    ('n_total_points', 'i', False),
    ('spacing_x', 'f', False),
    ('spacing_y', 'f', False),
    ('spacing_z', 'f', False),
    ('name_x', '16s', True),
    ('name_y', '16s', True),
    ('name_z', '16s', True),
    ('unit_step_x', '16s', False),
    ('unit_step_y', '16s', False),
    ('unit_step_z', '16s', False),
    ('unit_x', '16s', False),
    ('unit_y', '16s', False),
    ('unit_z', '16s', False),
    ('unit_ratio_x', 'f', False),
    ('unit_ratio_y', 'f', False),
    ('unit_ratio_z', 'f', False),
    ('replica', 'h', False),
    ('inverted', 'h', False),
    ('leveled', 'h', False),
    (None, 12, None), # Reserved
    ('seconds', 'h', True),
    ('minutes', 'h', True),
    ('hours', 'h', True),
    ('day', 'h', True),
    ('month', 'h', True),
    ('year', 'h', True),
    ('week_day', 'h', True),
    ('measurement_duration', 'f', True),
    (None, 10, None), # Reserved
    ('length_comment', 'h', False),
    ('length_private', 'h', False),
    ('client_zone', '128s', True),
    ('offset_x', 'f', True),
    ('offset_y', 'f', True),
    ('offset_z', 'f', True),
    ('spacing_t', 'f', True),
    ('offset_T', 'f', True),
    ('name_t', '13s', True),
    ('unit_step_t', '13s', True)
)

POINTSIZE = {16: 'h', 32: 'i'}

def read_sur(filepath):
    filesize = filepath.stat().st_size
    with open(filepath, 'rb') as filehandle:
        header = read_binary_layout(filehandle, LAYOUT_HEADER)

        if header['code'] != MAGIC or header['version_number'] != 1:
            raise CorruptedFileError

        if header['unit_ratio_x'] != 1 or header['unit_ratio_y'] != 1 or header['unit_ratio_z'] != 1:
            raise NotImplementedError("This file type cannot be correctly read currently.")

        filehandle.seek(header['length_comment'], 1)
        filehandle.seek(header['length_private'], 1)
        dtype = POINTSIZE[header['bits_per_point']]
        dsize = struct.calcsize(dtype)
        data_size = header['n_total_points'] * dsize

        data_size = header['n_total_points'] * dsize
        total_header_size = HEADER_SIZE + header['length_comment'] + header['length_private']
        expected_data_size = header['n_total_points'] * dsize
        if filesize - total_header_size > expected_data_size:
            filehandle.seek(filesize - data_size, 0)
        shape = (header['n_lines'], header['n_points_per_line'])
        data = np.fromfile(filehandle, dtype=np.dtype(dtype)).reshape(shape)


        data = data * get_unit_conversion(header['unit_z'], 'um') * header['spacing_z']
        step_x = get_unit_conversion(header['unit_x'], 'um') * header['spacing_x']
        step_y = get_unit_conversion(header['unit_y'], 'um') * header['spacing_y']

        return (data, step_x, step_y)