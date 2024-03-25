import struct
from datetime import datetime
from .common import read_binary_layout, write_binary_layout, get_unit_conversion
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
    ('offset_t', 'f', True),
    ('name_t', '13s', True),
    ('unit_step_t', '13s', True)
)

POINTSIZE = {16: 'h', 32: 'i'}

def write_sur(filepath, surface, encoding='utf-8'):
    INT32_MAX = int(2 ** 32 / 2) - 1
    INT32_MIN = -int(2 ** 32 / 2)

    data = surface.data
    nm_points = int(surface._nonmeasured_points_exist)

    INT_DATA_MIN = INT32_MIN + 2
    INT_DATA_MAX = INT32_MAX - 1
    data_max = surface.data.max()
    data_min = surface.data.min()
    data = ((surface.data - data_min) / (data_max - data_min)) * (INT_DATA_MAX - INT_DATA_MIN) + INT_DATA_MIN
    if nm_points:
        data[np.isnan(data)] = INT_DATA_MIN - 2
    data = data.astype('int32')
    spacing_z = (data_max - data_min) / (INT_DATA_MAX - INT_DATA_MIN)
    offset_z = offset = data_min + (data_max - data_min)/2
    timestamp = datetime.now()

    header = {
        'code': MAGIC,
        'format': 0,  # PC Format
        'n_objects': 1,
        'version_number': 1,
        'studiable_type': 2,
        'name_object': '',
        'name_operator': '',
        'non_measured_points': nm_points,
        'absolute_z_axis': 0,
        'bits_per_point': 32,
        'min_point': INT_DATA_MIN,
        'max_point': INT_DATA_MAX,
        'n_points_per_line': surface.size.x,
        'n_lines': surface.size.y,
        'n_total_points': surface.size.x * surface.size.y,
        'spacing_x': surface.step_x,
        'spacing_y': surface.step_y,
        'spacing_z': spacing_z,
        'name_x': 'X',
        'name_y': 'Y',
        'name_z': 'Z',
        'unit_step_x': 'µm',
        'unit_step_y': 'µm',
        'unit_step_z': 'µm',
        'unit_x': 'µm',
        'unit_y': 'µm',
        'unit_z': 'µm',
        'unit_ratio_x': 1.0,
        'unit_ratio_y': 1.0,
        'unit_ratio_z': 1.0,
        'replica': 1,
        'inverted': 0,
        'leveled': 0,
        'seconds': timestamp.second,
        'minutes': timestamp.minute,
        'hours': timestamp.hour,
        'day': timestamp.day,
        'month': timestamp.month,
        'year': timestamp.year,
        'week_day': timestamp.weekday(),
        'measurement_duration': 0,
        'length_comment': 0,
        'length_private': 0,
        'client_zone': 'Exported by surfalize',
        'offset_x': 0,
        'offset_y': 0,
        'offset_z': offset_z,
        'spacing_t': 0,
        'offset_t': 0,
        'name_t': '',
        'unit_step_t': ''
    }

    # Pad all strings with spaces
    for name, format_, _ in LAYOUT_HEADER:
        if name is None or not format_.endswith('s'):
            continue
        length = struct.calcsize(format_)
        header[name] = header[name].ljust(length)

    with open(filepath, 'wb') as file:
        write_binary_layout(file, LAYOUT_HEADER, header)
        data.tofile(file)

def read_sur(filepath, encoding='utf-8'):
    filesize = filepath.stat().st_size
    with open(filepath, 'rb') as filehandle:
        header = read_binary_layout(filehandle, LAYOUT_HEADER, encoding=encoding, fast=False)

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

        if header['non_measured_points'] == 1:
            invalidValue = header['min_point'] - 2
            nan_mask = data[data == invalidValue]

        data = data * get_unit_conversion(header['unit_z'], 'um') * header['spacing_z']
        step_x = get_unit_conversion(header['unit_x'], 'um') * header['spacing_x']
        step_y = get_unit_conversion(header['unit_y'], 'um') * header['spacing_y']

        data += header['offset_z']

        if header['non_measured_points'] == 1:
            data[nan_mask] = np.nan

        return (data, step_x, step_y)