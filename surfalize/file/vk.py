import zipfile
import struct
from datetime import datetime

import numpy as np
from .common import read_binary_layout, get_unit_conversion, RawSurface
from ..exceptions import CorruptedFileError

HEADER_SIZE = 12

DTYPE_MAP = {16: 'uint16', 32: 'uint32'}

LAYOUT_OFFSET_TABLE = (
    ('meas_conds', 'I', False),
    ('color_peak', 'I', True),
    ('color_light', 'I', True),
    ('light', 'I', True),
    (None, 8, None),
    ('height', 'I', False),
    (None, 8, None),
    ('clr_peak_thumb', 'I', True),
    ('clr_thumb', 'I', True),
    ('light_thumb', 'I', True),
    ('height_thumb', 'I', True),
    ('assembly_info', 'I', True),
    ('line_measure', 'I', True),
    ('line_thickness', 'I', True),
    ('string_data', 'I', True),
)

LAYOUT_MEASUREMENT_CONDITIONS = (
    ('size', 'I', True),
    ('year', 'I', True),
    ('month', 'I', True),
    ('day', 'I', True),
    ('hour', 'I', True),
    ('minute', 'I', True),
    ('second', 'I', True),
    ('diff_from_UTC', 'i', True),
    ('img_attributes', 'I', True),
    ('user_interface_mode', 'I', True),
    ('color_composite_mode', 'I', True),
    ('img_layer_number', 'I', True),
    ('run_mode', 'I', True),
    ('peak_mode', 'I', True),
    ('sharpening_level', 'I', True),
    ('speed', 'I', True),
    ('distance', 'I', True),
    ('pitch', 'I', True),
    ('optical_zoom', 'I', True),
    ('number_of_lines', 'I', True),
    ('line0_position', 'I', True),
    (None, 12, None),
    ('lens_magnification', 'I', True),
    ('PMT_gain_mode', 'I', True),
    ('PMT_gain', 'I', True),
    ('PMT_offset', 'I', True),
    ('ND_filter', 'I', True),
    (None, 4, None),
    ('persist_count', 'I', True),
    ('shutter_speed_mode', 'I', True),
    ('shutter_speed', 'I', True),
    ('white_balance_mode', 'I', True),
    ('white_balance_red', 'I', True),
    ('white_balance_blue', 'I', True),
    ('camera_gain', 'I', True),
    ('plane_compensation', 'I', True),
    ('xy_length_unit', 'I', False),
    ('z_length_unit', 'I', False),
    ('xy_decimal_place', 'I', False),
    ('z_decimal_place', 'I', False),
    ('x_length_per_pixel', 'I', False),
    ('y_length_per_pixel', 'I', False),
    ('z_length_per_digit', 'I', False),
    (None, 20, None),
    ('light_filter_type', 'I', True),
    (None, 4, None),
    ('gamma_reverse', 'I', True),
    ('gamma', 'I', True),
    ('gamma_correction_offset', 'I', True),
    ('CCD_BW_offset', 'I', True),
    ('num_aperture', 'I', True),
    ('head_type', 'I', True),
    ('PMT_gain_2', 'I', True),
    ('omit_color_img', 'I', True),
    ('lens_ID', 'I', True),
    ('light_lut_mode', 'I', True),
    ('light_lut_in0', 'I', True),
    ('light_lut_out0', 'I', True),
    ('light_lut_in1', 'I', True),
    ('light_lut_out1', 'I', True),
    ('light_lut_in2', 'I', True),
    ('light_lut_out2', 'I', True),
    ('light_lut_in3', 'I', True),
    ('light_lut_out3', 'I', True),
    ('light_lut_in4', 'I', True),
    ('light_lut_out4', 'I', True),
    ('upper_position', 'I', True),
    ('lower_position', 'I', True),
    ('light_effective_bit_depth', 'I', True),
    ('height_effective_bit_depth', 'I', True)
)

LAYOUT_HEIGHT_DATA = (
    ('width', 'I', False),
    ('height', 'I', False),
    ('bit_depth', 'I', False),
    ('compression', 'I', True),
    ('data_byte_size', 'I', True),
    ('palette_range_min', 'I', True),
    ('palette_range_max', 'I', True),
    (None, 768, None)
)

LAYOUT_IMAGE_DATA = (
    ('width', 'I', False),
    ('height', 'I', False),
    ('bit_depth', 'I', False),
    ('compression', 'I', True),
    ('data_byte_size', 'I', True)
)

def read_rgb_layer(filehandle, offset):
    filehandle.seek(offset, 0)
    channel_table = read_binary_layout(filehandle, LAYOUT_IMAGE_DATA, fast=False)
    channel_length = channel_table['width'] * channel_table['height'] * 3
    if channel_table['data_byte_size'] != channel_length * channel_table['bit_depth'] / (8 * 3):
        raise CorruptedFileError(f'Size of channel () does not correspond to expected size.')
    channel_data = np.fromfile(filehandle, dtype=np.uint8, count=channel_length)
    # It seems like vk4 encodes the color channels in the order GRB, therefore we flip the last axis to convert to RGB format
    channel_data = np.flip(channel_data.reshape(channel_table['height'], channel_table['width'], 3), axis=2)
    return channel_data

def read_height_layer(filehandle, offset):
    filehandle.seek(offset, 0)
    channel_table = read_binary_layout(filehandle, LAYOUT_HEIGHT_DATA, fast=False)
    channel_length = channel_table['width'] * channel_table['height']
    if channel_table['data_byte_size'] != channel_length * channel_table['bit_depth'] / 8:
        raise CorruptedFileError('Size of channel does not correspond to expected size.')
    dtype = DTYPE_MAP[channel_table['bit_depth']]
    channel_data = np.fromfile(filehandle, dtype=dtype, count=channel_length)
    channel_data = channel_data.reshape(channel_table['height'], channel_table['width'])
    return channel_data

def read_string_data(filehandle, offset):
    filehandle.seek(offset, 0)
    str_data = dict()
    size_title = struct.unpack('I', filehandle.read(4))[0] * 2
    # Every character is followed by a null byte. Therefore, we read twice the since of the expected string length and slice to
    # remove every second character
    str_data['title'] = filehandle.read(size_title).decode()[::2]
    size_lens_name = struct.unpack('I', filehandle.read(4))[0] * 2
    # Same here
    str_data['lens_name'] = filehandle.read(size_lens_name).decode()[::2]
    return str_data

def read_vk4(filepath, read_image_layers=False, encoding='utf-8'):
    metadata = dict()
    with open(filepath, 'rb') as filehandle:
        header = filehandle.read(HEADER_SIZE)
        offset_table = read_binary_layout(filehandle, LAYOUT_OFFSET_TABLE, fast=False)
        filehandle.seek(offset_table['meas_conds'], 0)
        measurement_conditions = read_binary_layout(filehandle, LAYOUT_MEASUREMENT_CONDITIONS, fast=False)

        if read_image_layers:
            image_layers = {}
            if measurement_conditions['omit_color_img'] == 0:
                image_layers['RGB'] = read_rgb_layer(filehandle, offset_table['color_peak'])
                image_layers['Laser+RGB'] = read_rgb_layer(filehandle, offset_table['color_light'])

            image_layers['Laser'] = read_height_layer(filehandle, offset_table['light'])
        else:
            image_layers = None
        height_layer = read_height_layer(filehandle, offset_table['height'])
        metadata.update(read_string_data(filehandle, offset_table['string_data']))

    scale_factor = get_unit_conversion('pm', 'um')
    scale_factor_height = scale_factor * measurement_conditions['z_length_per_digit']
    height_layer = height_layer * scale_factor_height
    step_x = measurement_conditions['x_length_per_pixel'] * scale_factor
    step_y = measurement_conditions['y_length_per_pixel'] * scale_factor

    metadata['timestamp'] = datetime(year=measurement_conditions['year'], month=measurement_conditions['month'],
                                     day=measurement_conditions['day'], hour=measurement_conditions['hour'],
                                     minute=measurement_conditions['minute'], second=measurement_conditions['second'])
    metadata['optical_zoom'] = measurement_conditions['optical_zoom'] / 10
    metadata['objective_magnification'] = measurement_conditions['lens_magnification'] / 10

    return RawSurface(height_layer, step_x, step_y, metadata, image_layers)

def read_vk6_vk7(filepath, encoding='utf-8'):
    with zipfile.ZipFile(filepath) as archive:
        with archive.open('Vk4File') as filehandle:
            filehandle.seek(HEADER_SIZE, 1)
            offset_table = read_binary_layout(filehandle, LAYOUT_OFFSET_TABLE, encoding=encoding)
            filehandle.seek(offset_table['meas_conds'], 0)
            measurement_conditions = read_binary_layout(filehandle, LAYOUT_MEASUREMENT_CONDITIONS, encoding=encoding)
            filehandle.seek(offset_table['height'], 0)
            height_data = read_binary_layout(filehandle, LAYOUT_HEIGHT_DATA, encoding=encoding)
            data_length = height_data['width'] * height_data['height']
            data = np.frombuffer(filehandle.read(data_length * 4), dtype=np.uint32, count=data_length) / 10_000  # to um

    data = data.reshape(height_data['height'], height_data['width'])

    step_x = measurement_conditions['x_length_per_pixel'] * get_unit_conversion('pm', 'um')
    step_y = measurement_conditions['y_length_per_pixel'] * get_unit_conversion('pm', 'um')
    return RawSurface(data, step_x, step_y)
