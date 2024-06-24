import zipfile
import struct
from datetime import datetime

import numpy as np
from .common import read_binary_layout, get_unit_conversion, RawSurface, np_fromany
from ..exceptions import CorruptedFileError

HEADER_SIZE = 12
FIXED_UNIT = 'pm'

DTYPE_MAP = {16: 'uint16', 32: 'uint32'}

LAYOUT_OFFSET_TABLE = (
    ('meas_conds', 'I'),
    ('color_peak', 'I'),
    ('color_light', 'I'),
    ('light', 'I'),
    (None, 8),
    ('height', 'I'),
    (None, 8),
    ('clr_peak_thumb', 'I'),
    ('clr_thumb', 'I'),
    ('light_thumb', 'I'),
    ('height_thumb', 'I'),
    ('assembly_info', 'I'),
    ('line_measure', 'I'),
    ('line_thickness', 'I'),
    ('string_data', 'I'),
)

LAYOUT_MEASUREMENT_CONDITIONS = (
    ('size', 'I'),
    ('year', 'I'),
    ('month', 'I'),
    ('day', 'I'),
    ('hour', 'I'),
    ('minute', 'I'),
    ('second', 'I'),
    ('diff_from_UTC', 'i'),
    ('img_attributes', 'I'),
    ('user_interface_mode', 'I'),
    ('color_composite_mode', 'I'),
    ('img_layer_number', 'I'),
    ('run_mode', 'I'),
    ('peak_mode', 'I'),
    ('sharpening_level', 'I'),
    ('speed', 'I'),
    ('distance', 'I'),
    ('pitch', 'I'),
    ('optical_zoom', 'I'),
    ('number_of_lines', 'I'),
    ('line0_position', 'I'),
    (None, 12),
    ('lens_magnification', 'I'),
    ('PMT_gain_mode', 'I'),
    ('PMT_gain', 'I'),
    ('PMT_offset', 'I'),
    ('ND_filter', 'I'),
    (None, 4),
    ('persist_count', 'I'),
    ('shutter_speed_mode', 'I'),
    ('shutter_speed', 'I'),
    ('white_balance_mode', 'I'),
    ('white_balance_red', 'I'),
    ('white_balance_blue', 'I'),
    ('camera_gain', 'I'),
    ('plane_compensation', 'I'),
    ('xy_length_unit', 'I'),
    ('z_length_unit', 'I'),
    ('xy_decimal_place', 'I'),
    ('z_decimal_place', 'I'),
    ('x_length_per_pixel', 'I'),
    ('y_length_per_pixel', 'I'),
    ('z_length_per_digit', 'I'),
    (None, 20),
    ('light_filter_type', 'I'),
    (None, 4),
    ('gamma_reverse', 'I'),
    ('gamma', 'I'),
    ('gamma_correction_offset', 'I'),
    ('CCD_BW_offset', 'I'),
    ('num_aperture', 'I'),
    ('head_type', 'I'),
    ('PMT_gain_2', 'I'),
    ('omit_color_img', 'I'),
    ('lens_ID', 'I'),
    ('light_lut_mode', 'I'),
    ('light_lut_in0', 'I'),
    ('light_lut_out0', 'I'),
    ('light_lut_in1', 'I'),
    ('light_lut_out1', 'I'),
    ('light_lut_in2', 'I'),
    ('light_lut_out2', 'I'),
    ('light_lut_in3', 'I'),
    ('light_lut_out3', 'I'),
    ('light_lut_in4', 'I'),
    ('light_lut_out4', 'I'),
    ('upper_position', 'I'),
    ('lower_position', 'I'),
    ('light_effective_bit_depth', 'I'),
    ('height_effective_bit_depth', 'I')
)

LAYOUT_HEIGHT_DATA = (
    ('width', 'I'),
    ('height', 'I'),
    ('bit_depth', 'I'),
    ('compression', 'I'),
    ('data_byte_size', 'I'),
    ('palette_range_min', 'I'),
    ('palette_range_max', 'I'),
    (None, 768)
)

LAYOUT_IMAGE_DATA = (
    ('width', 'I'),
    ('height', 'I'),
    ('bit_depth', 'I'),
    ('compression', 'I'),
    ('data_byte_size', 'I')
)

def read_rgb_layer(filehandle, offset):
    filehandle.seek(offset, 0)
    channel_table = read_binary_layout(filehandle, LAYOUT_IMAGE_DATA)
    channel_length = channel_table['width'] * channel_table['height'] * 3
    if channel_table['data_byte_size'] != channel_length * channel_table['bit_depth'] / (8 * 3):
        raise CorruptedFileError(f'Size of channel () does not correspond to expected size.')
    channel_data = np_fromany(filehandle, dtype=np.uint8, count=channel_length)
    # It seems like vk4 encodes the color channels in the order GRB, therefore we flip the last axis to convert to RGB format
    channel_data = np.flip(channel_data.reshape(channel_table['height'], channel_table['width'], 3), axis=2)
    return channel_data

def read_height_layer(filehandle, offset):
    filehandle.seek(offset, 0)
    channel_table = read_binary_layout(filehandle, LAYOUT_HEIGHT_DATA)
    channel_length = channel_table['width'] * channel_table['height']
    if channel_table['data_byte_size'] != channel_length * channel_table['bit_depth'] / 8:
        raise CorruptedFileError('Size of channel does not correspond to expected size.')
    dtype = DTYPE_MAP[channel_table['bit_depth']]
    channel_data = np_fromany(filehandle, dtype=dtype, count=channel_length)
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

def extract_vk4(filehandle, read_image_layers=False, encoding='utf-8'):
    metadata = dict()
    header = filehandle.read(HEADER_SIZE)
    offset_table = read_binary_layout(filehandle, LAYOUT_OFFSET_TABLE)
    filehandle.seek(offset_table['meas_conds'], 0)
    measurement_conditions = read_binary_layout(filehandle, LAYOUT_MEASUREMENT_CONDITIONS)

    if read_image_layers:
        image_layers = {}
        #if measurement_conditions['omit_color_img'] > 0:
        image_layers['RGB'] = read_rgb_layer(filehandle, offset_table['color_peak'])
        image_layers['Laser+RGB'] = read_rgb_layer(filehandle, offset_table['color_light'])

        image_layers['Laser'] = read_height_layer(filehandle, offset_table['light'])
    else:
        image_layers = None
    height_layer = read_height_layer(filehandle, offset_table['height'])
    metadata.update(read_string_data(filehandle, offset_table['string_data']))

    scale_factor = get_unit_conversion(FIXED_UNIT, 'um')
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

def read_vk4(filepath, read_image_layers=False, encoding='utf-8'):
    with open(filepath, 'rb') as filehandle:
        return extract_vk4(filehandle, read_image_layers=read_image_layers, encoding=encoding)

def read_vk6_vk7(filepath, read_image_layers=False, encoding='utf-8'):
    with zipfile.ZipFile(filepath) as archive:
        with archive.open('Vk4File') as filehandle:
            return extract_vk4(filehandle, read_image_layers=read_image_layers, encoding=encoding)
