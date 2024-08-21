import zipfile
import struct
from datetime import datetime

import numpy as np
from .common import get_unit_conversion, RawSurface, np_fromany, Entry, Reserved, Layout
from ..exceptions import CorruptedFileError

HEADER_SIZE = 12
FIXED_UNIT = 'pm'

DTYPE_MAP = {16: 'uint16', 32: 'uint32'}

LAYOUT_OFFSET_TABLE = Layout(
    Entry('meas_conds', 'I'),
    Entry('color_peak', 'I'),
    Entry('color_light', 'I'),
    Entry('light', 'I'),
    Reserved(8),
    Entry('height', 'I'),
    Reserved(8),
    Entry('clr_peak_thumb', 'I'),
    Entry('clr_thumb', 'I'),
    Entry('light_thumb', 'I'),
    Entry('height_thumb', 'I'),
    Entry('assembly_info', 'I'),
    Entry('line_measure', 'I'),
    Entry('line_thickness', 'I'),
    Entry('string_data', 'I'),
)

LAYOUT_MEASUREMENT_CONDITIONS = Layout(
    Entry('size', 'I'),
    Entry('year', 'I'),
    Entry('month', 'I'),
    Entry('day', 'I'),
    Entry('hour', 'I'),
    Entry('minute', 'I'),
    Entry('second', 'I'),
    Entry('diff_from_UTC', 'i'),
    Entry('img_attributes', 'I'),
    Entry('user_interface_mode', 'I'),
    Entry('color_composite_mode', 'I'),
    Entry('img_layer_number', 'I'),
    Entry('run_mode', 'I'),
    Entry('peak_mode', 'I'),
    Entry('sharpening_level', 'I'),
    Entry('speed', 'I'),
    Entry('distance', 'I'),
    Entry('pitch', 'I'),
    Entry('optical_zoom', 'I'),
    Entry('number_of_lines', 'I'),
    Entry('line0_position', 'I'),
    Reserved(12),
    Entry('lens_magnification', 'I'),
    Entry('PMT_gain_mode', 'I'),
    Entry('PMT_gain', 'I'),
    Entry('PMT_offset', 'I'),
    Entry('ND_filter', 'I'),
    Reserved(4),
    Entry('persist_count', 'I'),
    Entry('shutter_speed_mode', 'I'),
    Entry('shutter_speed', 'I'),
    Entry('white_balance_mode', 'I'),
    Entry('white_balance_red', 'I'),
    Entry('white_balance_blue', 'I'),
    Entry('camera_gain', 'I'),
    Entry('plane_compensation', 'I'),
    Entry('xy_length_unit', 'I'),
    Entry('z_length_unit', 'I'),
    Entry('xy_decimal_place', 'I'),
    Entry('z_decimal_place', 'I'),
    Entry('x_length_per_pixel', 'I'),
    Entry('y_length_per_pixel', 'I'),
    Entry('z_length_per_digit', 'I'),
    Reserved(20),
    Entry('light_filter_type', 'I'),
    Reserved(4),
    Entry('gamma_reverse', 'I'),
    Entry('gamma', 'I'),
    Entry('gamma_correction_offset', 'I'),
    Entry('CCD_BW_offset', 'I'),
    Entry('num_aperture', 'I'),
    Entry('head_type', 'I'),
    Entry('PMT_gain_2', 'I'),
    Entry('omit_color_img', 'I'),
    Entry('lens_ID', 'I'),
    Entry('light_lut_mode', 'I'),
    Entry('light_lut_in0', 'I'),
    Entry('light_lut_out0', 'I'),
    Entry('light_lut_in1', 'I'),
    Entry('light_lut_out1', 'I'),
    Entry('light_lut_in2', 'I'),
    Entry('light_lut_out2', 'I'),
    Entry('light_lut_in3', 'I'),
    Entry('light_lut_out3', 'I'),
    Entry('light_lut_in4', 'I'),
    Entry('light_lut_out4', 'I'),
    Entry('upper_position', 'I'),
    Entry('lower_position', 'I'),
    Entry('light_effective_bit_depth', 'I'),
    Entry('height_effective_bit_depth', 'I')
)

LAYOUT_HEIGHT_DATA = Layout(
    Entry('width', 'I'),
    Entry('height', 'I'),
    Entry('bit_depth', 'I'),
    Entry('compression', 'I'),
    Entry('data_byte_size', 'I'),
    Entry('palette_range_min', 'I'),
    Entry('palette_range_max', 'I'),
    Reserved(768)
)

LAYOUT_IMAGE_DATA = Layout(
    Entry('width', 'I'),
    Entry('height', 'I'),
    Entry('bit_depth', 'I'),
    Entry('compression', 'I'),
    Entry('data_byte_size', 'I')
)

def read_rgb_layer(filehandle, offset):
    filehandle.seek(offset, 0)
    channel_table = LAYOUT_IMAGE_DATA.read(filehandle)
    channel_length = channel_table['width'] * channel_table['height'] * 3
    if channel_table['data_byte_size'] != channel_length * channel_table['bit_depth'] / (8 * 3):
        raise CorruptedFileError(f'Size of channel () does not correspond to expected size.')
    channel_data = np_fromany(filehandle, dtype=np.uint8, count=channel_length)
    # It seems like vk4 encodes the color channels in the order GRB, therefore we flip the last axis to convert to RGB format
    channel_data = np.flip(channel_data.reshape(channel_table['height'], channel_table['width'], 3), axis=2)
    return channel_data

def read_height_layer(filehandle, offset):
    filehandle.seek(offset, 0)
    channel_table = LAYOUT_HEIGHT_DATA.read(filehandle)
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
    offset_table = LAYOUT_OFFSET_TABLE.read(filehandle)
    filehandle.seek(offset_table['meas_conds'], 0)
    measurement_conditions = LAYOUT_MEASUREMENT_CONDITIONS.read(filehandle)

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
