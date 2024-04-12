import zipfile
import io
import numpy as np
from .common import read_binary_layout, get_unit_conversion

HEADER_SIZE = 12

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

def read_vk4(filepath, encoding='utf-8'):
    with open(filepath, 'rb') as filehandle:
        filehandle.seek(HEADER_SIZE, 1)
        offset_table = read_binary_layout(filehandle, LAYOUT_OFFSET_TABLE, encoding=encoding)
        filehandle.seek(offset_table['meas_conds'], 0)
        measurement_conditions = read_binary_layout(filehandle, LAYOUT_MEASUREMENT_CONDITIONS, encoding=encoding)
        filehandle.seek(offset_table['height'], 0)
        height_data = read_binary_layout(filehandle, LAYOUT_HEIGHT_DATA, encoding=encoding)
        data_length = height_data['width'] * height_data['height']
        data = np.fromfile(filehandle, dtype=np.uint32, count=data_length) / 10_000  # to um

    data = data.reshape(height_data['height'], height_data['width'])

    step_x = measurement_conditions['x_length_per_pixel'] * get_unit_conversion('pm', 'um')
    step_y = measurement_conditions['y_length_per_pixel'] * get_unit_conversion('pm', 'um')
    return (data, step_x, step_y)

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
    return (data, step_x, step_y)
