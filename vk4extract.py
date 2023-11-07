"""vk4extract

This module extracts meta and image data from Keyence Profilometry vk4 data
files. The various functions are aimed at extracting specific layers of data
from a vk4 file, which are stored in dictionaries.

Author
------
Wylie Gunn
Behzad Torkian

Created
-------
11 June 2018

Last Modified
-------------
19 July 2018

"""
import logging
import struct
import numpy as np
# import readbinary as rb

log = logging.getLogger('vk4_driver.vk4extract')

# extract offsets for data sections of vk4 file
def extract_offsets(in_file):
    """extract_offsets

    Extract offset values from the offset table of a vk4 file. Stores offsets
    and returns values in dictionary

    :param in_file: open file obj, must be vk4 file
    """
    log.debug("Entering extract_offsets()")

    offsets = dict()
    in_file.seek(12)
    offsets['meas_conds'] = struct.unpack('<I', in_file.read(4))[0]
    offsets['color_peak'] = struct.unpack('<I', in_file.read(4))[0]
    offsets['color_light'] = struct.unpack('<I', in_file.read(4))[0]
    offsets['light'] = struct.unpack('<I', in_file.read(4))[0]
    in_file.seek(8, 1)
    offsets['height'] = struct.unpack('<I', in_file.read(4))[0]
    in_file.seek(8, 1)
    offsets['clr_peak_thumb'] = struct.unpack('<I', in_file.read(4))[0]
    offsets['clr_thumb'] = struct.unpack('<I', in_file.read(4))[0]
    offsets['light_thumb'] = struct.unpack('<I', in_file.read(4))[0]
    offsets['height_thumb'] = struct.unpack('<I', in_file.read(4))[0]
    offsets['assembly_info'] = struct.unpack('<I', in_file.read(4))[0]
    offsets['line_measure'] = struct.unpack('<I', in_file.read(4))[0]
    offsets['line_thickness'] = struct.unpack('<I', in_file.read(4))[0]
    offsets['string_data'] = struct.unpack('<I', in_file.read(4))[0]
    # not sure if reserved is necessary
    offsets['reserved'] = struct.unpack('<I', in_file.read(4))[0]

    log.debug("Exiting extract_offsets()")
    return offsets


# extracts metadata and measurement conditions
def extract_measurement_conditions(offset_dict, in_file):
    """extract_measurement_conditions

    Extracts various measurement conditions and metadata from  a vk4 file.
    Stores and returns data as dictionary.

    :param offset_dict: dictionary - offset values in vk4
    :param in_file: open file obj, must be vk4 file

    Note
    ----
        It seems that there is a fair amount of metadata\measurement data
        unaccounted for in this function. This data is pulled from the first
        300 bytes of data in this section of the vk4 file leaving some 384 bytes
        of between the last of this extracted data and the beginning of the
        'color peak' section of the file, according to the vk4 files I have
        worked with.
    """
    log.debug("Entering extract_measurement_conditions()")

    measurement_conditions = dict()
    measurement_conditions['name'] = 'measurement_conditions'
    in_file.seek(offset_dict['meas_conds'])
    measurement_conditions['size'] = struct.unpack('<I', in_file.read(4))[0]
    # file date
    measurement_conditions['year'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['month'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['day'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['hour'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['minute'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['second'] = struct.unpack('<I', in_file.read(4))[0]

    measurement_conditions['diff_from_UTC'] = struct.unpack('<i', in_file.read(4))[0]
    measurement_conditions['img_attributes'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['user_interface_mode'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['color_composite_mode'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['img_layer_number'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['run_mode'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['peak_mode'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['sharpening_level'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['speed'] = struct.unpack('<I', in_file.read(4))[0]
    # distance and pitch are considered in nanometers
    measurement_conditions['distance'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['pitch'] = struct.unpack('<I', in_file.read(4))[0]
    # optical zoom interpreted as float => optical_zoom/10.0
    measurement_conditions['optical_zoom'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['number_of_lines'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['line0_position'] = struct.unpack('<I', in_file.read(4))[0]
    # next 12 bytes 'reserved'
    measurement_conditions['reserved_1'] = []
    for x in range(3):
        measurement_conditions['reserved_1'].append(struct.unpack('<I', in_file.read(4))[0])

    measurement_conditions['lens_magnification'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['PMT_gain_mode'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['PMT_gain'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['PMT_offset'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['ND_filter'] = struct.unpack('<I', in_file.read(4))[0]
    # next 4 bytes 'reserved'
    measurement_conditions['reserved_2'] = struct.unpack('<I', in_file.read(4))[0]
    # image average frequency
    measurement_conditions['persist_count'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['shutter_speed_mode'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['shutter_speed'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['white_balance_mode'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['white_balance_red'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['white_balance_blue'] = struct.unpack('<I', in_file.read(4))[0]
    # multiply camera_gain by 6 to get camera gain in dB
    measurement_conditions['camera_gain'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['plane_compensation'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['xy_length_unit'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['z_length_unit'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['xy_decimal_place'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['z_decimal_place'] = struct.unpack('<I', in_file.read(4))[0]
    # the following three values are considered in picometers
    measurement_conditions['x_length_per_pixel'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['y_length_per_pixel'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['z_length_per_digit'] = struct.unpack('<I', in_file.read(4))[0]
    # next 20 bytes reserved
    measurement_conditions['reserved_3'] = []
    for x in range(5):
        measurement_conditions['reserved_3'].append(struct.unpack('<I', in_file.read(4))[0])

    measurement_conditions['light_filter_type'] = struct.unpack('<I', in_file.read(4))[0]
    # next 4 bytes reserved
    measurement_conditions['reserved_4'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['gamma_reverse'] = struct.unpack('<I', in_file.read(4))[0]
    # gamma interpreted as float => gamma/100.0
    measurement_conditions['gamma'] = struct.unpack('<I', in_file.read(4))[0]
    # gamma offset interpreted as float => gamma_correction_offset/65536.0
    measurement_conditions['gamma_correction_offset'] = struct.unpack('<I', in_file.read(4))[0]
    # CCD BW offset interpreted as float => CCD_BW_offset/100.0
    measurement_conditions['CCD_BW_offset'] = struct.unpack('<I', in_file.read(4))[0]
    # numerical aperture interpreted as float => num_aperture/1000.0
    measurement_conditions['num_aperture'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['head_type'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['PMT_gain_2'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['omit_color_img'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['lens_ID'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['light_lut_mode'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['light_lut_in0'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['light_lut_out0'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['light_lut_in1'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['light_lut_out1'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['light_lut_in2'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['light_lut_out2'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['light_lut_in3'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['light_lut_out3'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['light_lut_in4'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['light_lut_out4'] = struct.unpack('<I', in_file.read(4))[0]
    # upper and lower position considered in nanometers
    measurement_conditions['upper_position'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['lower_position'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['light_effective_bit_depth'] = struct.unpack('<I', in_file.read(4))[0]
    measurement_conditions['height_effective_bit_depth'] = struct.unpack('<I', in_file.read(4))[0]

    log.debug("Exiting extract_measurement_conditions()")
    return measurement_conditions


# color peak and color + light data extracted with extract_color_data
def extract_color_data(offset_dict, color_type, in_file):
    """extract_color_data

    Extracts RGB metadata and raw image data from a vk4 file. Stores data and
    returns as dictionary

    :param offset_dict: dictionary - offset values in vk4
    :param color_type: string - type of data, must be 'peak' or 'light'
    :param in_file: open file obj, must be vk4 file
    """
    log.debug("Entering extract_color_data()")

    rgb_types = {'peak': 'color_peak', 'light': 'color_light'}
    rgb_color_data = dict()
    rgb_color_data['name'] = 'RGB ' + color_type
    in_file.seek(offset_dict[rgb_types[color_type]])

    rgb_color_data['width'] = struct.unpack('<I', in_file.read(4))[0]
    rgb_color_data['height'] = struct.unpack('<I', in_file.read(4))[0]
    rgb_color_data['bit_depth'] = struct.unpack('<I', in_file.read(4))[0]
    rgb_color_data['compression'] = struct.unpack('<I', in_file.read(4))[0]
    rgb_color_data['data_byte_size'] = struct.unpack('<I', in_file.read(4))[0]

    rgb_color_arr = np.zeros(((rgb_color_data['width'] * rgb_color_data['height']),
                              (rgb_color_data['bit_depth'] // 8)), dtype=np.uint8)

    i = 0
    for val in range(rgb_color_data['width'] * rgb_color_data['height']):
        rgb = []
        for channel in range(3):
            rgb.append(ord(in_file.read(1)))
        rgb_color_arr[i] = rgb
        i = i + 1

    rgb_color_data['data'] = rgb_color_arr

    log.debug("Exiting extract_color_data()")
    return rgb_color_data


# light and height data extracted with extract_img_data
def extract_img_data(offset_dict, d_type, in_file):
    """extract_img_data

    Extracts image data, either height or light intensity, from the vk4 file.
    All metadata and raw image data pertaining to the particular aforementioned
    type is extracted and returned as a dictionary.

    :param offset_dict: dictionary - offset values in vk4
    :param d_type: string - type of data, must be 'height' or 'light'
    :param in_file: open file obj, must be vk4 file
    """
    log.debug("Entering extract_img_data()")

    data_types = {'height': ('height', np.uint32, '<I', 4),
                  'light': ('light', np.uint16, '<H', 2)}
    data = dict()
    data['name'] = d_type.capitalize()
    in_file.seek(offset_dict[data_types[d_type][0]])
    data['width'] = struct.unpack('<I', in_file.read(4))[0]
    data['height'] = struct.unpack('<I', in_file.read(4))[0]
    data['bit_depth'] = struct.unpack('<I', in_file.read(4))[0]
    data['compression'] = struct.unpack('<I', in_file.read(4))[0]
    data['data_byte_size'] = struct.unpack('<I', in_file.read(4))[0]
    data['palette_range_min'] = struct.unpack('<I', in_file.read(4))[0]
    data['palette_range_max'] = struct.unpack('<I', in_file.read(4))[0]
    # The palette section of the hexdump is 768 bytes long has 256 3-byte
    # repeats, for now I will store them as a 1d array of uint8 values
    palette = np.zeros(768, dtype=np.uint8)

    i = 0
    for val in range(768):
        palette[i] = ord(in_file.read(1))
        i = i+1
    data['palette'] = palette

    array = np.zeros((data['width']*data['height']), dtype=data_types[d_type][1])
    int_type = data_types[d_type][2]
    bytesize = data_types[d_type][3]
    i = 0
    for val in range(data['width']*data['height']):
        # array[i] = data_types[d_type][2](in_file)
        array[i] = struct.unpack(int_type, in_file.read(bytesize))[0]
        i = i + 1
    data['data'] = array

    log.debug("Exiting extract_img_data()")
    return data


# extract string meta data
def extract_string_data(offset_dict, in_file):
    """extract_string_data

    Extracts a couple pieces of metadata from the end of a vk4 file, namely
    the title of the file and the lens type. Returns data as dictionary

    :param offset_dict: dictionary - offset values in vk4
    :param in_file: open file obj, must be vk4 file
    """
    log.debug("Entering extract_string_data()")

    string_data = dict()
    string_data['name'] = 'string_data'
    in_file.seek(offset_dict['string_data'])
    title_length = struct.unpack('<I', in_file.read(4))[0]

    # string_data['title'] = in_file.read(title_length)
    string_data['title'] = string_from_chars(in_file, title_length)
    log.debug(string_data['title'])
    lens_name_length = struct.unpack('<I', in_file.read(4))[0]

    # string_data['lens_name'] = in_file.read(lens_name_length)
    string_data['lens_name'] = string_from_chars(in_file, lens_name_length)

    log.debug("Exiting extract_string_data()")
    return string_data


def string_from_chars(in_file, length):
    """string_from_chars

    A helper function which returns a string for metadata extracted in
    extract_string_data, as the chars that make up those strings are separated
    by a single byte in the vk4 file

    :param in_file: open file obj, must be vk4 file
    :param length: length of data to read
    """
    ret_str = ''
    for byte in range(length):
        ret_str += str(in_file.read(1))[2]
        # each character to extract is separated by a junk byte, so it is
        # necessary to seek ahead, skipping over the junk byte
        in_file.seek(1, 1)
    return ret_str
