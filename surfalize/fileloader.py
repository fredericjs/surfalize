import struct
import zipfile
import io
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np


def read_uint32_le(file):
    return struct.unpack('<I', file.read(4))[0]

def read_float_le(file):
    return struct.unpack('<f', file.read(4))[0]

def read_double_le(file):
    return struct.unpack('<d', file.read(8))[0]

def read_bool(file):
    return struct.unpack('b', file.read(1))[0]

def load_plu(filepath):
    NON_MEASURED_VALUE = 1000001

    DATE_SIZE = 128
    COMMENT_SIZE = 256
    HEADER_SIZE = DATE_SIZE + COMMENT_SIZE + 4

    calibration = dict()
    measure_config = dict()
    with open(filepath, 'rb') as file:
        file.read(HEADER_SIZE)
        calibration['yres'] = read_uint32_le(file)
        calibration['xres'] = read_uint32_le(file)
        calibration['N_tall'] = read_uint32_le(file)
        calibration['dy_multip'] = read_float_le(file)
        calibration['mppx'] = read_float_le(file)
        calibration['mppy'] = read_float_le(file)
        calibration['x_0'] = read_float_le(file)
        calibration['y_0'] = read_float_le(file)
        calibration['mpp_tall'] = read_float_le(file)
        calibration['z0'] = read_float_le(file)

        measure_config['type']  = read_uint32_le(file)
        measure_config['algorithm'] = read_uint32_le(file)
        measure_config['method'] = read_uint32_le(file)
        measure_config['objective'] = read_uint32_le(file)
        measure_config['area'] = read_uint32_le(file)
        measure_config['xres_area'] = read_uint32_le(file)
        measure_config['yres_area'] = read_uint32_le(file)
        measure_config['xres'] = read_uint32_le(file)
        measure_config['yres'] = read_uint32_le(file)
        measure_config['na'] = read_uint32_le(file)
        measure_config['incr_z'] = read_double_le(file)
        measure_config['range'] = read_float_le(file)
        measure_config['n_planes'] = read_uint32_le(file)
        measure_config['tpc_umbral_F'] = read_uint32_le(file)
        measure_config['restore'] = read_bool(file)
        measure_config['num_layers'] = read_bool(file)
        measure_config['version'] = read_bool(file)
        measure_config['config_hardware'] = read_bool(file)
        measure_config['stack_in_num'] = read_bool(file)
        measure_config['reserved'] = read_bool(file)
        file.read(2)
        measure_config['factorio_delmacio'] = read_uint32_le(file)

        data_length = calibration['xres'] * calibration['yres']
        data = np.zeros(data_length)
        for i in range(data_length):
            data[i] = read_float_le(file)
        data = data.reshape((calibration['yres'], calibration['xres']))
       
        data[data == NON_MEASURED_VALUE] = np.nan

        step_x = calibration['mppx']
        step_y = calibration['mppy']
        
        return (data, step_x, step_y)
    

def load_plux(filepath): 
    with zipfile.ZipFile(filepath) as archive:
        data = archive.read('LAYER_0.raw')
        metadata = archive.read('index.xml')

    xml_str = metadata.decode('utf-8')

    # Parse the XML string
    root = ET.fromstring(xml_str)
    shape_x = int(root.find('GENERAL/IMAGE_SIZE_X').text)
    shape_y = int(root.find('GENERAL/IMAGE_SIZE_Y').text)
    step_x = float(root.find('GENERAL/FOV_X').text)
    step_y = float(root.find('GENERAL/FOV_Y').text)
    size = shape_x * shape_y
    height_data = np.array(struct.unpack(f'<{size}f', data)).reshape((shape_y, shape_x))

    return (height_data, step_x, step_y)
    
def _vk4_extract_offsets(in_file):
    """extract_offsets

    Extract offset values from the offset table of a vk4 file. Stores offsets
    and returns values in dictionary

    :param in_file: open file obj, must be vk4 file
    """

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

    return offsets

def _vk4_extract_measurement_conditions(offset_dict, in_file):
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

    return measurement_conditions

def _vk4_extract_img_data(offset_dict, d_type, in_file):
    """extract_img_data

    Extracts image data, either height or light intensity, from the vk4 file.
    All metadata and raw image data pertaining to the particular aforementioned
    type is extracted and returned as a dictionary.

    :param offset_dict: dictionary - offset values in vk4
    :param d_type: string - type of data, must be 'height' or 'light'
    :param in_file: open file obj, must be vk4 file
    """

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

    return data  

def load_vk6_vk7(filepath):
    with zipfile.ZipFile(filepath) as archive:
        vk4file = io.BytesIO(archive.read('Vk4File'))
        
    offsets = _vk4_extract_offsets(vk4file)
    img_data = _vk4_extract_img_data(offsets, 'height', vk4file)
    measurement_conditions = _vk4_extract_measurement_conditions(offsets, vk4file)

    step_x = measurement_conditions['x_length_per_pixel'] / 1000000
    step_y = measurement_conditions['y_length_per_pixel'] / 1000000
    data = img_data['data'].reshape(img_data['height'], img_data['width']) / 10000
    
    return (data, step_x, step_y)
    

def load_vk4(filepath):
    with open(filepath, 'rb') as file:
        offsets = _vk4_extract_offsets(file)
        img_data = _vk4_extract_img_data(offsets, 'height', file)
        measurement_conditions = _vk4_extract_measurement_conditions(offsets, file)

    step_x = measurement_conditions['x_length_per_pixel'] / 1000000
    step_y = measurement_conditions['y_length_per_pixel'] / 1000000
    data = img_data['data'].reshape(img_data['height'], img_data['width']) / 10000
    
    return (data, step_x, step_y)

dispatch = {
    '.vk4': load_vk4,
    '.vk6': load_vk6_vk7,
    '.vk7': load_vk6_vk7,
    '.plu': load_plu,
    '.plux': load_plux,
}


class UnsupportedFileFormatError(Exception):
    pass


def load_file(filepath):
    filepath = Path(filepath)
    if filepath.suffix not in dispatch:
        raise UnsupportedFileFormatError
    return dispatch[filepath.suffix](filepath)