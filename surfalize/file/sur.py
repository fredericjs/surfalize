import struct
import zlib
from datetime import datetime
from enum import IntEnum
from dataclasses import dataclass

import numpy as np

from .common import read_binary_layout, write_binary_layout, get_unit_conversion
from ..exceptions import CorruptedFileError, UnsupportedFileFormatError

# This is not fully implemented! Won't work with all SUR files.

MAGIC_CLASSIC = 'DIGITAL SURF'
MAGIC_COMPRESSED = 'DSCOMPRESSED'
HEADER_SIZE = 512

@dataclass
class Directory:
    """
    Dataclass that represents a directory block in a compressed surf file data section
    """
    len_raw_data: int
    len_zipped_data: int

@dataclass
class SurObject:
    """
    Dataclass that represents a sur object, which contains a full header and a data block.
    """
    header: dict
    data: np.ndarray

class StudiableType(IntEnum):
    PROFILE = 1
    SURFACE = 2
    BINARY_IMAGE = 3
    SERIES_OF_PROFILES = 4
    SERIES_OF_SURFACES = 5
    MERIDIAN_DISC = 6
    MULTILAYER_PROFILE = 7
    MULTILAYER_SURFACE = 8
    PARALLEL_DISC = 9
    INTENSITY_IMAGE = 10
    INTENSITY_SURFACE = 11
    RGB_IMAGE = 12
    RGB_SURFACE = 13
    FORCE_CURVE = 14
    SERIES_OF_FORCE_CURVES = 15
    RGB_INTENSITY_SURFACE = 16
    PARAMETERIC_PROFILE = 17
    SERIES_OF_RGB_IMAGES = 18
    SPECTRUM_STUDIABLE = 19

class AcquisitionType(IntEnum):
    UNKNOWN = 0
    CONTACT_STYLUS = 1
    SCANNING_OPTICAL_GAUGE = 2
    THERMOCOUPLE = 3
    UNKNOWN_2 = 4
    CONTACT_STYLUS_WITH_SKID = 5
    AFM = 6
    STM = 7
    VIDEO = 8
    INTERFEROMETER = 9
    STRUCTURED_LIGHT_PROJECTION = 10


LAYOUT_HEADER = (
    ('code', '12s', False), # DIGITIAL SURF / DSCOMPRESSED
    ('format', 'h', False), # 0 for PC format
    ('n_objects', 'h', False),
    ('version_number', 'h', False),
    ('studiable_type', 'h', True),
    ('name_object', '30s', True),
    ('name_operator', '30s', True),
    ('p_size', 'h', True),
    ('acquisition_type', 'h', True),
    ('range_type', 'h', True),
    ('non_measured_points', 'h', False),
    ('absolute_z_axis', 'h', False),
    ('gauge_resolution', 'f', True),
    (None, 4, None), # Reserved
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
    ('compressed_data_size', 'I', False),
    (None, 6, None), # Reserved
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

DTYPE_MAP = {16: 'int16', 32: 'in32'}

def read_sur_header(filehandle, encoding='utf-8'):
    fp_start = filehandle.tell()
    header = read_binary_layout(filehandle, LAYOUT_HEADER, encoding=encoding, fast=False)

    if header['code'] not in (MAGIC_CLASSIC, MAGIC_COMPRESSED) or header['version_number'] != 1:
        raise CorruptedFileError('Unknown header format')

    if header['unit_ratio_x'] != 1 or header['unit_ratio_y'] != 1 or header['unit_ratio_z'] != 1:
        raise NotImplementedError("This file type cannot be correctly read currently.")

    header['studiable_type'] = StudiableType(header['studiable_type'])
    header['acquisition_type'] = AcquisitionType(header['acquisition_type'])

    if filehandle.tell() - fp_start != HEADER_SIZE:
        raise CorruptedFileError("Unknown header size.")

    header['comment'] = filehandle.read(header['length_comment'])
    header['private'] = filehandle.read(header['length_private'])

    if header['studiable_type'] not in [StudiableType.SURFACE, StudiableType.RGB_INTENSITY_SURFACE]:
        raise UnsupportedFileFormatError(f'Studiables of type {header['studiable_type'].name} are not supported.')
    if header['n_objects'] > 1:
        raise UnsupportedFileFormatError(f'Multilayer or series studiables are not supported.')
    return header

def read_directory(filehandle):
    """
    Reads a directory block from a compressed surf file's data section. This function expects the file pointer to
    point to the beginning of a directory block.

    Parameters
    ----------
    filehandle
        Handle to the file object.

    Notes
    -----
    The layout of a directory is as follows:

    raw data length - uint32
    zipped data length - uint32

    Returns
    -------
    Directory
    """
    return Directory(*struct.unpack('<2I', filehandle.read(8)))

def read_compressed_data(filehandle, dtype):
    """
    Reads a datablock from a compressed sur file. This function assumes that the filepointer points to the beginning
    of a datablock.

    Parameters
    ----------
    filehandle
        Handle to the file object.
    dtype
        Datatype of the binary data.

    Notes
    -----
    The datablock of a compressed sur file is organized as follows:

    directory count - uint32
    directory item 0
        raw data length - uint32
        zipped data length - uint32
    ...
    directory item n
        ...
    zipped data stream 0
    ...
    zipped data stream 1

    The compressed data is organized into an arbitrary amount of binary streams that must be concatenated before
    decompression. The data block of the file begins a number of directory blocks. The following bytes encode the
    directory blocks, which holds the raw and zipped data length of the streams. After the nth directory block, the
    raw streams are encoded. Currently, according to the file format specification, compressed files use only one
    datastream. However, in the future, this could change.

    Returns
    -------
    data: np.ndarray
        Decompressed 1d array of the data-
    """
    # The compressed datablock begins with a uint32 that encodes the number of directories
    dir_count = struct.unpack('I', filehandle.read(4))[0]
    # Afterwards, that number of directories is stored consecutively, containing 2 uint32 encoding the length of the
    # raw data and the length of the zipped data in the stream associated with that directory
    directories = []
    for _ in range(dir_count):
        directories.append(read_directory(filehandle))
    # For each directory, we read data equivalent to the compressed size attribute in the directory and descompress it
    # using zlib. Then we concatenate the uncompressed data streams and read them with numpy
    # each datastream into a single
    decompressed_data = b''
    for directory in directories:
        compressed_data_stream = filehandle.read(directory.len_zipped_data)
        decompressed_data_stream = zlib.decompress(compressed_data_stream)
        if len(decompressed_data_stream) != directory.len_raw_data:
            raise ValueError
        decompressed_data += decompressed_data_stream
    data = np.frombuffer(decompressed_data, dtype)
    return data


def read_sur_object(filehandle):
    header = read_sur_header(filehandle)
    dtype = DTYPE_MAP[header['bits_per_point']]
    n_points = header['n_total_points']
    ny = header['n_lines']
    nx = header['n_points_per_line']

    # Since 2010 version, there are two formats: compressed and uncompressed.
    # Which of the versions is used for a sur object is indicated by the file magic
    if header['code'] == MAGIC_CLASSIC:
        data = np.fromfile(filehandle, count=n_points, dtype=dtype).reshape(ny, nx)
    elif header['code'] == MAGIC_COMPRESSED:
        data = read_compressed_data(filehandle, dtype).reshape(ny, nx)
    else:
        raise CorruptedFileError('Unknown header format')

    if header['non_measured_points'] == 1:
        invalidValue = header['min_point'] - 2
        nan_mask = (data == invalidValue)

    data = data * get_unit_conversion(header['unit_step_z'], 'um') * header['spacing_z']
    step_x = get_unit_conversion(header['unit_step_x'], 'um') * header['spacing_x']
    step_y = get_unit_conversion(header['unit_step_y'], 'um') * header['spacing_y']

    if header['non_measured_points'] == 1:
        data[nan_mask] = np.nan

    data += header['offset_z']

    # This can be implemented in the future when metadata support is needed
    #timestamp = datetime.datetime(year=header['year'], month=header['month'], day=header['day'])

    return SurObject(header, data)

def read_sur(filepath, encoding='utf-8'):
    filesize = filepath.stat().st_size
    with open(filepath, 'rb') as filehandle:
        object_count = 0
        while True:
            sur_obj = read_sur_object(filehandle)
            if sur_obj.header['studiable_type'] == StudiableType.SURFACE:
                if filehandle.tell() != filesize:
                    raise CorruptedFileError
                break

        return (data, step_x, step_y)




def write_sur(filepath, surface, encoding='utf-8', compressed=False):
    INT32_MAX = int(2 ** 32 / 2) - 1
    INT32_MIN = -int(2 ** 32 / 2)

    data = surface.data
    nm_points = int(surface._nonmeasured_points_exist)

    INT_DATA_MIN = INT32_MIN + 2
    INT_DATA_MAX = INT32_MAX - 1
    data_max = np.nanmax(surface.data)
    data_min = np.nanmin(surface.data)
    data = ((surface.data - data_min) / (data_max - data_min)) * (INT_DATA_MAX - INT_DATA_MIN) + INT_DATA_MIN
    if nm_points:
        data[np.isnan(data)] = INT_DATA_MIN - 2
    data = data.astype('int32')
    spacing_z = (data_max - data_min) / (INT_DATA_MAX - INT_DATA_MIN)
    offset_z = offset = data_min + (data_max - data_min)/2
    timestamp = datetime.now()

    header = {
        'code': MAGIC_CLASSIC if not compressed else MAGIC_COMPRESSED,
        'format': 0,  # PC Format
        'n_objects': 1,
        'version_number': 1,
        'studiable_type': StudiableType.SURFACE,
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
        if not compressed:
            data.tofile(file)
            return
        else:
            uncompressed_data = data.tobytes()
            compressed_data = zlib.compress(uncompressed_data)
            # Write directory count = 1 and the length of a single data stream containing all the compressed data
            file.write(struct.pack('<3I', 1, len(uncompressed_data), len(compressed_data)))
            file.write(compressed_data)
            return