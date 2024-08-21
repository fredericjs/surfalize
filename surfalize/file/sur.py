import struct
import zlib
from datetime import datetime
from enum import IntEnum
from dataclasses import dataclass

import numpy as np

from .common import get_unit_conversion, RawSurface, Entry, Reserved, Layout
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


LAYOUT_HEADER = Layout(
    Entry('code', '12s'),  # DIGITIAL SURF / DSCOMPRESSED
    Entry('format', 'h'),  # 0 for PC format
    Entry('n_objects', 'h'),
    Entry('version_number', 'h'),
    Entry('studiable_type', 'h'),
    Entry('name_object', '30s'),
    Entry('name_operator', '30s'),
    Entry('p_size', 'h'),
    Entry('acquisition_type', 'h'),
    Entry('range_type', 'h'),
    Entry('non_measured_points', 'h'),
    Entry('absolute_z_axis', 'h'),
    Entry('gauge_resolution', 'f'),
    Reserved(4),
    Entry('bits_per_point', 'h'),
    Entry('min_point', 'i'),
    Entry('max_point', 'i'),
    Entry('n_points_per_line', 'i'),
    Entry('n_lines', 'i'),
    Entry('n_total_points', 'i'),
    Entry('spacing_x', 'f'),
    Entry('spacing_y', 'f'),
    Entry('spacing_z', 'f'),
    Entry('name_x', '16s'),
    Entry('name_y', '16s'),
    Entry('name_z', '16s'),
    Entry('unit_step_x', '16s'),
    Entry('unit_step_y', '16s'),
    Entry('unit_step_z', '16s'),
    Entry('unit_x', '16s'),
    Entry('unit_y', '16s'),
    Entry('unit_z', '16s'),
    Entry('unit_ratio_x', 'f'),
    Entry('unit_ratio_y', 'f'),
    Entry('unit_ratio_z', 'f'),
    Entry('replica', 'h'),
    Entry('inverted', 'h'),
    Entry('leveled', 'h'),
    Reserved(12),
    Entry('seconds', 'h'),
    Entry('minutes', 'h'),
    Entry('hours', 'h'),
    Entry('day', 'h'),
    Entry('month', 'h'),
    Entry('year', 'h'),
    Entry('week_day', 'h'),
    Entry('measurement_duration', 'f'),
    Entry('compressed_data_size', 'I'),
    Reserved(6),
    Entry('length_comment', 'h'),
    Entry('length_private', 'h'),
    Entry('client_zone', '128s'),
    Entry('offset_x', 'f'),
    Entry('offset_y', 'f'),
    Entry('offset_z', 'f'),
    Entry('spacing_t', 'f'),
    Entry('offset_t', 'f'),
    Entry('name_t', '13s'),
    Entry('unit_step_t', '13s')
)

DTYPE_MAP = {16: 'int16', 32: 'int32'}


def read_sur_header(filehandle, encoding='utf-8'):
    fp_start = filehandle.tell()
    header = LAYOUT_HEADER.read(filehandle, encoding=encoding)

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


def read_uncompressed_data(filehandle, dtype, num_points):
    return np.fromfile(filehandle, count=num_points, dtype=dtype)


def read_compressed_data(filehandle, dtype, expected_compressed_size):
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
    total_compressed_size = 4
    for _ in range(dir_count):
        directory = read_directory(filehandle)
        total_compressed_size += 8 + directory.len_zipped_data
        directories.append(directory)
    if total_compressed_size != expected_compressed_size:
        raise CorruptedFileError(
            f'Compressed data size {total_compressed_size} does not match expected size of {expected_compressed_size}.'
        )
    # For each directory, we read data equivalent to the compressed size attribute in the directory and descompress it
    # using zlib. Then we concatenate the uncompressed data streams and read them with numpy
    # each datastream into a single
    decompressed_data = b''
    for directory in directories:
        compressed_data_stream = filehandle.read(directory.len_zipped_data)
        decompressed_data_stream = zlib.decompress(compressed_data_stream)
        if len(decompressed_data_stream) != directory.len_raw_data:
            raise CorruptedFileError(
                f'Decrompressed data size {len(decompressed_data_stream)} does not match expected size \
                of {directory.len_raw_data}.'
            )
        decompressed_data += decompressed_data_stream
    data = np.frombuffer(decompressed_data, dtype)
    return data


def is_gwyddion_export(sur_obj):
    """
    Checks whether .sur file was exported from Gwyddion. Unfortunately, Gwyddion currently seems to indicate the
    studiable type as PROFILE for exports containing surfaces. If however, the object name and operator name parameters
    are defined by Gwyddion as SCRATCH and csm, we can assume with reasonable certainty, that it should be infact a
    surface, not a profile.

    Parameters
    ----------
    sur_obj: SurObject

    Returns
    -------
    True if the sur object meets the characteristics of a Gwyddion exported surface
    """
    return (sur_obj.header['studiable_type'] == StudiableType.PROFILE and sur_obj.header['name_object'] == 'SCRATCH'
            and sur_obj.header['name_operator'] == 'csm')


def read_sur_object(filehandle):
    """
    Reads a sur object from a file. The function assumes that the filepointer points to the beginning of a sur object.
    A sur object consists of a 512-byte long header, followed by a variable length comment zone, private zone and
    data section. The data section is either compressed or uncompressed, which is determined by the file magic (first
    few bytes of the header).

    Parameters
    ----------
    filehandle
        Handle to the file object.
    Returns
    -------
    SurObject
    """
    header = read_sur_header(filehandle)
    dtype = DTYPE_MAP[header['bits_per_point']]
    ny = header['n_lines']
    nx = header['n_points_per_line']

    # Since 2010 version, there are two formats: compressed and uncompressed.
    # Which of the versions is used for a sur object is indicated by the file magic
    if header['code'] == MAGIC_CLASSIC:
        data = read_uncompressed_data(filehandle, dtype, header['n_total_points']).reshape(ny, nx)
    elif header['code'] == MAGIC_COMPRESSED:
        data = read_compressed_data(filehandle, dtype, header['compressed_data_size']).reshape(ny, nx)
    else:
        raise CorruptedFileError(f'Unknown file magic found: {header["code"]}.')

    return SurObject(header, data)


def get_surface(sur_obj):
    if sur_obj.header['non_measured_points'] == 1:
        invalidValue = sur_obj.header['min_point'] - 2
        nan_mask = (sur_obj.data == invalidValue)

    # The conversion from int to float needs to happen before multiply by the unit conversion factor!
    # Otherwise, we might overflow the values in the array and end up with white noise
    data = sur_obj.data * sur_obj.header['spacing_z']
    data = data * get_unit_conversion(sur_obj.header['unit_step_z'], 'um')
    step_x = get_unit_conversion(sur_obj.header['unit_step_x'], 'um') * sur_obj.header['spacing_x']
    step_y = get_unit_conversion(sur_obj.header['unit_step_y'], 'um') * sur_obj.header['spacing_y']

    if sur_obj.header['non_measured_points'] == 1:
        data[nan_mask] = np.nan

    data += sur_obj.header['offset_z']

    # This can be implemented in the future when metadata support is needed
    # timestamp = datetime.datetime(year=header['year'], month=header['month'], day=header['day'])
    return (data, step_x, step_y)


def read_sur(filepath, read_image_layers=False, encoding='utf-8'):
    filesize = filepath.stat().st_size
    with (open(filepath, 'rb') as filehandle):
        top_level_sur_obj = read_sur_object(filehandle)
        if top_level_sur_obj.header['n_objects'] > 1:
            raise UnsupportedFileFormatError(f'Multilayer or series studiables are currently not supported.')
        image_layers = {}
        if top_level_sur_obj.header['studiable_type'] == StudiableType.SURFACE or is_gwyddion_export(top_level_sur_obj):
            data, step_x, step_y = get_surface(top_level_sur_obj)
        elif top_level_sur_obj.header['studiable_type'] == StudiableType.RGB_INTENSITY_SURFACE:
            # after the surface, the r,g,b channels and the intensity image follow.
            data, step_x, step_y = get_surface(top_level_sur_obj)

            if read_image_layers:
                # read rgb layers
                rgb_layers = []
                for i in range(3):
                    rgb_layers.append(read_sur_object(filehandle).data)
                # image is grayscale
                if np.all(rgb_layers[0] == rgb_layers[1]) and np.all(rgb_layers[0] == rgb_layers[2]):
                    image_layers['Grayscale'] = rgb_layers[0]
                # image is rgb
                else:
                    image_layers['RGB'] = np.stack(rgb_layers, axis=-1)

                # read intensity layer
                image_layers['Intensity'] = read_sur_object(filehandle).data
        else:
            raise UnsupportedFileFormatError(
                f'Studiables of type {top_level_sur_obj.header["studiable_type"].name} are not supported.'
            )
        return RawSurface(data, step_x, step_y, image_layers=image_layers, metadata=top_level_sur_obj.header)


def write_sur(filepath, surface, encoding='utf-8', compressed=False):
    INT32_MAX = int(2 ** 32 / 2) - 1
    INT32_MIN = -int(2 ** 32 / 2)

    nm_points = int(surface.has_missing_points)

    INT_DATA_MIN = INT32_MIN + 2
    INT_DATA_MAX = INT32_MAX - 1
    data_max = np.nanmax(surface.data)
    data_min = np.nanmin(surface.data)
    data = ((surface.data - data_min) / (data_max - data_min)) * (INT_DATA_MAX - INT_DATA_MIN) + INT_DATA_MIN
    if nm_points:
        data[np.isnan(data)] = INT_DATA_MIN - 2
    data = data.astype('int32')
    spacing_z = (data_max - data_min) / (INT_DATA_MAX - INT_DATA_MIN)
    offset_z = offset = data_min + (data_max - data_min) / 2
    timestamp = datetime.now()

    header = {
        'code': MAGIC_CLASSIC if not compressed else MAGIC_COMPRESSED,
        'format': 0,  # PC Format
        'n_objects': 1,
        'version_number': 1,
        'studiable_type': StudiableType.SURFACE,
        'name_object': '',
        'name_operator': '',
        'p_size': 0,
        'acquisition_type': 0,
        'range_type': 0,
        'non_measured_points': nm_points,
        'absolute_z_axis': 0,
        'gauge_resolution': 0,
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
        'compressed_data_size': 0,
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


    with open(filepath, 'wb') as file:
        LAYOUT_HEADER.write(file, header)
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

