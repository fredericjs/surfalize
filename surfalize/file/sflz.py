import json
import zlib
import lzma
import datetime

import numpy as np

from surfalize.file.common import (FormatFromPrevious, RawSurface, Apply, Entry, Layout, FileHandler,
                                    get_unit_conversion)
from ..exceptions import CorruptedFileError

MAGIC = b'SFLZ'
VERSION = '1.0'
HEIGHT_LAYER_NAME = 'topography'
# Internal unit used by surfalize for lateral spacing and height values
DEFAULT_UNIT = 'um'

COMPRESSION_TYPE_FROM_INT = {
    0: 'none',
    1: 'zlib',
    2: 'lzma'
}

INT_FROM_COMPRESSION_TYPE = {v: k for k, v in COMPRESSION_TYPE_FROM_INT.items()}


class ConvertCompression(Apply):

    def read(self, data):
        return COMPRESSION_TYPE_FROM_INT[data]

    def write(self, data):
        return INT_FROM_COMPRESSION_TYPE[data]


# The 10-byte version string is read/written separately from the rest of the header.
LAYOUT_HEADER = Layout(
    Entry('step_x', 'd'),
    Entry('step_y', 'd'),
    Entry('unit', '8s'),
    Entry('num_layers', 'H'),
    Entry('num_metadata', 'H'),
    Entry('compression_algorithm', ConvertCompression('B')),
    Entry('metadata_size', 'I'),
)

LAYOUT_LAYER_HEADER = Layout(
    Entry('name_length', 'H'),
    Entry('name', FormatFromPrevious('name_length', 's')),
    Entry('is_height', '?'),
    Entry('width', 'I'),
    Entry('height', 'I'),
    Entry('channels', 'B'),
    Entry('datatype', '4s'),
    Entry('scaled', '?'),
    Entry('has_nan', '?'),
    Entry('min_value', 'd'),
    Entry('max_value', 'd'),
    Entry('size', 'I'),
)


def compress(data, compression_type):
    if compression_type == 'none':
        return data
    elif compression_type == 'zlib':
        return zlib.compress(data)
    elif compression_type == 'lzma':
        return lzma.compress(data)
    raise ValueError(f'Unknown compression type: {compression_type}')


def decompress(data, compression_type):
    if compression_type == 'none':
        return data
    elif compression_type == 'zlib':
        return zlib.decompress(data)
    elif compression_type == 'lzma':
        return lzma.decompress(data)
    raise ValueError(f'Unknown compression type: {compression_type}')


def _json_default(obj):
    """Serializer fallback for metadata values that JSON does not handle natively."""
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return {'__datetime__': obj.isoformat()}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _json_object_hook(dct):
    """Restore datetime values that were encoded by `_json_default`."""
    if '__datetime__' in dct:
        try:
            return datetime.datetime.fromisoformat(dct['__datetime__'])
        except ValueError:
            return dct['__datetime__']
    return dct


def serialize_metadata(metadata):
    if not metadata:
        return b''
    return json.dumps(metadata, default=_json_default, ensure_ascii=False).encode('utf-8')


def deserialize_metadata(blob):
    if not blob:
        return {}
    return json.loads(blob.decode('utf-8'), object_hook=_json_object_hook)


def quantize(data, dtype):
    """
    Quantize floating point data to an integer dtype, scaling the value range [min, max] linearly onto
    the integer range. If the data contains NaNs, the largest integer value is reserved as a sentinel
    so that non-measured points survive the round trip.

    Returns
    -------
    tuple[np.ndarray, float, float, bool]
        The quantized array, the minimum and maximum value used for scaling, and whether a NaN
        sentinel was used.
    """
    info = np.iinfo(dtype)
    dtype_min, dtype_max = int(info.min), int(info.max)
    nan_mask = np.isnan(data)
    has_nan = bool(nan_mask.any())
    if nan_mask.all():
        min_value = max_value = 0.0
    else:
        min_value = float(np.nanmin(data))
        max_value = float(np.nanmax(data))
    # When a sentinel is needed, the top integer value is reserved for NaN
    top = dtype_max - 1 if has_nan else dtype_max
    levels = top - dtype_min
    safe = np.where(nan_mask, min_value, data)
    if max_value > min_value and levels > 0:
        normalized = (safe - min_value) / (max_value - min_value)
        quantized = np.clip(np.rint(normalized * levels + dtype_min), dtype_min, top)
    else:
        quantized = np.full(data.shape, dtype_min, dtype='float64')
    quantized = quantized.astype(dtype)
    if has_nan:
        quantized[nan_mask] = dtype_max
    return quantized, min_value, max_value, has_nan


def dequantize(quantized, dtype, min_value, max_value, has_nan):
    """Reverse `quantize`, restoring the floating point values (and NaNs)."""
    info = np.iinfo(dtype)
    dtype_min, dtype_max = int(info.min), int(info.max)
    top = dtype_max - 1 if has_nan else dtype_max
    levels = top - dtype_min
    if levels > 0 and max_value > min_value:
        data = (quantized.astype('float64') - dtype_min) / levels * (max_value - min_value) + min_value
    else:
        data = np.full(quantized.shape, min_value, dtype='float64')
    if has_nan:
        data[quantized == dtype_max] = np.nan
    return data


def _reshape_layer(data, layer_header):
    if layer_header['channels'] > 1:
        return data.reshape(layer_header['height'], layer_header['width'], layer_header['channels'])
    return data.reshape(layer_header['height'], layer_header['width'])


@FileHandler.register_reader(suffix='.sflz', magic=MAGIC)
def read_sflz(filehandle, encoding='utf-8', read_image_layers=True):
    if filehandle.read(len(MAGIC)) != MAGIC:
        raise CorruptedFileError('Incompatible file magic detected.')
    filehandle.read(10)  # version string, currently informational only

    header = LAYOUT_HEADER.read(filehandle)
    compression_algorithm = header['compression_algorithm']
    metadata = deserialize_metadata(filehandle.read(header['metadata_size']))
    # The lateral spacing and height values are stored in the file's unit; convert them to surfalize's
    # internal unit (micrometers). Image layers carry no physical length unit and are left untouched.
    to_um = get_unit_conversion(header['unit'] or DEFAULT_UNIT, DEFAULT_UNIT)
    image_layers = {}
    height_data = None
    for _ in range(header['num_layers']):
        layer_header = LAYOUT_LAYER_HEADER.read(filehandle)
        if not layer_header['is_height'] and not read_image_layers:
            # Skip the compressed payload of image layers we are not interested in
            filehandle.seek(layer_header['size'], 1)
            continue
        data = decompress(filehandle.read(layer_header['size']), compression_algorithm)
        dtype = np.dtype(layer_header['datatype'])
        data = _reshape_layer(np.frombuffer(data, dtype=dtype), layer_header)
        if layer_header['scaled']:
            data = dequantize(data, dtype, layer_header['min_value'], layer_header['max_value'],
                              layer_header['has_nan'])
        else:
            data = data.copy()
        if layer_header['is_height']:
            height_data = data * to_um if to_um != 1 else data
        else:
            image_layers[layer_header['name']] = data
    return RawSurface(height_data, header['step_x'] * to_um, header['step_y'] * to_um, metadata=metadata,
                      image_layers=image_layers)


def _write_layer(filehandle, name, data, compression, is_height, dtype=None):
    """Write a single layer. The height layer may be quantized to `dtype`; image layers are stored as-is."""
    scaled = False
    has_nan = False
    min_value = max_value = 0.0
    if is_height and dtype is not None and dtype.kind in ('i', 'u'):
        quantized, min_value, max_value, has_nan = quantize(data, dtype)
        raw = quantized.tobytes()
        scaled = True
        out_dtype = dtype
    else:
        # Lossless: store the array in its own (or the requested float) dtype, preserving NaNs natively
        out_dtype = dtype if (is_height and dtype is not None) else data.dtype
        raw = np.ascontiguousarray(data, dtype=out_dtype).tobytes()
    compressed = compress(raw, compression)
    layer_header = {
        'name_length': len(name),
        'name': name,
        'is_height': is_height,
        'width': data.shape[1],
        'height': data.shape[0],
        'channels': 1 if data.ndim == 2 else data.shape[-1],
        'datatype': np.dtype(out_dtype).str,
        'scaled': scaled,
        'has_nan': has_nan,
        'min_value': min_value,
        'max_value': max_value,
        'size': len(compressed),
    }
    LAYOUT_LAYER_HEADER.write(filehandle, layer_header)
    filehandle.write(compressed)


@FileHandler.register_writer(suffix='.sflz')
def write_sflz(filehandle, surface, encoding='utf-8', compression='zlib', save_image_layers=True,
               write_metadata=True, dtype='<u4', unit=DEFAULT_UNIT):
    dtype = np.dtype(dtype)
    metadata_blob = serialize_metadata(surface.metadata) if write_metadata else b''
    # surfalize works internally in micrometers; convert the lateral spacing and height values to the
    # requested storage unit (validated here, raising for unknown units).
    from_um = get_unit_conversion(DEFAULT_UNIT, unit)

    filehandle.write(MAGIC)
    filehandle.write(VERSION.ljust(10).encode('ascii'))

    header = {
        'step_x': surface.step_x * from_um,
        'step_y': surface.step_y * from_um,
        'unit': unit,
        'num_layers': 1 + (len(surface.image_layers) if save_image_layers else 0),
        'num_metadata': len(surface.metadata) if write_metadata else 0,
        'compression_algorithm': compression,
        'metadata_size': len(metadata_blob),
    }
    LAYOUT_HEADER.write(filehandle, header)
    filehandle.write(metadata_blob)

    height = surface.data * from_um if from_um != 1 else surface.data
    _write_layer(filehandle, HEIGHT_LAYER_NAME, height, compression, is_height=True, dtype=dtype)
    if save_image_layers and surface.image_layers:
        for name, layer in surface.image_layers.items():
            _write_layer(filehandle, name, layer.data, compression, is_height=False)
