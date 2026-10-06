# Information on the file format was reverse engineered from sample files recorded with a Heliotis H8 white light
# interferometer (Diaphus / heliService 1.12, file version "hdat_3.4.x") and the TIFF layers exported from them.
#
# A Heliotis .hdat file is a LabVIEW data structure that was serialized with LabVIEW's "Flatten To String" primitive,
# so all values are stored big-endian. The file starts with the format version as a LabVIEW string (u32 length +
# characters, e.g. "hdat_3.4.x") and a reserved u32, followed by a flattened LabVIEW variant:
#
#   u32         LabVIEW version that flattened the data (e.g. 0x15008000 for LabVIEW 2021)
#   u32         number of type descriptors
#   ...         type descriptors, each starting with u16 length, u8 flags and u8 type code. If bit 0x40 of the flags
#               is set, the descriptor ends with its label as a Pascal string (padded to even length). Composite
#               types (clusters, arrays) reference other descriptors by index.
#   u16, u16    number of top-level types (1) and the index of the top-level type descriptor
#   ...         the data, flattened according to the top-level type descriptor
#   u32         number of variant attributes
#
# The top-level type is the type definition "hdat_v3.4.x.ctl", a cluster called "core.measurement" which holds general
# information (time stamp, applied evaluation algorithms, ...), the optics calibration (in particular the lateral
# resolution in um/pixel in "optics3D") and a 2d array "matrix" of measurement cells. Each cell contains another
# variant that holds the actual measurement. For the raw data type "raw_C4universal", this is a cluster of
#
#   FloatData   3d array (layer, y, x) of float32, layers described by the JSON string FloatMeta
#   IntData     3d array (layer, y, x) of uint16, layers described by the JSON string IntMeta
#   FloatMeta   JSON list with one entry per layer of FloatData, holding its "Type" and "FixPointScaling"
#   CameraMeta  JSON object with acquisition parameters of the camera (z-tags, number of frames, ...)
#   IntMeta     JSON list with one entry per layer of IntData, holding its "Type" and "FixPointScaling"
#
# The height data is the FloatData layer of type "Surface". It is given in the distance unit of the camera
# (Scan3dDistanceUnit, micrometers in all known files) and contains the absolute z-position of the scanner, which is
# why the values typically carry a large offset. The other layers (e.g. Confidence, Phase and Amplitude) are imported
# as image layers. The processing steps listed in "heliAlgo" (e.g. outlier removal and leveling) are not applied to
# the stored data but only by the vendor software upon export, so they are only imported as metadata.

import datetime
import json
import struct
import warnings

import numpy as np

from .common import RawSurface, FileHandler, get_unit_conversion
from ..exceptions import CorruptedFileError, UnsupportedFileFormatError

MAGIC = b'hdat_'

# LabVIEW epoch for time stamps
LABVIEW_EPOCH = datetime.datetime(1904, 1, 1, tzinfo=datetime.timezone.utc)

# LabVIEW type codes
TYPE_VOID = 0x00
TYPE_BOOLEAN = 0x21
TYPE_STRING = 0x30
TYPE_ARRAY = 0x40
TYPE_CLUSTER = 0x50
TYPE_VARIANT = 0x53
TYPE_MEASURE_DATA = 0x54
TYPE_TYPEDEF = 0xF1

# Numeric type codes mapped to big-endian numpy dtypes. Enums (0x15 - 0x17) are flattened like unsigned integers.
NUMERIC_DTYPES = {
    0x01: '>i1', 0x02: '>i2', 0x03: '>i4', 0x04: '>i8',
    0x05: '>u1', 0x06: '>u2', 0x07: '>u4', 0x08: '>u8',
    0x09: '>f4', 0x0A: '>f8',
    0x0C: '>c8', 0x0D: '>c16',
}
ENUM_DTYPES = {0x15: '>u1', 0x16: '>u2', 0x17: '>u4'}

# Flag in the type descriptor header indicating that the descriptor carries a label
FLAG_HAS_LABEL = 0x40

# Flavor of the measure data type that represents a LabVIEW time stamp
FLAVOR_TIMESTAMP = 6

DATATYPE_RAW = 'raw_C4universal'
LAYER_TYPE_SURFACE = 'Surface'
DEFAULT_DISTANCE_UNIT = 'um'


class _Reader:
    """
    Minimal big-endian cursor over a bytes buffer.
    """

    def __init__(self, buffer, offset=0):
        self.buffer = buffer
        self.offset = offset

    def unpack(self, fmt):
        values = struct.unpack_from(fmt, self.buffer, self.offset)
        self.offset += struct.calcsize(fmt)
        return values if len(values) > 1 else values[0]

    def read_bytes(self, n):
        if self.offset + n > len(self.buffer):
            raise CorruptedFileError('Unexpected end of file.')
        data = self.buffer[self.offset:self.offset + n]
        self.offset += n
        return data

    def read_pascal_string(self):
        n = self.unpack('>B')
        return self.read_bytes(n).decode('latin1')

    def read_string(self):
        n = self.unpack('>I')
        return self.read_bytes(n).decode('latin1')


class _TypeDescriptor:

    def __init__(self, type_code, label=None, **properties):
        self.type_code = type_code
        self.label = label
        self.properties = properties


def _read_type_descriptor(reader):
    """
    Reads a single flattened LabVIEW type descriptor. The length stored in the header is used to advance to the next
    descriptor, so that unknown trailing information does not break parsing.
    """
    start = reader.offset
    length, flags, type_code = reader.unpack('>HBB')
    end = start + length
    properties = {}
    if type_code in NUMERIC_DTYPES:
        reader.unpack('>B')  # unit information, always zero for unitless numerics
    elif type_code in ENUM_DTYPES:
        n = reader.unpack('>H')
        properties['items'] = [reader.read_pascal_string() for _ in range(n)]
    elif type_code == TYPE_STRING:
        reader.unpack('>i')  # dimension, always -1 (variable length)
    elif type_code == TYPE_ARRAY:
        ndims = reader.unpack('>H')
        reader.unpack(f'>{ndims}i')  # dimension sizes, always -1 (variable size)
        properties['ndims'] = ndims
        properties['element'] = reader.unpack('>H')
    elif type_code == TYPE_CLUSTER:
        n = reader.unpack('>H')
        properties['items'] = list(struct.unpack_from(f'>{n}H', reader.buffer, reader.offset))
        reader.offset += 2 * n
    elif type_code == TYPE_MEASURE_DATA:
        properties['flavor'] = reader.unpack('>H')
    elif type_code == TYPE_TYPEDEF:
        reader.unpack('>I')  # checksum of the type definition
        n = reader.unpack('>I')
        properties['names'] = [reader.read_pascal_string() for _ in range(n)]
        # The type definition embeds the descriptor of the actual type, which we read recursively. Its own length
        # field does not correspond to the stored number of bytes, so we rely on the length of the outer descriptor.
        inner = _read_type_descriptor(reader)
        reader.offset = end
        return _TypeDescriptor(type_code, label=inner.label, inner=inner, **properties)
    elif type_code not in (TYPE_VOID, TYPE_BOOLEAN, TYPE_VARIANT):
        raise UnsupportedFileFormatError(f'Unsupported LabVIEW type code 0x{type_code:02X} at offset {start}.')
    label = None
    if flags & FLAG_HAS_LABEL and reader.offset < end:
        label = reader.read_pascal_string().strip()
    reader.offset = end
    return _TypeDescriptor(type_code, label=label, **properties)


def _read_type_descriptors(reader):
    n = reader.unpack('>I')
    return [_read_type_descriptor(reader) for _ in range(n)]


def _read_array(reader, descriptor, descriptors):
    shape = reader.unpack(f'>{descriptor.properties["ndims"]}I')
    shape = (shape,) if isinstance(shape, int) else shape
    count = int(np.prod(shape))
    element = descriptors[descriptor.properties['element']]
    while element.type_code == TYPE_TYPEDEF:
        element = element.properties['inner']
    dtype = NUMERIC_DTYPES.get(element.type_code, ENUM_DTYPES.get(element.type_code))
    if dtype is not None:
        dtype = np.dtype(dtype)
        data = reader.read_bytes(count * dtype.itemsize)
        return np.frombuffer(data, dtype).reshape(shape)
    # Arrays of non-numeric elements are returned as flat list in row-major order
    return [_read_data(reader, element, descriptors) for _ in range(count)]


def _read_cluster(reader, descriptor, descriptors):
    result = {}
    for i, index in enumerate(descriptor.properties['items']):
        item = descriptors[index]
        name = item.label or f'item_{i}'
        if name in result:
            name = f'{name}_{i}'
        result[name] = _read_data(reader, item, descriptors)
    return result


def _read_data(reader, descriptor, descriptors):
    """
    Reads flattened LabVIEW data described by a type descriptor.
    """
    type_code = descriptor.type_code
    if type_code == TYPE_TYPEDEF:
        return _read_data(reader, descriptor.properties['inner'], descriptors)
    if type_code in NUMERIC_DTYPES or type_code in ENUM_DTYPES:
        dtype = np.dtype(NUMERIC_DTYPES.get(type_code, ENUM_DTYPES.get(type_code)))
        return np.frombuffer(reader.read_bytes(dtype.itemsize), dtype)[0].item()
    if type_code == TYPE_BOOLEAN:
        return bool(reader.unpack('>B'))
    if type_code == TYPE_STRING:
        return reader.read_string()
    if type_code == TYPE_ARRAY:
        return _read_array(reader, descriptor, descriptors)
    if type_code == TYPE_CLUSTER:
        return _read_cluster(reader, descriptor, descriptors)
    if type_code == TYPE_VARIANT:
        return _read_variant(reader)
    if type_code == TYPE_MEASURE_DATA and descriptor.properties['flavor'] == FLAVOR_TIMESTAMP:
        seconds, fraction = reader.unpack('>qQ')
        return LABVIEW_EPOCH + datetime.timedelta(seconds=seconds + fraction / 2**64)
    if type_code == TYPE_VOID:
        return None
    raise UnsupportedFileFormatError(f'Unsupported LabVIEW data type 0x{type_code:02X}.')


def _read_variant(reader):
    """
    Reads a flattened LabVIEW variant and returns its data. Variant attributes are discarded.
    """
    reader.unpack('>I')  # LabVIEW version
    descriptors = _read_type_descriptors(reader)
    n_top = reader.unpack('>H')
    top = [reader.unpack('>H') for _ in range(n_top)]
    data = [_read_data(reader, descriptors[index], descriptors) for index in top]
    n_attributes = reader.unpack('>I')
    for _ in range(n_attributes):
        reader.read_string()
        _read_variant(reader)
    return data[0] if len(data) == 1 else data


def _parse_register(register):
    """
    Parses the camera register dump consisting of key=value lines into a dict. Values that follow a <name>Selector
    entry and whose key starts with <name> belong to the selected item, which is appended to the key in brackets.
    """
    result = {}
    selectors = {}
    for line in register.splitlines():
        key, sep, value = line.partition('=')
        key, value = key.strip(), value.strip()
        if not sep or not key:
            continue
        if key.endswith('Selector'):
            selectors[key[:-len('Selector')]] = value
            continue
        for prefix, selected in selectors.items():
            if key.startswith(prefix):
                key = f'{key}[{selected}]'
                break
        result[key] = value
    return result


def _load_json(string):
    try:
        return json.loads(string)
    except (TypeError, ValueError):
        return None


def _layer_info(layers, meta, default_prefix):
    """
    Pairs each layer of a 3d array with its type and fixpoint scaling from the corresponding JSON meta string.
    """
    meta = _load_json(meta)
    if not isinstance(meta, list) or len(meta) != layers.shape[0]:
        meta = [{} for _ in range(layers.shape[0])]
    info = []
    for i, (layer, entry) in enumerate(zip(layers, meta)):
        name = entry.get('Type') or f'{default_prefix}_{i}'
        scaling = entry.get('FixPointScaling', 1.0)
        info.append((name, layer, scaling))
    return info


def _find_raw_cell(measurement):
    cells = measurement.get('matrix')
    if not cells:
        raise CorruptedFileError('The file does not contain any measurement data.')
    raw_cells = [cell for cell in cells if isinstance(cell.get('data'), dict) and 'FloatData' in cell['data']]
    if not raw_cells:
        datatypes = ', '.join(sorted({str(cell.get('dataType')) for cell in cells}))
        raise UnsupportedFileFormatError(f'Unsupported hdat data type(s): {datatypes}. Only {DATATYPE_RAW} is '
                                         f'currently supported.')
    if len(raw_cells) > 1:
        warnings.warn(f'The file contains {len(raw_cells)} measurement cells. Only the first one is imported.')
    return raw_cells[0]


def _build_metadata(measurement, cell, register, camera_meta):
    metadata = {}
    timestamp = measurement.get('Time Stamp')
    if isinstance(timestamp, datetime.datetime):
        metadata['timestamp'] = timestamp
    for key in ('ID', 'label', 'heliAlgo'):
        if measurement.get(key):
            metadata[key] = measurement[key]
    scanner = measurement.get('scanner', {})
    for key, value in scanner.items():
        if value != '':
            metadata[f'scanner/{key}'] = value
    optics = measurement.get('optics3D', {})
    for key, value in optics.get('illumination', {}).items():
        if not isinstance(value, np.ndarray):
            metadata[f'illumination/{key}'] = value
    for key, value in optics.get('orientation', {}).items():
        metadata[f'orientation/{key}'] = value
    if optics.get('workDist [mm]'):
        metadata['workDist [mm]'] = optics['workDist [mm]']
    metadata['dataType'] = cell.get('dataType')
    for entry in cell.get('keys', []):
        metadata[entry['key']] = entry['value']
    for key, value in cell.get('config', {}).items():
        if key != 'Register' and value != '':
            metadata[f'config/{key}'] = value
    if isinstance(camera_meta, dict):
        for key, value in camera_meta.items():
            # Skip lists such as the z-tags of every recorded frame
            if isinstance(value, (int, float, str, bool)):
                metadata[f'camera/{key}'] = value
    for key, value in register.items():
        metadata[f'register/{key}'] = value
    return metadata


@FileHandler.register_reader(suffix='.hdat', magic=b'\x00\x00\x00\x0a' + MAGIC)
def read_hdat(filehandle, read_image_layers=False, encoding='utf-8'):
    buffer = filehandle.read()
    reader = _Reader(buffer)
    try:
        version = reader.read_string()
    except (struct.error, CorruptedFileError):
        raise CorruptedFileError('File is too short to be a Heliotis hdat file.') from None
    if not version.startswith(MAGIC.decode()):
        raise CorruptedFileError('File does not start with a Heliotis hdat header.')
    reader.unpack('>I')  # reserved
    try:
        measurement = _read_variant(reader)
    except struct.error:
        raise CorruptedFileError('Unexpected end of file.') from None
    if not isinstance(measurement, dict):
        raise CorruptedFileError('Unexpected top-level data structure.')

    cell = _find_raw_cell(measurement)
    data = cell['data']
    register = _parse_register(cell.get('config', {}).get('Register', ''))
    camera_meta = _load_json(data.get('CameraMeta'))

    float_layers = _layer_info(data['FloatData'], data.get('FloatMeta'), 'FloatData')
    int_layers = _layer_info(data['IntData'], data.get('IntMeta'), 'IntData') if 'IntData' in data else []

    surface_layers = [(layer, scaling) for name, layer, scaling in float_layers if name == LAYER_TYPE_SURFACE]
    if not surface_layers:
        raise CorruptedFileError('The file does not contain a surface layer.')
    layer, scaling = surface_layers[0]
    unit = register.get('Scan3dDistanceUnit', DEFAULT_DISTANCE_UNIT)
    height = layer.astype(np.float64) * (scaling * get_unit_conversion(unit, 'um'))

    resolution = measurement.get('optics3D', {}).get('res [um/pix]')
    if not resolution:
        raise CorruptedFileError('The file does not contain the lateral resolution.')
    step_x = resolution['dX']
    step_y = resolution['dY']

    image_layers = {}
    if read_image_layers:
        surface_seen = False
        for name, layer, scaling in float_layers + int_layers:
            if name == LAYER_TYPE_SURFACE and not surface_seen:
                surface_seen = True
                continue
            if scaling != 1:
                layer = layer.astype(np.float32) * np.float32(scaling)
            else:
                layer = layer.astype(layer.dtype.newbyteorder('='))
            if name in image_layers:
                name = f'{name}_{len(image_layers)}'
            image_layers[name] = layer

    metadata = _build_metadata(measurement, cell, register, camera_meta)
    return RawSurface(height, step_x, step_y, metadata=metadata, image_layers=image_layers)
