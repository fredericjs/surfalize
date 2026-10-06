import struct
from datetime import datetime
import numpy as np
from .common import RawSurface, get_unit_conversion, Entry, Layout, FileHandler, write_array, decode
from ..exceptions import CorruptedFileError, UnsupportedFileFormatError

# File format specification taken from ISO 25178-71. Supports the ISO-1.0 and ISO-2.0 dialects.
MAGIC_ASCII_ISO1 = b'aISO-1.0'
MAGIC_BINARY_ISO1 = b'bISO-1.0'
MAGIC_ASCII_ISO2 = b'aISO-2.0'
MAGIC_BINARY_ISO2 = b'bISO-2.0'

FIXED_UNIT = 'm'
CONVERSION_FACTOR = get_unit_conversion(FIXED_UNIT, 'um')
ASCII_DATE_FORMAT = "%d%m%Y%H%M"
ASCII_FLOAT_PRECISION = 10
MANUFACID_SIZE = 10
# Neither version of the standard defines an "unknown date" convention, but this all-zero value
# is undocumented, vendor-specific practice observed in real files for a date that was never
# recorded.
UNSET_DATE = '0' * 12

# Used for writing; write_sdf only supports ISO-1.0.
LAYOUT_HEADER = Layout(
    Entry("ManufacID", "10s"),
    Entry("CreateDate", "12s"),
    Entry("ModDate", "12s"),
    Entry("NumPoints", "H"),
    Entry("NumProfiles", "H"),
    Entry("Xscale", "d"),
    Entry("Yscale", "d"),
    Entry("Zscale", "d"),
    Entry("Zresolution", "d"),
    Entry("Compression", "B"),
    Entry("DataType", "B"),
    Entry("CheckType", "B"),
)

# Used for reading. ManufacID is read separately (see _decode_manufacturer_id) rather than
# through a Layout entry, since it needs custom NUL-tolerant decoding. NumPoints/NumProfiles are
# 16-bit for ISO-1.0 and 32-bit for ISO-2.0, so the ISO-2.0 layout just swaps those two entries.
_BINARY_HEADER_ENTRIES_ISO1 = [
    Entry("CreateDate", "12s"),
    Entry("ModDate", "12s"),
    Entry("NumPoints", "H"),
    Entry("NumProfiles", "H"),
    Entry("Xscale", "d"),
    Entry("Yscale", "d"),
    Entry("Zscale", "d"),
    Entry("Zresolution", "d"),
    Entry("Compression", "B"),
    Entry("DataType", "B"),
    Entry("CheckType", "B"),
]
_BINARY_HEADER_ENTRIES_ISO2 = list(_BINARY_HEADER_ENTRIES_ISO1)
_BINARY_HEADER_ENTRIES_ISO2[2] = Entry("NumPoints", "I")
_BINARY_HEADER_ENTRIES_ISO2[3] = Entry("NumProfiles", "I")

LAYOUT_BINARY_HEADER = {
    MAGIC_BINARY_ISO1: Layout(*_BINARY_HEADER_ENTRIES_ISO1),
    MAGIC_BINARY_ISO2: Layout(*_BINARY_HEADER_ENTRIES_ISO2),
}

ASCII_HEADER_TYPES = {
    "ManufacID": str,
    "CreateDate": str,
    "ModDate": str,
    "NumPoints": int,
    "NumProfiles": int,
    "Xscale": float,
    "Yscale": float,
    "Zscale": float,
    "Zresolution": float,
    "Compression": int,
    "DataType": int,
    "CheckType": int,
}

DTYPE_MAP = {
    3: "f",  # BINARY32
    4: "b",  # INT8
    5: "h",  # INT16
    6: "i",  # INT32
    7: "d",  # DOUBLE
}

ASCII_INVALID_VALUE = 'BAD'

# The standard's invalid-point sentinel is each type's minimum representable value.
BINARY_INVALID_VALUE_MAP = {
    3: np.finfo(np.float32).min,
    4: np.iinfo(np.int8).min,
    5: np.iinfo(np.int16).min,
    6: np.iinfo(np.int32).min,
    7: np.finfo(np.float64).min,
}

def _parse_date(value):
    if value == UNSET_DATE:
        return None
    return datetime.strptime(value, ASCII_DATE_FORMAT)

def _decode_manufacturer_id(raw, encoding):
    # The standard requires this field to be space-padded, but some real-world writers (e.g.
    # MountainsMap) NUL-terminate it instead and leave leftover memory content after the NUL,
    # which must be discarded rather than decoded.
    trimmed = raw.split(b'\x00', 1)[0]
    if encoding == 'auto':
        return decode(trimmed, encoding).strip()
    return trimmed.decode(encoding, errors='replace').strip()

def read_ascii_sdf(filehandle, encoding="utf-8"):
    contents = filehandle.read().decode('ascii').lstrip()
    # Handles files that omit the trailer record entirely, ending right after the data's
    # closing "*" and leaving only 2 delimiters instead of the usual 3.
    parts = contents.split('*')
    if len(parts) == 3:
        header_section, data_section, trailer_section = parts
        end = ''
    elif len(parts) == 4:
        header_section, data_section, trailer_section, end = parts
    else:
        raise ValueError
    if end.strip() != '':
        raise ValueError

    header = dict()
    for line in header_section.lstrip().splitlines():
        name, value = line.split('=')
        name, value = name.strip(), value.strip()
        if name not in ASCII_HEADER_TYPES:
            raise CorruptedFileError(f'Unknown header field "{name}" detected.')
        header[name] = ASCII_HEADER_TYPES[name](value)

    if header['DataType'] not in DTYPE_MAP:
        raise CorruptedFileError(f"Unsupported DataType in SDF file: {header['DataType']}")

    if 'CreateDate' in header:
        header['CreateDate'] = _parse_date(header['CreateDate'])
    if 'ModDate' in header:
        header['ModDate'] = _parse_date(header['ModDate'])

    data_section = data_section.replace(ASCII_INVALID_VALUE, 'NAN')
    data = np.fromstring(data_section, sep=' ', dtype='d').reshape(header['NumProfiles'], header['NumPoints'])
    data *= CONVERSION_FACTOR * header['Zscale']
    step_x = header['Xscale'] * CONVERSION_FACTOR
    step_y = header['Yscale'] * CONVERSION_FACTOR
    metadata = header

    return RawSurface(data, step_x, step_y, metadata=metadata, image_layers=None)

def read_binary_sdf(filehandle, magic, encoding="utf-8"):
    manufacturer_id = _decode_manufacturer_id(filehandle.read(MANUFACID_SIZE), encoding)
    header = LAYOUT_BINARY_HEADER[magic].read(filehandle)
    header['ManufacID'] = manufacturer_id
    num_points = header["NumPoints"]
    num_profiles = header["NumProfiles"]
    data_type = header["DataType"]

    if data_type not in DTYPE_MAP:
        raise CorruptedFileError(f"Unsupported DataType in SDF file: {data_type}")

    data_format = DTYPE_MAP[data_type]
    item_size = struct.calcsize(data_format)
    data_size = item_size * num_points * num_profiles
    data_bytes = filehandle.read(data_size)

    if len(data_bytes) != data_size:
        raise CorruptedFileError("Unexpected end of file or corrupt data section.")

    data = np.frombuffer(data_bytes, dtype=np.dtype(data_format))

    missing_value = BINARY_INVALID_VALUE_MAP[data_type]
    invalid_mask = (data == missing_value)

    data = data.astype('float64')
    data[invalid_mask] = np.nan
    data = data * header["Zscale"] * CONVERSION_FACTOR
    data = data.reshape((num_profiles, num_points))

    step_x = header["Xscale"] * CONVERSION_FACTOR
    step_y = header["Yscale"] * CONVERSION_FACTOR
    return RawSurface(data, step_x, step_y, metadata=header)

@FileHandler.register_reader(suffix='.sdf', magic=(MAGIC_ASCII_ISO1, MAGIC_BINARY_ISO1, MAGIC_ASCII_ISO2, MAGIC_BINARY_ISO2))
def read_sdf(filehandle, read_image_layers=False, encoding="utf-8"):
    magic = filehandle.read(8)
    if magic in (MAGIC_ASCII_ISO1, MAGIC_ASCII_ISO2):
        return read_ascii_sdf(filehandle, encoding=encoding)
    elif magic in (MAGIC_BINARY_ISO1, MAGIC_BINARY_ISO2):
        return read_binary_sdf(filehandle, magic, encoding=encoding)
    else:
        raise CorruptedFileError(f'Invalid file magic "{magic.decode()}" detected.')

@FileHandler.register_writer(suffix='.sdf')
def write_sdf(filehandle, surface, encoding='utf-8', binary=True):
    now = datetime.now()
    mod_date = now.strftime(ASCII_DATE_FORMAT)
    # if the surface contains a timestamp in the metadata, we use this one, otherwise we set the create date to the same
    # value as the modified date
    if 'timestamp' in surface.metadata:
        create_date = surface.metadata['timestamp'].strftime(ASCII_DATE_FORMAT)
    else:
        create_date = mod_date

    # Here, we divide the data by a power of 10 so that there is only one significant digit before
    # the decimal point. This way, we can make use of the maximum resolution of the ascii encoded
    # decimal places.
    data = surface.data.astype('float64')
    max_abs = np.nanmax(np.abs(data))
    if max_abs == 0:
        scale_factor = 1
    else:
        exponent = np.floor(np.log10(max_abs))
        scale_factor = 10 ** (-exponent)
    data = data * scale_factor

    conversion_factor = get_unit_conversion('um', FIXED_UNIT)
    header = {
        "ManufacID": 'surfalize'.ljust(10),
        "CreateDate": create_date,
        "ModDate": mod_date,
        "NumPoints": surface.size.x,
        "NumProfiles": surface.size.y,
        "Xscale": surface.step_x * conversion_factor,
        "Yscale": surface.step_y * conversion_factor,
        "Zscale": get_unit_conversion('um', FIXED_UNIT) / scale_factor,
        # Zresolution = original base resolution of the measurement instrument
        # The standard says to fill this with a negative number when the value is unknown
        "Zresolution": -1,
        "Compression": 0, # no compression
        "DataType": 7, # data type double should be default
        "CheckType": 0, # should be zero according to standard
    }

    # Write in binary mode
    if binary:
        filehandle.write(MAGIC_BINARY_ISO1) # write magic identifier
        LAYOUT_HEADER.write(filehandle, header)
        binary_data = np.where(np.isnan(data), BINARY_INVALID_VALUE_MAP[7], data)
        write_array(binary_data, filehandle)
    # Write in ascii mode
    else:
        CRLF = '\r\n'.encode('ascii')
        filehandle.write(MAGIC_ASCII_ISO1 + CRLF)
        for k, v in header.items():
            filehandle.write(f'{k} = {v}\r\n'.encode('ascii'))
        filehandle.write('*'.encode('ascii') + CRLF)
        line_values = []
        for i, value in enumerate(data.flatten()):
            if np.isnan(value):
                line_values.append('BAD'.ljust(ASCII_FLOAT_PRECISION + 2))
            else:
                line_values.append(f'{value:.{ASCII_FLOAT_PRECISION}f}')
            if i % 10 == 0 or i == data.size - 1:
                filehandle.write(' '.join(line_values).encode('ascii') + CRLF)
                line_values = []

        filehandle.write('*'.encode('ascii') + CRLF)
        filehandle.write('ExportedBy = Surfalize'.encode('ascii') + CRLF)
        filehandle.write('*'.encode('ascii') + CRLF)
