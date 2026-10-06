import hashlib
import zipfile
import xml.etree.ElementTree as ElementTree
from datetime import datetime

import dateutil
from dateutil import parser as date_parser
import numpy as np

from surfalize.exceptions import CorruptedFileError, UnsupportedFileFormatError
from surfalize.file.common import get_unit_conversion, FileHandler, read_array, RawSurface, decode

UNIT = 'm'
CONVERSION_FACTOR = get_unit_conversion(UNIT, 'um')
MAGIC = b'PK\x03\x04\x14'

DTYPE_MAP = {
    "I": "<u2",
    "L": "<u4",
    "F": "f4",
    "D": "f8",
}

NAMESPACE = 'http://www.opengps.eu/2008/ISO5436_2'
SCHEMA_LOCATION = f'{NAMESPACE} {NAMESPACE}/ISO5436_2.xsd'
XSI_NAMESPACE = 'http://www.w3.org/2001/XMLSchema-instance'
POINT_DATA_PATH = 'bindata/data.bin'
CHECKSUM_FILE = 'md5checksum.hex'
# Data types supported for writing. Only floating point types are written, since they represent non-measured points
# natively as NaN (ISO 25178-72), which makes a separate validity bit mask (bindata/valid.bin) unnecessary.
WRITE_DTYPE_MAP = {
    'D': '<f8',
    'F': '<f4',
}
# Allowed values of Record2/ProbingSystem/Type according to the ISO5436_2 schema
PROBING_SYSTEM_TYPES = ('Contacting', 'NonContacting', 'Software')
NOT_AVAILABLE = 'not available'

def xml_to_flat_dict(element, parent_key=''):
    data = {}
    # If the element has text (non-whitespace), add it directly to the dictionary
    if element.text and element.text.strip():
        data[parent_key] = element.text.strip()
    # Iterate over children and flatten them by appending the tag name to the parent key
    for child in element:
        child_key = f'{parent_key}{child.tag}' if parent_key else child.tag
        data.update(xml_to_flat_dict(child, child_key))
    return data

def _format_float(value):
    """Formats a float as the shortest string that round-trips exactly, which is a valid xsd:double."""
    return repr(float(value))

def _format_datetime(value, fallback):
    """
    Converts a datetime or a date string to an ISO 8601 string (xsd:dateTime). Returns the fallback if the value is
    None or cannot be interpreted as a date.
    """
    if isinstance(value, str):
        try:
            value = date_parser.parse(value)
        except (ValueError, OverflowError):
            return fallback
    if isinstance(value, datetime):
        return value.isoformat()
    return fallback

def _get_version():
    try:
        from surfalize import __version__
    except ImportError:
        return ''
    return __version__

def _sub_element(parent, tag, text=None):
    element = ElementTree.SubElement(parent, tag)
    if text is not None:
        element.text = str(text)
    return element

def _build_main_xml(nx, ny, step_x, step_y, datatype, data_checksum, metadata, comment):
    # The root element is qualified with the 'p' prefix, while all child elements are unqualified, as defined by the
    # ISO5436_2 schema (elementFormDefault="unqualified"). The prefix is written literally instead of being registered
    # globally with ElementTree.register_namespace, which would affect other users of ElementTree.
    root = ElementTree.Element('p:ISO5436_2', {
        'xmlns:p': NAMESPACE,
        'xmlns:xsi': XSI_NAMESPACE,
        'xsi:schemaLocation': SCHEMA_LOCATION,
    })

    # Record1: axis definition. X and Y are incremental axes with constant spacing, Z is an absolute axis whose values
    # are stored directly (in meters) in the binary point data file.
    record1 = _sub_element(root, 'Record1')
    _sub_element(record1, 'Revision', 'ISO5436 - 2000')
    _sub_element(record1, 'FeatureType', 'SUR')
    axes = _sub_element(record1, 'Axes')
    for name, step in (('CX', step_x), ('CY', step_y)):
        axis = _sub_element(axes, name)
        _sub_element(axis, 'AxisType', 'I')
        _sub_element(axis, 'DataType', 'D')
        _sub_element(axis, 'Increment', _format_float(step / CONVERSION_FACTOR))
        _sub_element(axis, 'Offset', '0')
    cz = _sub_element(axes, 'CZ')
    _sub_element(cz, 'AxisType', 'A')
    _sub_element(cz, 'DataType', datatype)
    _sub_element(cz, 'Increment', '1')
    _sub_element(cz, 'Offset', '0')

    # Record2: measurement metadata. Values that were read from an X3P file are stored by the reader under the
    # flattened element names (e.g. 'InstrumentManufacturer'), so they are written back if present.
    now = datetime.now().astimezone().isoformat()
    date = _format_datetime(metadata.get('Date', metadata.get('timestamp')), now)
    calibration_date = _format_datetime(metadata.get('CalibrationDate'), date)
    probing_type = str(metadata.get('ProbingSystemType', 'Software'))
    if probing_type not in PROBING_SYSTEM_TYPES:
        probing_type = 'Software'
    if comment is None:
        comment = metadata.get('Comment', f'Exported by surfalize {_get_version()}'.rstrip())

    record2 = _sub_element(root, 'Record2')
    _sub_element(record2, 'Date', date)
    if 'Creator' in metadata:
        _sub_element(record2, 'Creator', metadata['Creator'])
    instrument = _sub_element(record2, 'Instrument')
    for key in ('Manufacturer', 'Model', 'Serial', 'Version'):
        _sub_element(instrument, key, metadata.get(f'Instrument{key}', NOT_AVAILABLE))
    _sub_element(record2, 'CalibrationDate', calibration_date)
    probing_system = _sub_element(record2, 'ProbingSystem')
    _sub_element(probing_system, 'Type', probing_type)
    _sub_element(probing_system, 'Identification', metadata.get('ProbingSystemIdentification', NOT_AVAILABLE))
    _sub_element(record2, 'Comment', comment)

    # Record3: matrix dimensions and link to the binary point data
    record3 = _sub_element(root, 'Record3')
    matrix_dimension = _sub_element(record3, 'MatrixDimension')
    _sub_element(matrix_dimension, 'SizeX', nx)
    _sub_element(matrix_dimension, 'SizeY', ny)
    _sub_element(matrix_dimension, 'SizeZ', 1)
    data_link = _sub_element(record3, 'DataLink')
    _sub_element(data_link, 'PointDataLink', POINT_DATA_PATH)
    _sub_element(data_link, 'MD5ChecksumPointData', data_checksum.upper())

    # Record4: reference to the file containing the checksum of main.xml
    record4 = _sub_element(root, 'Record4')
    _sub_element(record4, 'ChecksumFile', CHECKSUM_FILE)

    ElementTree.indent(root, space='  ')
    # The XML declaration is written manually to match the declaration used by the OpenGPS reference implementation
    declaration = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
    return (declaration + ElementTree.tostring(root, encoding='unicode') + '\n').encode('utf-8')

@FileHandler.register_writer(suffix='.x3p')
def write_x3p(filehandle, surface, encoding='utf-8', dtype='D', comment=None):
    """
    Writes a surface to an OpenGPS X3P file according to ISO 25178-72 (ISO 5436-2).

    The file is a zip archive containing the XML header ``main.xml``, the binary point data ``bindata/data.bin`` and
    the MD5 checksum of ``main.xml`` in ``md5checksum.hex``. The lateral axes are written as incremental axes and the
    height axis as an absolute axis. All lengths are stored in meters, as required by the format. The point data is
    stored line by line in little-endian byte order with x as the fastest varying index. Non-measured points are
    stored as NaN, which is how ISO 25178-72 marks invalid points for floating point data types, so no separate
    validity file (``bindata/valid.bin``) is written.

    Metadata read from an X3P file (Record2) is written back where possible. Missing entries are filled with
    defaults. The measurement date falls back to the ``timestamp`` metadata entry or the current time.

    Parameters
    ----------
    filehandle : file-like
        Binary file-like object to write to.
    surface : Surface
        Surface object to write.
    encoding : str, Default utf-8
        Not used. The XML header is always encoded as UTF-8, as declared in its XML declaration. The argument only
        exists for compatibility with the common writer interface.
    dtype : {'D', 'F'}, Default 'D'
        Data type of the height values. 'D' writes 64-bit double precision floats, which preserves the full
        precision of the surface. 'F' writes 32-bit single precision floats, which halves the size of the point data
        at the cost of precision.
    comment : str | None, Default None
        Comment written to Record2. If None, the comment from the surface metadata is used if available, otherwise
        a default comment.

    Returns
    -------
    None
    """
    datatype = str(dtype).upper()
    if datatype not in WRITE_DTYPE_MAP:
        raise ValueError(f"Unsupported data type '{dtype}'. Supported data types are "
                         f"{', '.join(WRITE_DTYPE_MAP)}.")
    # Rows correspond to y and columns to x, so the C-ordered array has x as the fastest varying index, as required
    # by the format. surfalize works in micrometers, while X3P stores all values in meters.
    data = np.ascontiguousarray(np.asarray(surface.data, dtype=np.float64) / CONVERSION_FACTOR,
                                dtype=WRITE_DTYPE_MAP[datatype])
    data_bytes = data.tobytes()
    data_checksum = hashlib.md5(data_bytes).hexdigest()
    ny, nx = data.shape

    main_xml = _build_main_xml(nx, ny, surface.step_x, surface.step_y, datatype, data_checksum,
                               surface.metadata, comment)
    main_checksum = hashlib.md5(main_xml).hexdigest()

    with zipfile.ZipFile(filehandle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('main.xml', main_xml)
        archive.writestr(POINT_DATA_PATH, data_bytes)
        archive.writestr(CHECKSUM_FILE, f'{main_checksum} *main.xml\n'.encode('ascii'))

@FileHandler.register_reader(suffix='.x3p', magic=MAGIC)
def read_x3p(filehandle, read_image_layers=False, encoding='utf-8'):
    with zipfile.ZipFile(filehandle) as archive:
        contents = archive.namelist()
        if 'main.xml' not in contents:
            raise CorruptedFileError('File does not contain required main.xml file.') from None
        if 'md5checksum.hex' not in contents:
            raise CorruptedFileError('File does not contain required md5checksum.hex file.') from None
        with archive.open('md5checksum.hex') as checksum_file:
            checksum = decode(checksum_file.read(), encoding).split()[0]

        with archive.open('main.xml') as file:
            md5_hash = hashlib.md5()
            for chunk in iter(lambda: file.read(4096), b""):
                md5_hash.update(chunk)
            computed_checksum = md5_hash.hexdigest()

        if computed_checksum != checksum:
            raise CorruptedFileError('Checksum of main.xml file does not match expected checksum.') from None

        with archive.open('main.xml') as xml_file:
            tree = ElementTree.parse(xml_file)
        root = tree.getroot()
        record1 = root.find('Record1')
        if record1 is None:
            raise CorruptedFileError('File does not contain necessarcy Record1.') from None
        record2 = root.find('Record2')
        record3 = root.find('Record3')
        if record3 is None:
            raise CorruptedFileError('File does not contain necessarcy Record3.') from None

        feature_type = record1.find('FeatureType').text
        if feature_type != 'SUR':
            raise UnsupportedFileFormatError(
                f'The file containts features of type {feature_type}. However, only SUR is supported.')

        axes = record1.find("Axes")
        cx = axes.find("CX")
        cy = axes.find("CY")
        cz = axes.find("CZ")

        dtype = DTYPE_MAP[cz.find("DataType").text]

        step_x = float(cx.find("Increment").text) * CONVERSION_FACTOR
        step_y = float(cy.find("Increment").text) * CONVERSION_FACTOR

        matrix_dimensions = record3.find('MatrixDimension')
        nx = int(matrix_dimensions.find('SizeX').text)
        ny = int(matrix_dimensions.find('SizeY').text)
        nz = int(matrix_dimensions.find('SizeZ').text)

        if nz != 1:
            raise UnsupportedFileFormatError('Multilayer or volumetric file format is not supported.') from None

        bin_path = record3.find('DataLink/PointDataLink').text
        if bin_path is None:
            raise CorruptedFileError('Binary file containing topographical data not found.') from None

        with archive.open(bin_path, 'r') as data_file:
            data = read_array(data_file, dtype=dtype).reshape(ny, nx) * CONVERSION_FACTOR

        metadata = {}
        if record2 is not None:
            metadata = xml_to_flat_dict(record2)
            if 'Date' in metadata:
                metadata['Date'] = dateutil.parser.parse(metadata['Date'])
            if 'CalibrationDate' in metadata:
                metadata['CalibrationDate'] = dateutil.parser.parse(metadata['CalibrationDate'])

    return RawSurface(data, step_x, step_y, metadata=metadata)