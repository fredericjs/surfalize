import hashlib
import zipfile
import xml.etree.ElementTree as ElementTree

import dateutil

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