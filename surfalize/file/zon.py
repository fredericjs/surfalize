# Information on the file format was reconstructed from the Gwyddion `zonfile.c` module
# (https://gwyddion.net). The Keyence ZON (.zon) format is a container that starts with an 8-byte
# "KPK" header, followed by an embedded BMP preview image and a trailing ZIP archive. The ZIP archive
# holds a number of XML files describing the acquisition (including the lateral and vertical
# calibration) and several binary data files. Each binary data file begins with a 16-byte header
# (xres, yres, itemsize, rowstride as little-endian uint32) and is identified by its item size:
# 4 bytes -> int32 height data, 1 byte -> uint8 mask of invalid pixels, 3 bytes -> RGB image. The
# files inside the archive are frequently compressed individually using zstd.

import io
import struct
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter

import numpy as np
from dateutil import parser as date_parser

from .common import RawSurface, FileHandler
from ..exceptions import CorruptedFileError, FileFormatError, UnsupportedFileFormatError

MAGIC_KPK = b'KPK'
MAGIC_BM = b'BM'
MAGIC_PK = b'PK\x03\x04'
MAGIC_ZSTD = b'\x28\xb5\x2f\xfd'
UTF8_BOM = b'\xef\xbb\xbf'

# The leading header consists of an 8-byte "KPK" block, after which the embedded BMP starts.
PK_HEADER_SIZE = 8

# XML files / subtrees that are skipped because they are either huge (palettes, data maps) or hold
# verbose device dumps that are of no interest for the reconstruction of the surface.
IGNORE_ROOTS = {'DataMap', 'DeviceDumpInformations'}
IGNORE_TAGS = {'Palette'}

CALIBRATION_XY_KEY = 'Calibration/XYCalibration/MeterPerPixel'
CALIBRATION_Z_KEY = 'Calibration/ZCalibration/MeterPerUnit'
SCAN_DATETIME_KEY = 'ScanInformations/ScanDateTime'

# Item sizes used to identify the type of a binary data file
ITEMSIZE_HEIGHT = 4
ITEMSIZE_MASK = 1
ITEMSIZE_RGB = 3

_zstd_decompress = None


def _get_zstd_decompress():
    """
    Return a function that decompresses a single zstd frame, looking up an available backend.

    The zstd frames written by the LEXP/ZON software do not store the decompressed size, so the
    backend must support streaming decompression. The stdlib ``compression.zstd`` module (Python 3.14+)
    is preferred, falling back to the ``pyzstd`` or ``zstandard`` third-party packages.
    """
    global _zstd_decompress
    if _zstd_decompress is not None:
        return _zstd_decompress
    try:
        from compression import zstd  # Python >= 3.14
        _zstd_decompress = zstd.decompress
        return _zstd_decompress
    except ImportError:
        pass
    try:
        import pyzstd
        _zstd_decompress = pyzstd.decompress
        return _zstd_decompress
    except ImportError:
        pass
    try:
        import zstandard

        def _decompress(data):
            # decompressobj handles frames that do not store the content size
            return zstandard.ZstdDecompressor().decompressobj().decompress(data)

        _zstd_decompress = _decompress
        return _zstd_decompress
    except ImportError:
        pass
    raise UnsupportedFileFormatError(
        'Reading ZON files requires zstd support. Install the "zstandard" package '
        '(pip install zstandard) or run on Python 3.14 or newer.')


def _flatten_xml(element, result, path):
    """
    Recursively flatten an XML element into ``result``, mapping the full element path to its text
    content. Repeated sibling elements (i.e. arrays such as a calibration table) are disambiguated by
    an index suffix ``Tag[i]`` so that no entries are silently overwritten. Subtrees whose tag is
    contained in ``IGNORE_TAGS`` are skipped.

    Parameters
    ----------
    element : xml.etree.ElementTree.Element
        The element to flatten.
    result : dict
        Dictionary that the flattened entries are written into.
    path : str
        The path of ``element`` itself, already including its (possibly indexed) tag.
    """
    if element.text is not None and element.text.strip():
        result[path] = element.text.strip()
    tag_counts = Counter(child.tag for child in element)
    seen = {}
    for child in element:
        if child.tag in IGNORE_TAGS:
            continue
        if tag_counts[child.tag] > 1:
            index = seen.get(child.tag, 0)
            seen[child.tag] = index + 1
            child_path = f'{path}/{child.tag}[{index}]'
        else:
            child_path = f'{path}/{child.tag}'
        _flatten_xml(child, result, child_path)


def _extract_zip_bytes(filehandle):
    """
    Read the container, validate the header and return the bytes of the trailing ZIP archive.
    """
    buffer = filehandle.read()
    if buffer[:3] != MAGIC_KPK or buffer[PK_HEADER_SIZE:PK_HEADER_SIZE + 2] != MAGIC_BM:
        raise FileFormatError('File does not start with a ZON KPK/BM header.')
    # The size of the embedded BMP is stored in its own header, right after the "BM" magic
    bmp_size = struct.unpack('<I', buffer[PK_HEADER_SIZE + 2:PK_HEADER_SIZE + 6])[0]
    zip_start = PK_HEADER_SIZE + bmp_size
    if buffer[zip_start:zip_start + len(MAGIC_PK)] != MAGIC_PK:
        # The stored BMP size is occasionally off; search a little further for the ZIP signature
        found = buffer.find(MAGIC_PK, zip_start, zip_start + 1024 + len(MAGIC_PK))
        if found == -1:
            raise CorruptedFileError('Could not locate the embedded ZIP archive.')
        zip_start = found
    return buffer[zip_start:]


def _read_data_file(content):
    """
    Parse a binary data file. Returns ``(xres, yres, itemsize, rowstride, body)``.
    """
    xres, yres, itemsize, rowstride = struct.unpack('<IIII', content[:16])
    return xres, yres, itemsize, rowstride, content[16:]


def _reshape_rows(body, dtype, yres, rowstride, xres, channels=1):
    """
    Build a 2d (or 3d for multi-channel) array from the raw row-major buffer, honouring a row stride
    that may include trailing padding.
    """
    array = np.frombuffer(body, dtype=dtype)
    values_per_row = rowstride // np.dtype(dtype).itemsize
    array = array.reshape(yres, values_per_row)[:, :xres * channels]
    if channels > 1:
        array = array.reshape(yres, xres, channels)
    return array.copy()


@FileHandler.register_reader(suffix='.zon', magic=MAGIC_KPK)
def read_zon(filehandle, read_image_layers=False, encoding='utf-8'):
    zip_bytes = _extract_zip_bytes(filehandle)

    metadata = {}
    data_files = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for name in archive.namelist():
            content = archive.read(name)
            if content[:len(MAGIC_ZSTD)] == MAGIC_ZSTD:
                content = _get_zstd_decompress()(content)
            if content[:len(UTF8_BOM)] == UTF8_BOM:
                content = content[len(UTF8_BOM):]
            if content[:1] == b'<':
                try:
                    root = ET.fromstring(content)
                except ET.ParseError:
                    continue
                if root.tag in IGNORE_ROOTS:
                    continue
                _flatten_xml(root, metadata, root.tag)
            elif content[:len(MAGIC_PK)] == MAGIC_PK:
                # Nested ZIP archives only contain data already present in the main archive
                continue
            elif len(content) > 16:
                data_files.append(_read_data_file(content))

    if CALIBRATION_XY_KEY not in metadata:
        raise CorruptedFileError(f'Missing calibration field {CALIBRATION_XY_KEY}.')
    if CALIBRATION_Z_KEY not in metadata:
        raise CorruptedFileError(f'Missing calibration field {CALIBRATION_Z_KEY}.')
    meter_per_pixel = float(metadata[CALIBRATION_XY_KEY])
    meter_per_unit = float(metadata[CALIBRATION_Z_KEY])
    step = meter_per_pixel * 1e6  # m -> um

    height = None
    mask = None
    rgb_layers = []
    for xres, yres, itemsize, rowstride, body in data_files:
        if itemsize == ITEMSIZE_HEIGHT and height is None:
            raw = _reshape_rows(body, '<i4', yres, rowstride, xres)
            height = raw.astype(np.float64) * meter_per_unit * 1e6  # raw -> um
        elif itemsize == ITEMSIZE_MASK and mask is None:
            mask = _reshape_rows(body, np.uint8, yres, rowstride, xres)
        elif itemsize == ITEMSIZE_RGB and read_image_layers:
            rgb_layers.append(_reshape_rows(body, np.uint8, yres, rowstride, xres, channels=3))

    if height is None:
        raise CorruptedFileError('The file does not contain a height channel.')

    # The mask flags non-measured pixels, which we represent as NaN like the other readers.
    if mask is not None and mask.shape == height.shape:
        height[mask > 0] = np.nan

    # A ZON file may store the same RGB image more than once; expose only the distinct ones.
    image_layers = {}
    unique_layers = []
    for rgb in rgb_layers:
        if not any(np.array_equal(rgb, existing) for existing in unique_layers):
            unique_layers.append(rgb)
    for index, rgb in enumerate(unique_layers):
        image_layers['RGB' if index == 0 else f'RGB_{index + 1}'] = rgb

    scan_datetime = metadata.get(SCAN_DATETIME_KEY)
    if scan_datetime:
        try:
            metadata['timestamp'] = date_parser.parse(scan_datetime)
        except (ValueError, OverflowError):
            pass

    return RawSurface(height, step, step, metadata=metadata, image_layers=image_layers)
