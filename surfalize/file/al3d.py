"""
Reader and writer for the AL3D file format (``.al3d``).

The format is specified in the following documents. Where they differ, the newer one is followed.

- AL3D File Format Specification, version 1.0 (2005)
- Bruker Alicona File Formats, version 1.0 (2025), chapter AL3D Format

Only the depth image is read and written. Depths that are not repeatable enough are set to NaN by the reader.
"""
import struct
import numpy as np
from ..exceptions import CorruptedFileError
from .common import RawSurface, FileHandler, decode

MAGIC = b'AliconaImaging\x00\r\n'
TAG_LAYOUT = '20s30s2s'
# The tags are ISO 8859-1 strings, so the encoding arguments of the reader and writer are not used
ENCODING = 'latin-1'
# Scanlines are padded to a multiple of this many bytes
ALIGNMENT = 8
# Depths from this value upwards are invalid. This is 3e15 as float32.
INVALID_PIXEL_VALUE = float(np.float32(3.0e15))

def read_tag(filehandle):
    # The strings of a tag are zero-terminated, anything after the first zero byte is ignored
    key, value, lf = [decode(val, ENCODING).split('\x00', 1)[0] for val in struct.unpack(TAG_LAYOUT, filehandle.read(52))]
    if lf != '\r\n':
        raise CorruptedFileError('Tag with incorrect delimiter detected.')
    return key, value

def write_tag(filehandle, key, value):
    binary_tag = struct.pack(TAG_LAYOUT,
                             key.encode(ENCODING),
                             str(value).encode(ENCODING),
                             '\r\n'.encode(ENCODING))
    filehandle.write(binary_tag)

def scanline_size(nx, itemsize):
    return -(-nx * itemsize // ALIGNMENT) * ALIGNMENT

def read_image(filehandle, offset, nx, ny, dtype):
    itemsize = np.dtype(dtype).itemsize
    stride = scanline_size(nx, itemsize)
    filehandle.seek(offset)
    buffer = filehandle.read(stride * ny)
    # The padding of the last scanline does not need to be present
    if len(buffer) < stride * (ny - 1) + nx * itemsize:
        raise CorruptedFileError('Image data is truncated.')
    return np.ndarray((ny, nx), dtype=dtype, buffer=buffer, strides=(stride, itemsize))

def read_unrepeatable(filehandle, header, nx, ny):
    """
    Returns a mask of the depths that are not repeatable enough, or None if the file has no repeatability map. The
    repeatability is (pixel value * DepthResShiftVal + DepthResMinVal) / sqrt(8) and the maximum DepthResFilterVal /
    sqrt(8).

    The depths are masked because the Bruker Alicona software has a dynamic filter for the repeatability, which is set
    to DepthResFilterVal by default. Without the mask, more points would be shown than by default in that software.

    The repeatability is only read from the file itself. The dem.al3d of a database folder might have none, since the
    Bruker Alicona software can store it in the files next to it: the map as qualitymap.png or qualitymap.bmp and the
    parameters as the minVal, shift and filterVal attributes of the depthQuality element in info.xml. These files are
    currently not read, so the depths are then not masked.
    """
    plane = header.get('DepthImageResPtr')
    filter_value = header.get('DepthResFilterVal')
    texture_offset = int(header.get('TextureImageOffset', 0))
    if not plane or not filter_value or texture_offset == 0:
        return None
    codes = read_image(filehandle, texture_offset + int(plane) * scanline_size(nx, 1) * ny, nx, ny, np.uint8)
    # The division by sqrt(8) cancels out in the comparison
    return codes * float(header.get('DepthResShiftVal', 0)) + float(header.get('DepthResMinVal', 0)) > float(filter_value)

@FileHandler.register_writer(suffix='.al3d')
def write_al3d(filehandle, surface, encoding=ENCODING):
    header = dict()
    header['Version'] = 1
    header['TagCount'] = 9
    header['Cols'] = surface.size.x
    header['IconOffset'] = 0
    header['DepthImageOffset'] = 845
    header['InvalidPixelValue'] = INVALID_PIXEL_VALUE
    header['PixelSizeYMeter'] = surface.step_y * 1e-6
    header['PixelSizeXMeter'] = surface.step_x * 1e-6
    header['NumberOfPlanes'] = 0
    header['Rows'] = surface.size.y
    header['TextureImageOffset'] = 0

    filehandle.write(MAGIC)
    for key, value in header.items():
        write_tag(filehandle, key, value)
    pos = filehandle.tell()
    n_padding = header['DepthImageOffset'] - pos - 2
    filehandle.write(b'\x00' * n_padding + b'\r\n')
    # Non-measured points are stored as the invalid pixel value
    data = np.nan_to_num(surface.data * 1e-6, nan=INVALID_PIXEL_VALUE, posinf=INVALID_PIXEL_VALUE,
                         neginf=INVALID_PIXEL_VALUE)
    image = np.zeros((header['Rows'], scanline_size(header['Cols'], 4) // 4), dtype='<f4')
    image[:, :header['Cols']] = data
    filehandle.write(image.tobytes())

@FileHandler.register_reader(suffix='.al3d', magic=MAGIC)
def read_al3d(filehandle, read_image_layers=False, encoding=ENCODING):
    """Reads the depth image of an AL3D file. Depths that are not repeatable enough are set to NaN."""
    magic = filehandle.read(17)
    if magic != MAGIC:
        raise CorruptedFileError('Incompatible file magic detected.')
    header = dict()
    key, value = read_tag(filehandle)
    if key != 'Version':
        raise CorruptedFileError('Version tag expected but not found.')
    header[key] = value

    key, value = read_tag(filehandle)
    if key != 'TagCount':
        raise CorruptedFileError('TagCount tag expected but not found.')
    header[key] = value

    for _ in range(int(header['TagCount'])):
        key, value = read_tag(filehandle)
        header[key] = value

    nx = int(header['Cols'])
    ny = int(header['Rows'])
    step_x = float(header['PixelSizeXMeter']) * 1e6
    step_y = float(header['PixelSizeYMeter']) * 1e6
    offset = int(header['DepthImageOffset'])
    data = read_image(filehandle, offset, nx, ny, '<f4').astype(np.float64)

    # Only depths smaller than the invalid pixel value are valid. This also excludes NaN.
    invalid_value = float(header.get('InvalidPixelValue', 'nan'))
    if not np.isfinite(invalid_value):
        invalid_value = INVALID_PIXEL_VALUE
    data[~(data < invalid_value)] = np.nan

    unrepeatable = read_unrepeatable(filehandle, header, nx, ny)
    if unrepeatable is not None:
        data[unrepeatable] = np.nan

    data *= 1e6 # Conversion from m to um

    return RawSurface(data, step_x, step_y)
