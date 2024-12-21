import struct
import numpy as np
from ..exceptions import CorruptedFileError
from .common import RawSurface, FileHandler, read_array, write_array

MAGIC = b'AliconaImaging\x00\r\n'
TAG_LAYOUT = '20s30s2s'
DTYPE = 'float32'

def read_tag(filehandle):
    key, value, lf = [val.decode().rstrip('\x00') for val in struct.unpack(TAG_LAYOUT, filehandle.read(52))]
    if lf != '\r\n':
        raise CorruptedFileError('Tag with incorrect delimiter detected.')
    return key, value

def write_tag(filehandle, key, value, encoding='utf-8'):
    binary_tag = struct.pack(TAG_LAYOUT,
                             key.encode(encoding),
                             str(value).encode(encoding),
                             '\r\n'.encode(encoding))
    filehandle.write(binary_tag)

@FileHandler.register_writer(suffix='.al3d')
def write_al3d(filehandle, surface, encoding='utf-8'):
    header = dict()
    header['Version'] = 1
    header['TagCount'] = 9
    header['Cols'] = surface.size.x
    header['IconOffset'] = 0
    header['DepthImageOffset'] = 845
    header['InvalidPixelValue'] = float('nan')
    header['PixelSizeYMeter'] = surface.step_y * 1e-6
    header['PixelSizeXMeter'] = surface.step_x * 1e-6
    header['NumberOfPlanes'] = 0
    header['Rows'] = surface.size.y
    header['TextureImageOffset'] = 0

    filehandle.write(MAGIC)
    for key, value in header.items():
        write_tag(filehandle, key, value, encoding=encoding)
    pos = filehandle.tell()
    n_padding = header['DepthImageOffset'] - pos - 2
    filehandle.write(b'\x00' * n_padding + b'\r\n')
    data = surface.data.astype(DTYPE) * 1e-6
    write_array(data, filehandle)

@FileHandler.register_reader(suffix='.al3d', magic=MAGIC)
def read_al3d(filehandle, read_image_layers=False, encoding='utf-8'):
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
    filehandle.seek(offset)
    data = read_array(filehandle, dtype=np.float32, count=nx * ny, offset=0).reshape(ny, nx)

    invalidValue = float(header['InvalidPixelValue'])
    data[data == invalidValue] = np.nan

    data *= 1e6 # Conversion from m to um

    return RawSurface(data, step_x, step_y)
