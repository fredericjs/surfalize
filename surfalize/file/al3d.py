import struct
import numpy as np
from ..exceptions import CorruptedFileError


MAGIC = b'AliconaImaging\x00\r\n'
TAG_LAYOUT = '20s30s2s'

def read_tag(filehandle):
    key, value, lf = [val.decode().rstrip('\x00') for val in struct.unpack(TAG_LAYOUT, filehandle.read(52))]
    if lf != '\r\n':
        raise CorruptedFileError('Tag with incorrect delimiter detected.')
    return key, value

def read_al3d(filepath):
    with open(filepath, 'rb') as file:
        magic = file.read(17)
        if magic != MAGIC:
            raise CorruptedFileError('Incompatible file magic detected.')
        header = dict()
        key, value = read_tag(file)
        if key != 'Version':
            raise CorruptedFileError('Version tag expected but not found.')
        header[key] = value

        key, value = read_tag(file)
        if key != 'TagCount':
            raise CorruptedFileError('TagCount tag expected but not found.')
        header[key] = value

        for _ in range(int(header['TagCount'])):
            key, value = read_tag(file)
            header[key] = value

        nx = int(header['Cols'])
        ny = int(header['Rows'])
        step_x = float(header['PixelSizeXMeter']) * 1e6
        step_y = float(header['PixelSizeYMeter']) * 1e6
        offset = int(header['DepthImageOffset'])
        file.seek(offset)
        data = np.fromfile(file, dtype=np.float32, count=nx * ny, offset=0).reshape(ny, nx)

    invalidValue = float(header['InvalidPixelValue'])
    data[data == invalidValue] = np.nan

    data *= 1e6 # Conversion from m to um

    return (data, step_x, step_y)
