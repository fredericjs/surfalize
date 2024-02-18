# This code was only tested on .opd files with an itemsize of 2

from dataclasses import dataclass
import struct
import numpy as np
from ..exceptions import CorruptedFileError

BLOCK_SIZE = 24
BLOCK_NAME_SIZE = 16
INT16_MAX = 32767

dtype = {1: 'c', 2: '<i2', 4: '4f'}


@dataclass
class Block:
    type: int
    size: int
    flags: int
    offset: int = None


def read_block(filehandle):
    name = filehandle.read(16).decode().rstrip('\x00')
    type_, size, flags = struct.unpack('<hlH', filehandle.read(8))
    return name, Block(type_, size, flags)

def read_opd(filepath):
    with open(filepath, 'rb') as file:
        file.read(2)  # skipping header
        name, directory_block = read_block(file)
        if name != 'Directory':
            raise CorruptedFileError('Directory block not found.')
        n_blocks = int(directory_block.size / BLOCK_SIZE)
        blocks = dict()
        for _ in range(n_blocks - 1):
            name, block = read_block(file)
            blocks[name] = block
        offset = file.tell()
        for block in blocks.values():
            block.offset = offset
            offset += block.size

        file.seek(blocks['RAW_DATA'].offset, 0)
        nx, ny, itemsize = struct.unpack('<HHH', file.read(6))
        data_length = nx * ny
        if data_length * itemsize != blocks['RAW_DATA'].size - 6:
            raise CorruptedFileFormatError('Size of data does not match expected size.')

        data = np.fromfile(file, dtype=dtype[itemsize], count=data_length)
        data = np.rot90(data.reshape(nx, ny)).astype('float64')

        # Mask invalid datapoints
        if itemsize == 2:
            data[data == INT16_MAX] = np.nan
            data[data == -INT16_MAX - 1] = np.nan

        if 'Wavelength' in blocks:
            file.seek(blocks['Wavelength'].offset, 0)
            wavelength = struct.unpack('<f', file.read(blocks['Wavelength'].size))[0]
        else:
            wavelength = 1.0

        if 'Mult' in blocks:
            file.seek(blocks['Mult'].offset, 0)
            mult = struct.unpack('<H', file.read(blocks['Mult'].size))[0]
        else:
            mult = 1.0

        if 'Aspect' in blocks:
            file.seek(blocks['Aspect'].offset, 0)
            aspect = struct.unpack('<f', file.read(blocks['Aspect'].size))[0]
        else:
            aspect = 1.0

        if 'Pixel_size' in blocks:
            file.seek(blocks['Pixel_size'].offset, 0)
            pixel_size = struct.unpack('<f', file.read(blocks['Pixel_size'].size))[0]
        else:
            pixel_size = 1.0

        scale_z = wavelength / mult * 1e-6

        data *= scale_z

        step_x = pixel_size
        step_y = pixel_size * aspect

        return data, step_x, step_y