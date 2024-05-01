from dataclasses import dataclass
import struct
from enum import IntEnum
import dateutil
import numpy as np
from ..exceptions import CorruptedFileError, CorruptedFileError
from .common import RawSurface, get_unit_conversion

# This code was only tested on .opd files with an itemsize of 2

FIXED_UNIT_Z = 'nm'
FIXED_UNIT_XY = 'mm'

BLOCK_SIZE = 24
BLOCK_NAME_SIZE = 16
INT16_MAX = 32767


class BlockType(IntEnum):
    NONE = 0
    DIRECTORY = 1
    ARRAY = 3
    TEXT = 5
    SHORT = 6
    FLOAT = 7
    DOUBLE = 8
    LONG = 12


dtypes = {
    BlockType.ARRAY: {1: 'uint8', 2: 'int16', 4: 'float32'},
    BlockType.SHORT: 'h',
    BlockType.FLOAT: 'f',
    BlockType.DOUBLE: 'd',
    BlockType.LONG: 'l'
}

invalid_value = {'int16': 32767, 'float32': 1e38}


@dataclass
class Block:
    type: int
    size: int
    flags: int
    offset: int = None

    def _read_array(self, filehandle):
        nx, ny, itemsize = struct.unpack('<HHH', filehandle.read(6))
        data_length = nx * ny
        if data_length * itemsize != self.size - 6:
            raise CorruptedFileError(f'Size of data ({data_length}) does not match expected size ({self.size - 6}).')
        data = np.fromfile(filehandle, dtype=dtypes[BlockType.ARRAY][itemsize], count=data_length)
        data = np.rot90(data.reshape(nx, ny))
        return data

    def _read_text(self, filehandle, encoding='utf-8'):
        return filehandle.read(self.size).decode(encoding).rstrip('\x00')

    def _read_number(self, filehandle):
        dtype = dtypes[self.type]
        itemsize = struct.calcsize(dtype)
        return struct.unpack(f'{int(self.size / itemsize)}{dtype}', filehandle.read(self.size))[0]

    def read_contents(self, filehandle, encoding='utf-8'):
        current_pos = filehandle.tell()
        filehandle.seek(self.offset, 0)
        if self.type == BlockType.ARRAY:
            result = self._read_array(filehandle)
        elif self.type == BlockType.TEXT:
            result = self._read_text(filehandle, encoding=encoding)
        else:
            result = self._read_number(filehandle)
        filehandle.seek(current_pos, 0)
        return result


def read_block_definition(filehandle, encoding='utf-8'):
    name = filehandle.read(16).decode().rstrip('\x00')
    type_, size, flags = struct.unpack('<hlH', filehandle.read(8))
    return name, Block(BlockType(type_), size, flags)


def read_opd(filepath, read_image_layers=False, encoding='utf-8'):
    with open(filepath, 'rb') as filehandle:
        magic = filehandle.read(2)  # skipping header
        name, directory_block = read_block_definition(filehandle, encoding=encoding)
        if name != 'Directory':
            raise CorruptedFileError('Directory block not found.')
        n_blocks = int(directory_block.size / BLOCK_SIZE)
        blocks = dict()
        for _ in range(n_blocks - 1):
            name, block = read_block_definition(filehandle, encoding=encoding)
            blocks[name] = block
        offset = filehandle.tell()
        for block in blocks.values():
            block.offset = offset
            offset += block.size

        data = blocks['RAW_DATA'].read_contents(filehandle, encoding=encoding)
        image_layers = {}
        if read_image_layers and 'Image' in blocks:
            image_layers['Grayscale'] = blocks['Image'].read_contents(filehandle, encoding=encoding)

        metadata = dict()
        for name, block in blocks.items():
            if block.type in [BlockType.TEXT, BlockType.SHORT, BlockType.FLOAT, BlockType.DOUBLE, BlockType.LONG]:
                contents = block.read_contents(filehandle, encoding=encoding)
                if block.type == BlockType.TEXT and not contents:
                    # Skip empty strings
                    continue
                metadata[name] = contents

        metadata['timestamp'] = dateutil.parser.parse(metadata['Date'] + ' ' + metadata['Time'])
        del metadata['Date']
        del metadata['Time']

        for label in ['Wavelength', 'Mult', 'Aspect', 'Pixel_size']:
            if label not in metadata:
                metadata[label] = 1.0

        nan_mask = None
        # Mask invalid datapoints
        if data.dtype in ['int16', 'float32']:
            nan_mask = (data == invalid_value[data.dtype.name])

        metadata['Wavelength'] *= get_unit_conversion(FIXED_UNIT_Z, 'um')
        scale_z = metadata['Wavelength'] / metadata['Mult']

        data = data.astype('float64') * scale_z
        if nan_mask is not None:
            data[nan_mask] = np.nan

        step_x = metadata['Pixel_size'] * get_unit_conversion(FIXED_UNIT_XY, 'um')
        step_y = step_x * metadata['Aspect']
        return RawSurface(data, step_x, step_y, metadata=metadata, image_layers=image_layers)
