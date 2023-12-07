import os
import struct
import numpy as np

from ..exceptions import FileFormatError, CorruptedFileError

MAGIC_NUMBER = 'DIGITAL SURF'

FORMAT_SUR = (
    ('Code', 's', 12),
    ('Format', 'h', 2),
    ('Number of objects', 'h', 2),
    ('Version number', 'h', 2),
    ('Studiable type', 'h', 2),
    ('Name object', 's', 30),
    ('Name operator', 's', 30),
    (None, None, 6), # Reserved
    ('Non measured points', 'h', 2),
    ('Absolute Z axis', 'h', 2),
    (None, None, 8), # Reserved
    ('Nb of bits of points', 'h', 2),
    ('Minimum point', 'i', 4),
    ('Maximum point', 'i', 4),
    ('Nb of points of a line', 'i', 4),
    ('Nb of lines', 'i', 4),
    ('Total nb of points', 'i', 4),
    ('Spacing in X', 'f', 4),
    ('Spacing in y', 'f', 4),
    ('Spacing in Z', 'f', 4),
    ('Name of X axis', 's', 16),
    ('Name of Y axis', 's', 16),
    ('Name of Z axis', 's', 16),
    ('Unit of step in X', 's', 16),
    ('Unit of step in Y', 's', 16),
    ('Unit of step in Z', 's', 16),
    ('Length unit of X axis', 's', 16),
    ('Length unit of Y axis', 's', 16),
    ('Length unit of Z axis', 's', 16),
    ('Unit ratio in X', 'f', 4),
    ('Unit ratio in Y', 'f', 4),
    ('Unit ratio in Z', 'f', 4),
    ('Replica', 'h', 2),
    ('Inverted', 'h', 2),
    ('Leveled', 'h', 2),
    (None, None, 12), # Reserved
    ('Seconds', 'h', 2),
    ('Minutes', 'h', 2),
    ('Hours', 'h', 2),
    ('Day', 'h', 2),
    ('Month', 'h', 2),
    ('Year', 'h', 2),
    ('Week day', 'h', 2),
    ('Measurement duration', 'f', 4),
    (None, None, 10), # Reserved
    ('Length of comment zone', 'h', 2),
    ('Length of private zone', 'h', 2),
    ('Client zone', 's', 128),
    ('Offset in X', 'f', 4),
    ('Offset in Y', 'f', 4),
    ('Offset in Z', 'f', 4),
    ('Spacing in T', 'f', 4),
    ('Offset in T', 'f', 4),
    ('Name of T axis', 's', 13),
    ('Unit of step in T', 's', 13)
)

class SurfFileReader:

    def __init__(self, filepath, format_specification):
        self.filepath = filepath
        self.filesize = os.path.getsize(filepath)
        self.format_specification = format_specification

    def _read_header(self, filehandle):
        header = dict()
        for field, type_, size in self.format_specification:
            if field is None:
                filehandle.seek(size, 1)  # 1 means relative to current position of file pointer
                continue
            n = int(size / struct.calcsize(type_))
            unpacked_data = struct.unpack(f'{n}{type_}', file.read(size))[0]
            if isinstance(unpacked_data, bytes):
                unpacked_data = unpacked_data.decode().strip()
            header[filed] = unpacked_data
        print(filehandle.tell())

    def _read_data(self, filehandle):
        filehandle.seek(self.header['Length of comment zone'], 1)
        filehandle.seek(self.header['Length of private zone'], 1)
        pointsize_bits = self.header['Nb of bits of points']
        if pointsize_bits == 8:
            type_ = 'e'
        if pointsize_bits == 16:
            type_ = 'f'
        elif pointsize_bits == 32:
            type_ = 'd'
        else:
            raise FileFormatError('Invalid point size detected.')
        data_size = self.header['Total nb of points'] * struct.calcsize(type_)
        print(data_size)
        data = filehandle.read(data_size)
        print(len(data))
        data = np.array(struct.unpack(f"<{self.header['Total nb of points']}{type_}", data))
        data = data.reshape(self.header['Nb of lines'], self.header['Nb of points of a line'])
        return data

    def read_file(self):
        with open(self.filepath, 'rb') as filehandle:
            self._read_header(filehandle)
            return self._read_data(filehandle)