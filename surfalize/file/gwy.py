import struct
import numpy as np

from .common import get_unit_conversion, RawSurface
from ..exceptions import FileFormatError, UnsupportedFileFormatError

MAGIC = b'GWYP'
# This is not specified by the file standard but we nonetheless assume that the name will never be longer than that
STR_MAX_SIZE = 4096


def read_null_terminated_string(filehandle, maxsize=STR_MAX_SIZE):
    """
    Reads a null-terminated string from a Gwyddion file. Stops on a null character or when maxsize is reached.

    Parameters
    ----------
    filehandle
        Handle to the opened file. Is assumed to be at the correct position.
    maxsize: int
        Maximum size of the string to read.

    Returns
    -------
    Decoded string
    """
    string = b''
    i = 0
    while i < maxsize:
        char = filehandle.read(1)
        if char == b'\x00':
            break
        string += char
        i += 1
    return string.decode('utf-8')


class Container:

    def __init__(self, filehandle):
        self.filehandle = filehandle
        self.name = read_null_terminated_string(filehandle)
        self.size = struct.unpack('I', filehandle.read(4))[0]

    def __repr__(self):
        name = self.name
        size = self.size
        return f'{self.__class__.__name__}({name=}, {size=})'

    def read_contents(self):
        pos = self.filehandle.tell()
        components = {}
        while self.filehandle.tell() < pos + self.size:
            component = Component(self.filehandle)
            components[component.name] = component.read_contents()
        return components


class Component:

    def __init__(self, filehandle):
        self.filehandle = filehandle
        self.name = read_null_terminated_string(filehandle)
        datatype = filehandle.read(1).decode('utf-8')
        self.is_array = datatype.isupper()
        self.datatype = datatype.lower()

    def __repr__(self):
        name = self.name
        datatype = self.datatype
        return f'{self.__class__.__name__}({name=}, {datatype=})'

    def read_contents(self):
        if self.is_array:
            return self._read_array(self.filehandle)
        return self._read_atomic(self.filehandle)

    def _read_array(self, filehandle):
        array_size = struct.unpack('I', filehandle.read(4))[0]
        if self.datatype == 'o':
            return [Container(filehandle).read_contents() for _ in range(array_size)]
        elif self.datatype == 's':
            return [read_null_terminated_string(filehandle) for _ in range(array_size)]
        return np.fromfile(filehandle, count=array_size, dtype=self.datatype)

    def _read_atomic(self, filehandle):
        if self.datatype == 'o':
            return Container(filehandle).read_contents()
        elif self.datatype == 's':
            return read_null_terminated_string(filehandle)
        result = struct.unpack(f'{self.datatype}', filehandle.read(struct.calcsize(self.datatype)))[0]
        if self.datatype == 'b':
            # Gwyddion docs state that all non-zero values are to be interpreted as true
            return result != 0
        return result


def parse_gwy_tree(filehandle):
    container = Container(filehandle)
    return container.read_contents()

def read_gwy(filepath, read_image_layers=False, encoding='utf-8'):
    with open(filepath, 'rb') as filehandle:
        if filehandle.read(4) != MAGIC:
            raise FileFormatError('Unknown file magic detected.')

        tree = parse_gwy_tree(filehandle)

        if '/0/data' not in tree:
            raise UnsupportedFileFormatError('No height data section found.')

        data = tree['/0/data']['data']
        unit_xy = tree['/0/data']['si_unit_xy']['unitstr']
        unit_z = tree['/0/data']['si_unit_z']['unitstr']
        xreal = tree['/0/data']['xreal']
        yreal = tree['/0/data']['yreal']
        nx = tree['/0/data']['xres']
        ny = tree['/0/data']['yres']

        xy_conversion_factor = get_unit_conversion(unit_xy, 'um')
        step_x = xreal / nx * xy_conversion_factor
        step_y = yreal / ny * xy_conversion_factor

        data = data * get_unit_conversion(unit_z, 'um')

        if '/0/mask' in tree:
            mask = tree['/0/mask']['data'].astype('bool')
            data[mask] = np.nan

        data = data.reshape(ny, nx)

        metadata = {}
        if '/0/meta' in tree:
            metadata.update(tree['/0/meta'])

        # Todo: Read image data
        # Todo: Don't assume the height data is in /0/data, but could be anywhere else. Check for presence of z-unit

        return RawSurface(data, step_x, step_y, metadata=metadata)



