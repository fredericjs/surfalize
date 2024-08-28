import struct
import numpy as np
from abc import abstractmethod, ABC

MU_ALIASES = {
    chr(181): 'u',
    chr(956): 'u',
    chr(13211): 'um'
}

UNIT_EXPONENT = {
    'm':   0,
    'meter': 0,
    'dm': -1,
    'decimeter': -1,
    'cm': -2,
    'centimeter': -2,
    'mm': -3,
    'millimeter': -3,
    'um': -6,
    'micrometer': -6,
    'nm': -9,
    'nanometer': -9,
    'pm': -12,
    'picometer': -12
}

def _sanitize_mu(string):
    """
    replaces all possible unicode versions of µm with um.

    Parameters
    ----------
    string : str
        Input string.

    Returns
    -------
    str
    """
    for alias, replacement in MU_ALIASES.items():
        string = string.replace(alias, replacement)
    return string


def get_unit_conversion(from_unit, to_unit):
    """
    Calculates unit conversion factor.

    Parameters
    ----------
    from_unit : str
        Unit from which to convert.
    to_unit : str
        Unit to which to convert.

    Returns
    -------
    factor : float
        Factor by which to multiply the original values.
    """
    from_unit = _sanitize_mu(from_unit)
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()
    if from_unit not in UNIT_EXPONENT or to_unit not in UNIT_EXPONENT:
        raise ValueError('Unit does not exist.')
    exponent = UNIT_EXPONENT[from_unit] - UNIT_EXPONENT[to_unit]
    return 10**exponent

class FormatFromPrevious:

    def __init__(self, previous, dtype):
        self.previous = previous
        self.dtype = dtype

    def get_format(self, layout_dict):
        size = layout_dict[self.previous]
        return f'{size}{self.dtype}'

class Apply:

    def __init__(self, dtype):
        self.dtype = dtype

    @abstractmethod
    def read(self, data):
        raise NotImplementedError

    @abstractmethod
    def write(self, data):
        raise NotImplementedError

    def convert_from_file(self, filehandle):
        size = struct.calcsize(self.dtype)
        unpacked_data = struct.unpack(f'{self.dtype}', filehandle.read(size))[0]
        return self.read(unpacked_data)

    def convert_to_file(self, filehandle, value):
        size = struct.calcsize(self.dtype)
        filehandle.write(struct.pack(self.dtype, self.write(value)))

class BaseEntry(ABC):

    @abstractmethod
    def read(self, filehandle, data, encoding):
        raise NotImplementedError

    @abstractmethod
    def write(self, filehandle, data, encoding):
        raise NotImplementedError

class Reserved(BaseEntry):

    def __init__(self, nbytes):
        self.nbytes = nbytes

    def read(self, filehandle, data, encoding):
        filehandle.seek(self.nbytes, 1)

    def write(self, filehandle, data, encoding):
        filehandle.write(b'\x00' * self.nbytes)


class Entry(BaseEntry):

    def __init__(self, name, format):
        self.name = name
        self.format = format

    def write(self, filehandle, data, encoding):
        if isinstance(self.format, Apply):
            self.format.convert_to_file(filehandle, data[self.name])
            return
        if isinstance(self.format, FormatFromPrevious):
            format = self.format.get_format(data)
        else:
            format = self.format
        value = data[self.name]
        if isinstance(value, str):
            # Pad all strings with spaces
            length = struct.calcsize(format)
            value = value.ljust(length).encode(encoding)
        filehandle.write(struct.pack(format, value))

    def read(self, filehandle, data, encoding):
        if isinstance(self.format, Apply):
            data[self.name] = self.format.convert_from_file(filehandle)
            return
        if isinstance(self.format, FormatFromPrevious):
            format = self.format.get_format(data)
        else:
            format = self.format
        size = struct.calcsize(format)
        unpacked_data = struct.unpack(f'{format}', filehandle.read(size))[0]
        if isinstance(unpacked_data, bytes):
            unpacked_data = unpacked_data.decode(encoding).rstrip(' \x00')
        data[self.name] = unpacked_data


class Layout:

    def __init__(self, *entries):
        self._entries = entries

    def write(self, filehandle, data, encoding='utf-8'):
        """
        Writes a binary layout to a file.

        Parameters
        ----------
        data : dict[str: any]
            Dictionary containing keys that correspond to the name value in the layout tuple and the values to write
            to the file as keys.
        encoding : str, Default utf-8
                Encoding of characters in the file. Defaults to utf-8.

        Returns
        -------
        None
        """
        for entry in self._entries:
            entry.write(filehandle, data, encoding)

    def read(self, filehandle, encoding='utf-8'):
        """
        Reads a binary layout specified by a tuple of tuples from an opened file and returns a dict with the read values.
        The layout must be provided in the form:

        LAYOUT = (
            (<name>, <format_specifier>),
            (...),
            ...
        )

        Each tuple in the layout contains three values. The first is a name that will be used as a key for the returned
        dictionary. The second value is a format specified according to the struct module.

        Reserved bytes in the layout should be indicated by specifying None for the name and the number of bytes to skip as
        an int for the format specified, e.g. (None, <n_bytes: int>).

        Parameters
        ----------
        filehandle : file object
            File-like object to read the data from.
        layout : tuple[tuple[str, str, bool] | tuple[None, int, None]]
            Layout of the bytes to read as a tuple of tuples in the form (<name>, <format>, <skip_fast>) or
            (None, <n_bytes>, None) for reserved bytes.
        encoding : str, Default utf-8
                Encoding of characters in the file. Defaults to utf-8.

        Returns
        -------
        dict[str: any]
        """
        data = dict()
        for entry in self._entries:
            entry.read(filehandle, data, encoding)
        return data

def np_fromany(fileobject, dtype, count=-1, offset=0):
    """
    Function that invokes either np.frombuffer or np.fromfile depending on whether the object is a file-like object
    or a buffer.

    Parameters
    ----------
    fileobject : buffer_like or file-like or str or Path
        An object that exposes the buffer interface or a file-like object or a str or Path representing a filepath.
    dtype : data-type
        Data-type of the returned array.
    count : int, Default -1.
        Number of items to read. -1 means all data in the buffer or file.
    offset : int
        Start reading the buffer from this offset (in bytes); default: 0.

    Returns
    -------
    np.ndarray
    """
    try:
        return np.frombuffer(fileobject, dtype, count=count, offset=offset)
    except TypeError:
        if offset > 0:
            fileobject.seek(offset, 1)
        if count == -1:
            buffer = fileobject.read()
        else:
            buffer = fileobject.read(count * np.dtype(dtype).itemsize)
        return np.frombuffer(buffer, dtype)


class RawSurface:

    def __init__(self, data: np.ndarray, step_x: float, step_y: float, metadata: dict=None,
                 image_layers: dict=None):
        self.data = data
        self.step_x = step_x
        self.step_y = step_y
        self.metadata = {} if metadata is None else metadata
        self.image_layers = {} if image_layers is None else image_layers
