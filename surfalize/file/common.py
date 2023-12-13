import struct
from abc import ABC, abstractmethod
from collections.abc import MutableMapping

def read_binary_layout(filehandle, layout):
    result = dict()
    for name, format, skip_fast in layout:
        size = struct.calcsize(format)
        unpacked_data = struct.unpack(f'{format}', filehandle.read(size))[0]
        if isinstance(unpacked_data, bytes):
            unpacked_data = unpacked_data.decode().strip()
        result[name] = unpacked_data
    return result


class BinaryLayout(MutableMapping):

    def __init__(self, fields, format):
        self._format = format
        self._fields = dict()

    def __getattr__(self, item):
        return self._fields[item]

    def read(self, filehandle):
        for field in self._format:
            data = field.read()
            if data is None:
                continue
            self._fields[field.name] = data

class BaseReader(ABC):

    def __init__(self, filepath):
        self._filepath = filepath

    @abstractmethod
    def read_data(self):
        pass

    @abstractmethod
    def read(self):
        pass


