import struct

units = {
    'mm': 10**-3,
    'um': 10**-6,
    'nm': 10**-9,
    'pm': 10**-12
}

def get_unit_conversion(from_unit, to_unit):
    return units[from_unit] / units[to_unit]


def read_binary_layout(filehandle, layout, fast=True):
    result = dict()
    for name, format, skip_fast in layout:
        if name is None:
            filehandle.seek(format, 1)
            continue
        size = struct.calcsize(format)
        if fast and skip_fast:
            filehandle.seek(size, 1)
            continue
        unpacked_data = struct.unpack(f'{format}', filehandle.read(size))[0]
        if isinstance(unpacked_data, bytes):
            unpacked_data = unpacked_data.decode().strip()
        result[name] = unpacked_data
    return result

