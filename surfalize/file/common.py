import struct

MU_ALIASES = {
    chr(181): 'u',
    chr(956): 'u',
    chr(13211): 'um'
}

UNIT_EXPONENT = {
    'mm': -3,
    'um': -6,
    'nm': -9,
    'pm': -12
}

def _sanitize_mu(string):
    """
    replaces all possible unicode versions of µm with um.

    Parameters
    ----------
    string: str
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
    from_unit: str
        Unit from which to convert.
    to_unit
        Unit to which to convert.

    Returns
    -------
    factor: float
        Factor by which to multiply the original values.
    """
    from_unit = _sanitize_mu(from_unit)
    if from_unit not in UNIT_EXPONENT or to_unit not in UNIT_EXPONENT:
        raise ValueError('Unit does not exist.')
    exponent = UNIT_EXPONENT[from_unit] - UNIT_EXPONENT[to_unit]
    return 10**exponent

def write_binary_layout(filehandle, layout, data, encoding='utf-8'):
    """
    Writes a binary layout to a file.

    Parameters
    ----------
    filehandle: file object
        File-like object to read the data from.
    layout: tuple[tuple[str, str, bool] | tuple[None, int, None]]
        Layout of the bytes to read as a tuple of tuples in the form (<name>, <format>, <skip_fast>) or
        (None, <n_bytes>, None) for reserved bytes.
    data: dict[str: any]
        Dictionary containing keys that correspond to the name value in the layout tuple and the values to write
        to the file as keys.
    encoding: str, Default utf-8
            Encoding of characters in the file. Defaults to utf-8.

    Returns
    -------
    None
    """
    for name, format_, _ in layout:
        if name is None:
            filehandle.write(b'\x00' * format_)
            continue
        value = data[name]
        if isinstance(value, str):
            value = value.encode(encoding)
        filehandle.write(struct.pack(format_, value))

def read_binary_layout(filehandle, layout, fast=True, encoding='utf-8'):
    """
    Reads a binary layout specified by a tuple of tuples from an opened file and returns a dict with the read values.
    The layout must be provided in the form:

    LAYOUT = (
        (<name>, <format_specifier>, <skip_fast>),
        (...),
        ...
    )

    Each tuple in the layout contains three values. The first is a name that will be used as a key for the returned
    dictionary. The second value is a format specified according to the struct module. The last value is a flag that
    indicates whether the value can be skipped for fast reading mode.

    Reserved bytes in the layout should be indicated by specifying None for the name and the number of bytes to skip as
    an int for the format specified, e.g. (None, <n_bytes: int>, None).

    Parameters
    ----------
    filehandle: file object
        File-like object to read the data from.
    layout: tuple[tuple[str, str, bool] | tuple[None, int, None]]
        Layout of the bytes to read as a tuple of tuples in the form (<name>, <format>, <skip_fast>) or
        (None, <n_bytes>, None) for reserved bytes.
    fast: bool, default True
        If true, skips all values that are marked for skipping in fast mode.
    encoding: str, Default utf-8
            Encoding of characters in the file. Defaults to utf-8.

    Returns
    -------
    dict[str: any]
    """
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
            unpacked_data = unpacked_data.decode(encoding).rstrip(' \x00')
        result[name] = unpacked_data
    return result

