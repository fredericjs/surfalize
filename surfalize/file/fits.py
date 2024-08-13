import numpy as np
from .common import RawSurface, get_unit_conversion
from ..exceptions import FileFormatError, UnsupportedFileFormatError

MAGIC = b'SIMPLE'
BLOCKSIZE = 2880

class HeaderDataUnit:

    def __init__(self, header, data):
        self.header = header
        self.data = data

dtype_map = {
    8: 'uint8',
    16: '>i2',
    32: '>i4',
    64: '>i8',
    -32: '>f4',
    -64: '>f8'
}

def convert_dtype(string):
    try:
        return int(string)
    except ValueError:
        pass
    try:
        return float(string)
    except ValueError:
        pass
    return string.strip("' ")


def parse_header_block(block):
    lines = [block[i * 80:(i + 1) * 80].rstrip() for i in range(int(BLOCKSIZE / 80))]
    header = dict()
    block_ended = False
    for line in lines:
        if not line or line.startswith('COMMENT'):
            continue
        if line == 'END':
            block_ended = True
            break
        if '/' in line:
            line = line.split('/')[0]
        left, right = line.split('=')
        left = left.strip()
        right = right.strip()
        if left.startswith('HIERARCH'):
            left = left.replace('HIERARCH ', '')
        header[left] = convert_dtype(right)
    return header, block_ended


def read_header(filehandle):
    has_ended = False
    header = dict()
    while not has_ended:
        block = filehandle.read(BLOCKSIZE)
        partial_header, has_ended = parse_header_block(block.decode())
        header.update(partial_header)
    return header

def read_fits(filepath, read_image_layers=False, encoding='utf-8'):
    filesize = filepath.stat().st_size
    with open(filepath, 'rb') as filehandle:
        magic = filehandle.read(len(MAGIC))
        if magic != MAGIC:
            raise FileFormatError("Invalid file magic detected.")

        filehandle.seek(0)
        header = read_header(filehandle)

        layers = dict()
        while True:
            block = filehandle.read(BLOCKSIZE)
            if block.startswith(b'XTENSION'):
                hdu_header, has_ended = parse_header_block(block.decode())
                if hdu_header['NAXIS'] != 2:
                    raise UnsupportedFileFormatError("Array with number of dimensions not equal to 2 detected.")
                datasize = hdu_header['NAXIS1'] * hdu_header['NAXIS2'] * int(abs(hdu_header['BITPIX'] / 8))
                data = np.fromfile(filehandle, dtype=dtype_map[hdu_header['BITPIX']],
                                   count=hdu_header['NAXIS1'] * hdu_header['NAXIS2']).reshape(hdu_header['NAXIS2'],
                                                                                              hdu_header['NAXIS1'])
                # Skip the remaining blocksize
                filehandle.seek(BLOCKSIZE - (datasize % BLOCKSIZE), 1)
                layers[hdu_header['EXTNAME']] = HeaderDataUnit(hdu_header, data)

            if filehandle.tell() >= filesize:
                break

    metadata = header
    image_layers = dict()

    if not 'HEIGHTS' in layers:
        raise FileFormatError("No height layer found!")

    data = layers['HEIGHTS'].data.astype('float64')
    if 'MASK' in layers:
        mask = layers['MASK'].data.astype('bool')
        data[~mask] = np.nan
    x_factor = header['UnitMultiplicatorDeltas'] * get_unit_conversion(header['UNITX'], 'um')
    y_factor = header['UnitMultiplicatorDeltas'] * get_unit_conversion(header['UNITX'], 'um')
    z_factor = header['UnitMultiplicatorHeights'] * get_unit_conversion(header['UNITZ'], 'um')
    step_x = header['DELTAX'] * x_factor
    step_y = header['DELTAY'] * y_factor
    data = data * z_factor

    if read_image_layers:
        for k, v in layers.items():
            if k not in ['HEIGHTS', 'MASK']:
                image_layers[k.capitalize()] = v.data

    return RawSurface(data, step_x, step_y, metadata=metadata, image_layers=image_layers)




