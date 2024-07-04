# Information on the file format taken from here: https://digitalmetrology.com/omnisurf3d-file-format/

import struct
import io
import dateutil
from PIL import Image
import numpy as np
from .common import read_binary_layout, FormatFromPrevious, RawSurface
from ..exceptions import CorruptedFileError

MAGIC = b'OmniSurf3D'
INVALID_VALUE = -3.4028235e38
THRESHOLD = -3e38

LAYOUT_HEADER = (
    ('nMajorVersion', 'i'),
    ('nMinorVersion', 'i'),
    ('nIdentificationStringLength', 'i'),
    ('chArrayIdentification', FormatFromPrevious('nIdentificationStringLength', 's')),
    ('nMeasureDateTimeStringLength', 'i'),
    ('chArrayMeasureDateTime', FormatFromPrevious('nMeasureDateTimeStringLength', 's')),
    ('nPointsAlongX', 'i'),
    ('nPointsAlongY', 'i'),
    ('dSpacingAlongXUM', 'd'),
    ('dSpacingAlongYUM', 'd'),
    ('dXOriginUM', 'd'),
    ('dYOriginUM', 'd'),
)

def read_os3d(filepath, read_image_layers=False, encoding='utf-8'):
    with open(filepath, 'rb') as filehandle:
        magic = filehandle.read(len(MAGIC))
        if magic != MAGIC:
            raise CorruptedFileError(f'Unknown file magic detected: {magic.decode()}')
        header = read_binary_layout(filehandle, LAYOUT_HEADER, encoding=encoding)
        data = np.fromfile(filehandle, count=header['nPointsAlongX'] * header['nPointsAlongY'], dtype='float32')
        data = data.reshape(header['nPointsAlongY'], header['nPointsAlongX'])
        data[data < THRESHOLD] = np.nan
        step_x = header['dSpacingAlongXUM']
        step_y = header['dSpacingAlongYUM']

        metadata = header
        metadata['timestamp'] = dateutil.parser.parse(header['chArrayMeasureDateTime'])

        has_image = struct.unpack('b', filehandle.read(1))
        image_layers = {}
        if has_image and read_image_layers:
            img_data = io.BytesIO(filehandle.read())
            image_layers['RGBA'] = np.array(Image.open(img_data))
    return RawSurface(data, step_x, step_y, image_layers=image_layers, metadata=metadata)