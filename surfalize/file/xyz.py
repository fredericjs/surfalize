# This code assumes units of meters for xyz data
import numpy as np
from ..exceptions import UnsupportedFileFormatError, CorruptedFileError
from .common import RawSurface

def read_xyz(filepath, read_image_layers=False, encoding='utf-8'):
    with open(filepath) as file:
        try:
            raw_data = np.loadtxt(file)
        except UnicodeDecodeError:
            raise UnsupportedFileFormatError('The xyz file contains binary data. Only ASCII xyz files are supported.')
        except ValueError:
            raise UnsupportedFileFormatError('The xyz file format type is not supported.')
    x = np.unique(raw_data[:,0])
    y = np.unique(raw_data[:,1])
    nx = x.size
    ny = y.size
    if nx * ny != raw_data.shape[0]:
        raise CorruptedFileError('Number of datapoints does not match expected size.')
    step_x = (x.max() - x.min()) / (nx) * 10**6
    step_y = (y.max() - y.min()) / (ny) * 10**6
    data = raw_data[:,2].copy().reshape(ny, nx) * 10**6

    return RawSurface(data, step_x, step_y)