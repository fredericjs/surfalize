import zlib
import lzma
import numpy as np
from surfalize.file.common import FormatFromPrevious, RawSurface, Apply, Entry, Layout

MAGIC = b'SFLZ'

COMPRESSION_TYPE_FROM_INT = {
    0: 'none',
    1: 'zlib',
    2: 'lzma'
}

INT_FROM_COMPRESSION_TYPE = {v: k for k, v in COMPRESSION_TYPE_FROM_INT.items()}

class ConvertCompression(Apply):

    def read(self, data):
        return COMPRESSION_TYPE_FROM_INT[data]

    def write(self, data):
        return INT_FROM_COMPRESSION_TYPE[data]


LAYOUT_HEADER = Layout(
    Entry('version', '10s'),
    Entry('step_x', 'd'),
    Entry('step_y', 'd'),
    Entry('scaled', '?'),
    Entry('min_value', 'd'),
    Entry('max_value', 'd'),
    Entry('num_layers', 'H'),
    Entry('num_metadata', 'H'),
    Entry('compression_algorithm', ConvertCompression('B'))
)

LAYOUT_LAYER_HEADER = Layout(
    Entry('name_length', 'H'),
    Entry('name', FormatFromPrevious('name_length', 's')),
    Entry('width', 'I'),
    Entry('height', 'I'),
    Entry('channels', 'B'),
    Entry('datatype', '3s'),
    Entry('size', 'I'),
)

def compress(data, compression_type):
    if compression_type == 'none':
        return data
    elif compression_type == 'zlib':
        return zlib.compress(data)
    elif compression_type == 'lzma':
        return lzma.compress(data)

def decompress(data, compression_type):
    if compression_type == 'none':
        return data
    elif compression_type == 'zlib':
        return zlib.decompress(data)
    elif compression_type == 'lzma':
        return lzma.decompress(data)

def read_sflz(filepath, encoding='utf-8', read_image_layers=True):
    with open(filepath, 'rb') as filehandle:
        magic = filehandle.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError
        header = LAYOUT_HEADER.read(filehandle)
        compression_algorithm = header['compression_algorithm']
        image_layers = dict()
        height_data = None
        read_n_layers = header['num_layers'] if read_image_layers else 1
        for i in range(read_n_layers):
            layer_header = LAYOUT_LAYER_HEADER.read(filehandle)
            compressed_data = filehandle.read(layer_header['size'])
            data = decompress(compressed_data, compression_algorithm)
            data = np.frombuffer(data, dtype=layer_header['datatype'])
            if layer_header['channels'] > 1:
                data = data.reshape(layer_header['height'], layer_header['width'], layer_header['channels'])
            else:
                data = data.reshape(layer_header['height'], layer_header['width'])
            if i == 0 and layer_header['name'] == 'topography':
                height_data = data
                if header['scaled']:
                    min_val = header['min_value']
                    max_val = header['max_value']
                    dtype_min = np.iinfo(layer_header['datatype']).min
                    dtype_max = np.iinfo(layer_header['datatype']).max
                    height_data = (height_data.astype('float64') + dtype_min) / (dtype_max - dtype_min)
                    height_data = height_data * (max_val - min_val) + min_val
            else:
                image_layers[layer_header['name']] = data
        return RawSurface(height_data, header['step_x'], header['step_y'], image_layers=image_layers)


def write_sflz(filepath, surface, encoding='utf-8', compression='zlib', save_image_layers=True,
               write_metadata=True, dtype='<u4'):
    with open(filepath, 'wb') as filehandle:
        filehandle.write(MAGIC)

        header = {
            'version': '1.0'.ljust(10),
            'step_x': surface.step_x,
            'step_y': surface.step_y,
            'num_layers': 1 + len(surface.image_layers) if save_image_layers else 1,
            'num_metadata': len(surface.metadata) if write_metadata else 0,
            'compression_algorithm': compression
        }

        dtype = np.dtype(dtype)
        data = surface.data
        if dtype.kind in ['i', 'u']:
            # perform scaling
            min_val = data.min()
            max_val = data.max()
            header['scaled'] = True
            header['min_value'] = min_val
            header['max_value'] = max_val
            dtype_min = np.iinfo(dtype).min
            dtype_max = np.iinfo(dtype).max
            data_norm = (data - min_val) / (max_val - min_val)
            data = data_norm * (dtype_max - dtype_min) + dtype_min
        else:
            header['scaled'] = False
            header['min_value'] = 0
            header['max_value'] = 0

        LAYOUT_HEADER.write(filehandle, header)
        data = compress(data.astype(dtype).tobytes(), compression)

        layer_header = {
            'name_length': len('topography'),
            'name': 'topography',
            'width': surface.size.x,
            'height': surface.size.y,
            'channels': 1,
            'datatype': dtype.str,
            'size': len(data)
        }

        LAYOUT_LAYER_HEADER.write(filehandle, layer_header)
        filehandle.write(data)
        if save_image_layers and surface.image_layers:
            for name, layer in surface.image_layers.items():
                data = compress(layer.data.tobytes(), compression)

                layer_header = {
                    'name_length': len(name),
                    'name': name,
                    'width': layer.data.shape[1],
                    'height': layer.data.shape[0],
                    'channels': 1 if layer.data.ndim == 2 else layer.data.shape[-1],
                    'datatype': layer.data.dtype.str,
                    'size': len(data)
                }

                LAYOUT_LAYER_HEADER.write(filehandle, layer_header)
                filehandle.write(data)
