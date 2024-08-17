import zlib
import lzma
import numpy as np
from surfalize.file.common import FormatFromPrevious, read_binary_layout, RawSurface, write_binary_layout

MAGIC = b'SFLZ'

COMPRESSION_TYPE_FROM_INT = {
    0: 'none',
    1: 'zlib',
    2: 'lzma'
}

INT_FROM_COMPRESSION_TYPE = {v: k for k, v in COMPRESSION_TYPE_FROM_INT.items()}

class ConvertCompression(Apply):



LAYOUT_HEADER = (
    ('version', '10s'),
    ('step_x', 'd'),
    ('step_y', 'd'),
    ('num_layers', 'H'),
    ('num_metadata', 'H'),
    ('compression_algorithm', 'B'),
)

LAYOUT_LAYER_HEADER = (
    ('name_length', 'H'),
    ('name', FormatFromPrevious('name_length', 's')),
    ('width', 'I'),
    ('height', 'I'),
    ('channels', 'B'),
    ('datatype', '3s'),
    ('size', 'I'),
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
        header = read_binary_layout(filehandle, LAYOUT_HEADER)
        compression_algorithm = COMPRESSION_TYPE_FROM_INT[header['compression_algorithm']]
        image_layers = dict()
        height_data = None
        read_n_layers = header['num_layers'] if read_image_layers else 1
        for i in range(read_n_layers):
            layer_header = read_binary_layout(filehandle, LAYOUT_LAYER_HEADER)
            compressed_data = filehandle.read(layer_header['size'])
            data = decompress(compressed_data, compression_algorithm)
            data = np.frombuffer(data, dtype=layer_header['datatype'])
            if layer_header['channels'] > 1:
                data = data.reshape(layer_header['height'], layer_header['width'], layer_header['channels'])
            else:
                data = data.reshape(layer_header['height'], layer_header['width'])
            if i == 0 and layer_header['name'] == 'topography':
                height_data = data
            else:
                image_layers[layer_header['name']] = data
        return RawSurface(height_data, header['step_x'], header['step_y'], image_layers=image_layers)


def write_sflz(filepath, surface, encoding='utf-8', compression_type='zlib', save_image_layers=True,
               write_metadata=True):
    with open('file.sflz', 'wb') as filehandle:
        filehandle.write(MAGIC)

        header = {
            'version': '1.0'.ljust(10),
            'step_x': surface.step_x,
            'step_y': surface.step_y,
            'num_layers': 1 + len(surface.image_layers) if save_image_layers else 1,
            'num_metadata': len(surface.metadata) if write_metadata else 0,
            'compression_algorithm': INT_FROM_COMPRESSION_TYPE[compression_type]
        }

        write_binary_layout(filehandle, LAYOUT_HEADER, header)
        data = compress(surface.data.tobytes(), compression_type)

        layer_header = {
            'name_length': len('topography'),
            'name': 'topography',
            'width': surface.size.x,
            'height': surface.size.y,
            'channels': 1,
            'datatype': surface.data.dtype.str,
            'size': len(data)
        }

        write_binary_layout(filehandle, LAYOUT_LAYER_HEADER, layer_header)
        filehandle.write(data)
        if save_image_layers and surface.image_layers:
            for name, layer in surface.image_layers.items():
                data = compress(layer.data.tobytes(), compression_type)

                layer_header = {
                    'name_length': len(name),
                    'name': name,
                    'width': layer.data.shape[1],
                    'height': layer.data.shape[0],
                    'channels': 1 if layer.data.ndim == 2 else layer.data.shape[-1],
                    'datatype': layer.data.dtype.str,
                    'size': len(data)
                }

                write_binary_layout(filehandle, LAYOUT_LAYER_HEADER, layer_header)
                filehandle.write(data)
