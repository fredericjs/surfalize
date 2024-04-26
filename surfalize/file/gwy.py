import re
import struct
import numpy as np

from .common import get_unit_conversion, RawSurface, UNIT_EXPONENT
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


def _filter_candidates_by_z_unit_presence(tree, candidates):
    reduced_candidates = []
    for layer_key in candidates:
        if 'si_unit_z' in tree['GwyContainer'][f'/{layer_key}/data']['GwyDataField']:
            unit = tree['GwyContainer'][f'/{layer_key}/data']['GwyDataField']['si_unit_z']['GwySIUnit']['unitstr']
            if unit in UNIT_EXPONENT.keys():
                reduced_candidates.append(layer_key)
    return reduced_candidates

def _filter_candidates_by_name(tree, candidates, guesses=('height', 'topo')):
    """
    Filter layers by whether their title contains one of the guesses. If no candidate title contains any of the guesses,
    the original candidate list is returned.

    Parameters
    ----------
    tree: dict
        Dictionary representing the Gwyddion file tree.
    candidates: list[str]
        List of image layer candidates represented by their number string.
    guesses: tuple[str]
        Tuple of guesses.

    Returns
    -------
    Filtered candidates: List[str]
        List of candidates that contain any of the guesses in their title. If no candidate fulfills this requirement,
        the original list is returned.
    """
    reduced_candidates = []
    for layer_key in candidates:
        title = tree['GwyContainer'][f'/{layer_key}/data/title'].lower()
        if any([guess in title for guess in guesses]):
            reduced_candidates.append(layer_key)
    if not reduced_candidates:
        return candidates
    return reduced_candidates


def _filter_candidates_by_number(tree, candidates):
    """
    Returns the candidate with the lowest number. The candidate is returned inside a list of length 1 to conform to
    the return value signature of the other filter functions.

    Parameters
    ----------
    tree: dict
        Dictionary representing the Gwyddion file tree.
    candidates: list[str]
        List of image layer candidates represented by their number string.

    Returns
    -------
    Filtered candidates: List[str]
        List of the candidate with the lowest number.
    """
    return [str(sorted([int(layer_key) for layer_key in candidates])[0])]


_filters = [
    _filter_candidates_by_z_unit_presence,
    _filter_candidates_by_name,
    _filter_candidates_by_number
]


def guess_height_channel(tree, layer_candidates):
    for filter_ in _filters:
        layer_candidates = filter_(tree, layer_candidates)
        if len(layer_candidates) == 0:
            raise UnsupportedFileFormatError('The file does not seem to contain height data.')
        elif len(layer_candidates) == 1:
            return layer_candidates[0]
    raise UnsupportedFileFormatError('The height data channel could not be detected.')

def guess_image_channels(tree, layer_candidates):
    image_channels = []
    for layer_key in layer_candidates:
        if 'si_unit_z' not in tree['GwyContainer'][f'/{layer_key}/data']['GwyDataField']:
            image_channels.append(layer_key)
    return image_channels


def get_image_related_layer_keys(tree):
    image_related_layers = []
    for key in tree['GwyContainer']:
        mo = re.search(r'/(\d+)/', key)
        if mo:
            image_related_layers.append(mo.group(1))
    return list(set(image_related_layers))


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
        return {self.name: components}


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


# We run into a problem here: Gwyddion has no concept of a pimary height data layer. Contrary to profilometer
# file formats, the gwy format allows for an arbitrary number of channels, which could theoretically represent
# any possible physical or nonphysical quantity. Moreover, the layer numbers are arbitrary according to the
# documentation. To make this compatible with the concept applied in surfalize, where each Surface object is
# only ever associated with one single height layer and possible additional image layers without physical
# units, we need to guess the image layer in the gwy file that most likely corresponds to the height layer.
# We do this by employing different heuristics until only one layer emerges:
# 1. Select all layers that start with /n/, which indicate image related layers.
# 2. Select only layers which have an associated z-unit. E.g. RGB or intensity images have no encoded z-unit.
# 3. Select only layers which have a known length unit for the z-axis. Layers with units such as µm² are discarded.
# 4. Select only layers which have a title that indicates topographic data. This is purly guesswork.
# 5. Select the layer with the lowest number, which is oftentimes indeed the height data.

# E.g. if a gwy file is saved from a Gwyddion session, where multiple panels have been generated, such as DFTs,
# they will also be present as layers in the file, possibly even have a length unit for the z-axis. Such layers
# can only be disqualified using heuristic 4 or 5.
def read_gwy(filepath, read_image_layers=False, encoding='utf-8'):
    with open(filepath, 'rb') as filehandle:
        if filehandle.read(4) != MAGIC:
            raise FileFormatError('Unknown file magic detected.')

        tree = parse_gwy_tree(filehandle)
        # Image related layers refers to all layers / channels that contain 2d data that can be represented as an image
        # This also encompasses height data, or DFT, etc.
        image_related_layers = get_image_related_layer_keys(tree)
        height_layer_key = guess_height_channel(tree, image_related_layers)

        datafield = tree['GwyContainer'][f'/{height_layer_key}/data']['GwyDataField']

        data = datafield['data']
        unit_xy = datafield['si_unit_xy']['GwySIUnit']['unitstr']
        unit_z = datafield['si_unit_z']['GwySIUnit']['unitstr']
        xreal = datafield['xreal']
        yreal = datafield['yreal']
        nx = datafield['xres']
        ny = datafield['yres']

        xy_conversion_factor = get_unit_conversion(unit_xy, 'um')
        step_x = xreal / nx * xy_conversion_factor
        step_y = yreal / ny * xy_conversion_factor

        data = data * get_unit_conversion(unit_z, 'um')

        if f'/{height_layer_key}/mask' in tree['GwyContainer']:
            mask = tree['GwyContainer'][f'/{height_layer_key}/mask']['GwyDataField']['data'].astype('bool')
            data[mask] = np.nan

        data = data.reshape(ny, nx)

        metadata = {}
        if f'/{height_layer_key}/meta' in tree['GwyContainer']:
            metadata.update(tree['GwyContainer'][f'/{height_layer_key}/meta']['GwyContainer'])

        image_layers = {}
        if read_image_layers:
            image_channel_keys = guess_image_channels(tree, image_related_layers)
            for layer_key in image_channel_keys:
                datafield = tree['GwyContainer'][f'/{layer_key}/data']['GwyDataField']
                title = tree['GwyContainer'][f'/{layer_key}/data/title']
                img_data = datafield['data']

                img_unit_xy = datafield['si_unit_xy']['GwySIUnit']['unitstr']
                img_nx = datafield['xres']
                img_ny = datafield['yres']

                if img_nx != nx or img_ny != ny or img_unit_xy != unit_xy:
                    continue

                image_layers[title] = img_data.reshape(ny, nx)

    return RawSurface(data, step_x, step_y, metadata=metadata, image_layers=image_layers)
