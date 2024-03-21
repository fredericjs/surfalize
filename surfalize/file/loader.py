from pathlib import Path
from functools import partial
from ..exceptions import UnsupportedFileFormatError

from .vk import read_vk4, read_vk6_vk7
from .plu import read_plu
from .plux import read_plux
from .sur import read_sur
from .zmg import read_zmg
from .opd import read_opd
from .xyz import read_xyz
from .nms import read_nms
from .al3d import read_al3d

dispatcher = {
    '.sur': read_sur,
    '.vk4': read_vk4,
    '.vk6': read_vk6_vk7,
    '.vk7': read_vk6_vk7,
    '.plu': read_plu,
    '.plux': read_plux,
    '.zmg': read_zmg,
    '.opd': read_opd,
    '.xyz': read_xyz,
    '.nms': read_nms,
    '.al3d': read_al3d
}

supported_formats = list(dispatcher.keys())

def load_file(filepath, encoding='utf-8'):
    filepath = Path(filepath)
    try:
        loader = dispatcher[filepath.suffix]
        if 'encoding' in loader.__code__.co_varnames:
            loader = partial(loader, encoding=encoding)
    except KeyError:
        raise UnsupportedFileFormatError(f"File format {filepath.suffix} is currently not supported.")
    return loader(filepath)