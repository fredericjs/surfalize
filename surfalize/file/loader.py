from pathlib import Path

from ..exceptions import UnsupportedFileFormatError

from .vk import read_vk4, read_vk6_vk7
from .plu import read_plu
from .plux import read_plux
from .sur import read_sur, write_sur
from .zmg import read_zmg
from .opd import read_opd
from .xyz import read_xyz
from .nms import read_nms
from .al3d import read_al3d, write_al3d
from .sdf import read_sdf, write_sdf
from .gwy import read_gwy
from .os3d import read_os3d
from .fits import read_fits
from .sflz import read_sflz, write_sflz

dispatcher = {
    '.sur':     {'read': read_sur, 'write': write_sur},
    '.vk4':     {'read': read_vk4},
    '.vk6':     {'read': read_vk6_vk7},
    '.vk7':     {'read': read_vk6_vk7},
    '.plu':     {'read': read_plu},
    '.plux':    {'read': read_plux},
    '.zmg':     {'read': read_zmg},
    '.opd':     {'read': read_opd},
    '.xyz':     {'read': read_xyz},
    '.nms':     {'read': read_nms},
    '.al3d':    {'read': read_al3d, 'write': write_al3d},
    '.sdf':     {'read': read_sdf, 'write': write_sdf},
    '.gwy':     {'read': read_gwy},
    '.os3d':    {'read': read_os3d},
    '.fits':    {'read': read_fits},
    '.sflz':    {'read': read_sflz, 'write': write_sflz}
}

supported_formats = list(dispatcher.keys())

def load_file(filepath, read_image_layers=False, encoding="utf-8"):
    filepath = Path(filepath)
    try:
        loader = dispatcher[filepath.suffix]['read']
    except KeyError:
        raise UnsupportedFileFormatError(f"File format {filepath.suffix} is currently not supported.") from None
    return loader(filepath, read_image_layers=read_image_layers, encoding=encoding)

def write_file(filepath, surface, encoding='utf-8', **kwargs):
    filepath = Path(filepath)
    ext = filepath.suffix
    if not ext:
        raise ValueError('No format for the file specified.') from None
    try:
        writer = dispatcher[filepath.suffix]['write']
    except KeyError:
        raise UnsupportedFileFormatError(
            f"File format {filepath.suffix} is currently not supported for writing.") from None
    return writer(filepath, surface, encoding=encoding, **kwargs)
