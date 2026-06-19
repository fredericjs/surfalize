try:
    from ._version import __version__
except ImportError:
    __version__ = 'Unknown'
from .surface import Surface
from .profile import Profile
from .mask import Mask
from .batch import Batch, FileInput
