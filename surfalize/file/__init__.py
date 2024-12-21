import pkgutil
import importlib

# load every module once to ensure that their reader and writer functions can register themselves with the FileHandler
# class. If we don't do this, we would otherwise need to import every module here once.
for loader, name, is_pkg in pkgutil.iter_modules(__path__):
    module = importlib.import_module('.' + name, package=__package__)
    globals()[name] = module

from .common import FileHandler

supported_formats_read = FileHandler.get_supported_formats_read()
supported_formats_write = FileHandler.get_supported_formats_write()
