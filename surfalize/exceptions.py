class FileFormatError(Exception):
    pass

class UnsupportedFileFormatError(FileFormatError):
    pass

class CorruptedFileError(FileFormatError):
    pass
