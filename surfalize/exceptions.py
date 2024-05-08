class FileFormatError(Exception):
    """
    Base class for file related errors.
    """
    pass


class UnsupportedFileFormatError(FileFormatError):
    """
    Raised when a file format is not supported.
    """
    pass


class CorruptedFileError(FileFormatError):
    """
    Raised when a file is corrupted.
    """
    pass

class CalculationError(Exception):
    """
    Base class for all calculation related errors.
    """

class FittingError(CalculationError):
    """
    Raised when a fitting operation has failed.
    """
    pass


class BatchError(Exception):
    """
    Raised when an error occurs in batch processing.
    """
    pass