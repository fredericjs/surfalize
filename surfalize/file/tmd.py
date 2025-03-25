import os
import struct
import tempfile
import numpy as np
import logging

from ..exceptions import CorruptedFileError
from .common import RawSurface, FileHandler, np_from_any, np_to_any

logger = logging.getLogger(__name__)

# Constants specific to TMD v2 files
HEADER_LEN = 32      # bytes for the header string
COMMENT_LEN = 24     # bytes reserved for comment (v2 files)
INT_LEN = 4          # bytes per integer field (e.g., width, height)
FLOAT_LEN = 4        # bytes per float field (e.g., spatial parameters)

def process_tmd_file(file_path, force_offset=None, debug=False):
    """
    Process a TMD v2 file and extract metadata and the height map.

    Args:
        file_path (str): Path to the TMD file.
        force_offset (tuple, optional): Tuple (x_offset, y_offset) to override file values.
        debug (bool, optional): Enable debug logging.

    Returns:
        tuple: (metadata, height_map) where metadata is a dict and height_map is a numpy.ndarray.

    Raises:
        ValueError: If the file is too small or lacks sufficient data.
    """
    with open(file_path, 'rb') as f:
        data = f.read()

    # Minimum required size: header + comment + dimensions (2 ints) + spatial parameters (4 floats)
    min_size = HEADER_LEN + COMMENT_LEN + (2 * INT_LEN) + (4 * FLOAT_LEN)
    if len(data) < min_size:
        raise ValueError("File is too small to be a valid TMD v2 file.")

    pos = 0
    # Read header string (32 bytes) and decode as ASCII.
    header_str = data[pos:pos + HEADER_LEN].decode("ascii", errors="ignore").strip()
    pos += HEADER_LEN

    # Read comment (24 bytes); stop at the first null terminator.
    comment_bytes = data[pos:pos + COMMENT_LEN]
    null_index = comment_bytes.find(b'\0')
    if null_index != -1:
        comment_str = comment_bytes[:null_index].decode("ascii", errors="ignore").strip()
    else:
        comment_str = comment_bytes.decode("ascii", errors="ignore").strip()
    pos += COMMENT_LEN

    # Read image dimensions: width and height (unsigned int, little-endian)
    if len(data) < pos + 2 * INT_LEN:
        raise ValueError("Insufficient data for dimensions.")
    width = struct.unpack("<I", data[pos:pos + INT_LEN])[0]
    pos += INT_LEN
    height = struct.unpack("<I", data[pos:pos + INT_LEN])[0]
    pos += INT_LEN

    # Read spatial parameters: x_length, y_length, x_offset, y_offset (floats, little-endian)
    if len(data) < pos + 4 * FLOAT_LEN:
        raise ValueError("Insufficient data for spatial parameters.")
    x_length = struct.unpack("<f", data[pos:pos + FLOAT_LEN])[0]
    pos += FLOAT_LEN
    y_length = struct.unpack("<f", data[pos:pos + FLOAT_LEN])[0]
    pos += FLOAT_LEN
    x_offset = struct.unpack("<f", data[pos:pos + FLOAT_LEN])[0]
    pos += FLOAT_LEN
    y_offset = struct.unpack("<f", data[pos:pos + FLOAT_LEN])[0]
    pos += FLOAT_LEN

    if force_offset is not None:
        x_offset, y_offset = force_offset

    mmpp = x_length / width if width > 0 else 0.0

    metadata = {
        "header": header_str,
        "comment": comment_str,
        "width": width,
        "height": height,
        "x_length": x_length,
        "y_length": y_length,
        "x_offset": x_offset,
        "y_offset": y_offset,
        "mmpp": mmpp,
        "version": 2,
    }

    # Calculate expected size for the height map data.
    expected_bytes = width * height * FLOAT_LEN
    if len(data) < pos + expected_bytes:
        raise ValueError("Not enough data in the file for the height map.")

    # Read the height map data using numpy for efficiency.
    height_map = np.frombuffer(data, dtype=np.float32, offset=pos, count=width * height)
    height_map = height_map.reshape((height, width))

    if debug:
        logger.debug("Processed TMD file:")
        logger.debug(f"Header: {header_str}")
        logger.debug(f"Comment: {comment_str}")
        logger.debug(f"Dimensions: {width} x {height}")
        logger.debug(f"Spatial parameters: x_length={x_length}, y_length={y_length}, x_offset={x_offset}, y_offset={y_offset}")

    return metadata, height_map

@FileHandler.register_reader(suffix='.tmd', magic=b'Binary TrueMap Data File v2.0')
def read_tmd_v2(filehandle, read_image_layers=False, encoding='utf-8'):
    """
    Reader for TMD v2 files.

    This function validates the TMD v2 header, then processes the file using the
    locally implemented process_tmd_file function. It supports both file paths and
    file-like objects. If the file-like object lacks a 'name' attribute, its content
    is temporarily written to disk for processing.

    Args:
        filehandle: A file-like object to read TMD v2 data from.
        read_image_layers: (Optional) Flag to read image layers if present.
        encoding: (Optional) Text encoding for string fields.

    Returns:
        RawSurface: An object containing the height map, spatial resolution, and metadata.

    Raises:
        CorruptedFileError: If the file header does not match the expected TMD v2 magic.
    """
    # Validate the header magic.
    filehandle.seek(0)
    header_bytes = filehandle.read(HEADER_LEN)
    expected_magic_str = b'Binary TrueMap Data File v2.0'
    if not header_bytes.startswith(expected_magic_str):
        raise CorruptedFileError(f"File header does not match expected TMD v2 magic: {header_bytes}")
    filehandle.seek(0)

    # Process the file using its file name if available; otherwise, use a temporary file.
    try:
        file_path = filehandle.name
        metadata, height_map = process_tmd_file(file_path, force_offset=None, debug=False)
    except AttributeError:
        content = filehandle.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmd') as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            metadata, height_map = process_tmd_file(tmp_path, force_offset=None, debug=False)
        finally:
            os.remove(tmp_path)

    # Compute spatial resolution per pixel.
    width = metadata.get("width", 1)
    height = metadata.get("height", 1)
    step_x = metadata.get("x_length", 1) / width if width else 1
    step_y = metadata.get("y_length", 1) / height if height else 1

    return RawSurface(height_map, step_x, step_y, metadata=metadata)
