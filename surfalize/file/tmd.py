import os
import struct
import tempfile
import numpy as np
import logging

from ..exceptions import CorruptedFileError
from .common import RawSurface, FileHandler, np_from_any, np_to_any

logger = logging.getLogger(__name__)

# Constants for TMD v2 files
HEADER_LEN = 32      # bytes for header string
COMMENT_LEN = 24     # bytes for comment (v2 files)
INT_LEN = 4          # bytes per integer field
FLOAT_LEN = 4        # bytes per float field

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

    min_size = HEADER_LEN + COMMENT_LEN + (2 * INT_LEN) + (4 * FLOAT_LEN)
    if len(data) < min_size:
        raise ValueError("File is too small to be a valid TMD v2 file.")

    pos = 0
    # Read header string
    header_str = data[pos:pos + HEADER_LEN].decode("ascii", errors="ignore").strip()
    pos += HEADER_LEN

    # Read comment (24 bytes), stop at null terminator if any.
    comment_bytes = data[pos:pos + COMMENT_LEN]
    null_index = comment_bytes.find(b'\0')
    if null_index != -1:
        comment_str = comment_bytes[:null_index].decode("ascii", errors="ignore").strip()
    else:
        comment_str = comment_bytes.decode("ascii", errors="ignore").strip()
    pos += COMMENT_LEN

    # Read dimensions (width and height)
    width = struct.unpack("<I", data[pos:pos + INT_LEN])[0]
    pos += INT_LEN
    height = struct.unpack("<I", data[pos:pos + INT_LEN])[0]
    pos += INT_LEN

    # Read spatial parameters: x_length, y_length, x_offset, y_offset
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

    expected_bytes = width * height * FLOAT_LEN
    if len(data) < pos + expected_bytes:
        raise ValueError("Not enough data in the file for the height map.")

    height_map = np.frombuffer(data, dtype=np.float32, offset=pos, count=width * height)
    height_map = height_map.reshape((height, width))

    if debug:
        logger.debug("Processed TMD file:")
        logger.debug(f"Header: {header_str}")
        logger.debug(f"Comment: {comment_str}")
        logger.debug(f"Dimensions: {width} x {height}")
        logger.debug(f"Spatial parameters: x_length={x_length}, y_length={y_length}, x_offset={x_offset}, y_offset={y_offset}")

    return metadata, height_map

@FileHandler.register_reader(suffix=['.tmd'], magic=b'Binary TrueMap Data File v2.0')
def read_tmd_v2(filehandle, read_image_layers=False, encoding='utf-8'):
    """
    Reader for TMD v2 files.

    Validates the TMD v2 header, then processes the file using process_tmd_file.
    Supports file paths and file-like objects (using a temporary file if necessary).

    Args:
        filehandle: A file-like object for TMD v2 data.
        read_image_layers: (Optional) Flag to read image layers.
        encoding: (Optional) Text encoding for string fields.

    Returns:
        RawSurface: An object containing the height map, spatial resolution, and metadata.

    Raises:
        CorruptedFileError: If the file header does not match expected TMD v2 magic.
    """
    filehandle.seek(0)
    header_bytes = filehandle.read(HEADER_LEN)
    expected_magic_str = b'Binary TrueMap Data File v2.0'
    if not header_bytes.startswith(expected_magic_str):
        raise CorruptedFileError(f"File header does not match expected TMD v2 magic: {header_bytes}")
    filehandle.seek(0)

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

    width = metadata.get("width", 1)
    height = metadata.get("height", 1)
    step_x = metadata.get("x_length", 1) / width if width else 1
    step_y = metadata.get("y_length", 1) / height if height else 1

    return RawSurface(height_map, step_x, step_y, metadata=metadata)

@FileHandler.register_writer(suffix=['.tmd'])
def write_tmd_v2(filehandle, surface, encoding='utf-8'):
    """
    Writer for TMD v2 files.

    This function writes a TMD v2 file with the following structure:
      - A 32-byte header string ("Binary TrueMap Data File v2.0") padded with nulls.
      - A 24-byte comment string padded with nulls.
      - Two unsigned integers (4 bytes each, little-endian) for width and height.
      - Four floats (4 bytes each, little-endian) for x_length, y_length, x_offset, and y_offset.
      - A block of height map data as float32 values.

    Args:
        filehandle: A file-like object to write the TMD v2 data to.
        surface: A RawSurface object containing the height map, spatial resolution, and metadata.
        encoding: (Optional) The text encoding used for string fields (default 'utf-8').

    Returns:
        None
    """
    # Write header string, padded to HEADER_LEN bytes.
    header_str = "Binary TrueMap Data File v2.0"
    header_bytes = header_str.encode("ascii")
    if len(header_bytes) < HEADER_LEN:
        header_bytes += b'\0' * (HEADER_LEN - len(header_bytes))
    else:
        header_bytes = header_bytes[:HEADER_LEN]
    filehandle.write(header_bytes)

    # Write comment string, padded to COMMENT_LEN bytes.
    comment = surface.metadata.get("comment", "Created by surfalize TMD v2 writer")
    comment_bytes = comment.encode("ascii")
    if len(comment_bytes) < COMMENT_LEN:
        comment_bytes += b'\0' * (COMMENT_LEN - len(comment_bytes))
    else:
        comment_bytes = comment_bytes[:COMMENT_LEN]
    filehandle.write(comment_bytes)

    # Retrieve dimensions from surface data (assumed shape: (height, width))
    height, width = surface.data.shape
    filehandle.write(struct.pack("<I", width))
    filehandle.write(struct.pack("<I", height))

    # Determine spatial parameters:
    # Use metadata if provided, otherwise derive from step sizes.
    x_length = surface.metadata.get("x_length", surface.step_x * width)
    y_length = surface.metadata.get("y_length", surface.step_y * height)
    x_offset = surface.metadata.get("x_offset", 0.0)
    y_offset = surface.metadata.get("y_offset", 0.0)

    filehandle.write(struct.pack("<f", x_length))
    filehandle.write(struct.pack("<f", y_length))
    filehandle.write(struct.pack("<f", x_offset))
    filehandle.write(struct.pack("<f", y_offset))

    # Write the height map data as float32.
    data = surface.data.astype(np.float32)
    filehandle.write(data.tobytes())

    logger.debug("TMD v2 file written:")
    logger.debug(f"Dimensions: {width} x {height}")
    logger.debug(f"Spatial parameters: x_length={x_length}, y_length={y_length}, x_offset={x_offset}, y_offset={y_offset}")
