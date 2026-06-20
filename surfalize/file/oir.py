# Information on the file format was reconstructed from the Gwyddion `oirfile.c` module
# (https://gwyddion.net). The Olympus OIR (.oir) format is a binary container that starts with a
# 96-byte header (magic "OLYMPUSRAWFORMAT"), followed by a stream of chunks. Each chunk starts with a
# little-endian uint32 size and a uint32 type. The relevant chunk types are a BMP thumbnail (skipped),
# a triplet of reference images (skipped), and an image chunk that holds a small XML fragment
# describing the frame dimensions followed by three binary image data blocks. The channel each data
# block belongs to (HEIGHT, INTENSITY, INVALID for LSM data, or the RGB planes for camera data) is
# matched by its UUID against a separate `imageProperties` XML fragment that also carries the lateral
# and vertical calibration. The packed OIR (.poir) format is simply a ZIP archive bundling one or more
# .oir files (e.g. the LSM height data and the color camera image).

import io
import re
import struct
import zipfile
from collections import Counter
import xml.etree.ElementTree as ET

import numpy as np
from dateutil import parser as date_parser

from .common import RawSurface, FileHandler
from ..exceptions import CorruptedFileError, FileFormatError

OIR_MAGIC = b'OLYMPUSRAWFORMAT'
HEADER_LENGTH = 96

CHUNK_XML0 = 0
CHUNK_XML = 1
CHUNK_BMP = 2
CHUNK_WTF = 3
CHUNK_TERMINATOR = 96

# Pixel depth (bytes per sample) to numpy dtype
DEPTH_DTYPE = {1: np.uint8, 2: '<u2'}

# XML preamble before the frame metadata: content_size, unknown1, id and 8 further unknowns, all
# little-endian uint32, followed by the uint32 XML length.
IMAGE_META_PREAMBLE = 12 * 4


def _u32(buffer, offset):
    return struct.unpack_from('<I', buffer, offset)[0]


def _localname(tag):
    """Strip the XML namespace from a tag or attribute name."""
    return tag.split('}', 1)[1] if '}' in tag else tag


def _read_data_block(buffer, p):
    """
    Read one image data block. Returns ``(uuid, data, new_position)``.

    Layout: remainder_size, chunktype, unknown, image_size (4 uint32), uuid_size (uint32), the uuid
    bytes, image_size_again and unknown (2 uint32), and finally ``image_size`` bytes of raw data.
    """
    image_size = _u32(buffer, p + 12)
    p += 16
    uuid_size = _u32(buffer, p)
    p += 4
    uuid = buffer[p:p + uuid_size].decode('latin1')
    p += uuid_size
    p += 8  # image_size_again, unknown
    data = buffer[p:p + image_size]
    if len(data) != image_size:
        raise CorruptedFileError('Truncated image data block.')
    p += image_size
    return uuid, data, p


def _read_frame_xml(buffer, p):
    """Read the frame metadata XML fragment of an image chunk. Returns ``(xml_bytes, new_position)``."""
    xml_size = _u32(buffer, p + IMAGE_META_PREAMBLE - 4)
    start = p + IMAGE_META_PREAMBLE
    return buffer[start:start + xml_size], start + xml_size


def _extract_image_blocks(buffer):
    """
    Walk the chunk stream and return ``(frame_xml, blocks)`` where ``blocks`` is the list of the three
    ``(uuid, data)`` image data blocks. Returns ``(None, None)`` if the file contains no image chunk.
    """
    if buffer[:len(OIR_MAGIC)] != OIR_MAGIC:
        raise FileFormatError('File does not start with the OLYMPUSRAWFORMAT magic.')
    p = HEADER_LENGTH
    n = len(buffer)
    while p + 8 <= n:
        chunktype = _u32(buffer, p + 4)
        if chunktype == CHUNK_BMP:
            thumbnail_size = _u32(buffer, p)
            p += 8 + thumbnail_size
        elif chunktype == CHUNK_WTF:
            # A triplet of reference images we are not interested in
            for _ in range(3):
                _, _, p = _read_data_block(buffer, p)
        elif chunktype == CHUNK_XML:
            frame_xml, p = _read_frame_xml(buffer, p)
            blocks = []
            for _ in range(3):
                uuid, data, p = _read_data_block(buffer, p)
                blocks.append((uuid, data))
            return frame_xml, blocks
        else:
            # Terminator or anything unexpected: no (more) image data
            break
    return None, None


def _extract_named_fragment(buffer, localname):
    """
    Extract a complete XML fragment ``<prefix:localname ...> ... </prefix:localname>`` from the raw
    buffer. Returns the fragment bytes or ``None`` if it is not present.
    """
    match = re.search(rb'<([A-Za-z0-9_]+):' + localname.encode() + rb'[ >]', buffer)
    if match is None:
        return None
    prefix = match.group(1).decode()
    closing = f'</{prefix}:{localname}>'.encode()
    end = buffer.find(closing, match.start())
    if end == -1:
        return None
    return buffer[match.start():end + len(closing)]


def _parse_xml(fragment):
    """Parse an XML fragment, returning the root element (namespaces are handled via local names)."""
    return ET.fromstring(fragment)


def _find_all(element, path):
    """Return all descendant elements matching a ``/``-separated path of local names."""
    nodes = [element]
    for segment in path.split('/'):
        nodes = [child for node in nodes for child in node if _localname(child.tag) == segment]
    return nodes


def _find_text(element, path):
    """Return the stripped text of the first element matching ``path``, or ``None``."""
    nodes = _find_all(element, path)
    if nodes and nodes[0].text is not None and nodes[0].text.strip():
        return nodes[0].text.strip()
    return None


def _flatten_et(element, result, path):
    """
    Flatten an XML element tree into ``result`` using local names, indexing repeated sibling elements
    so that arrays are not silently overwritten.
    """
    if element.text is not None and element.text.strip():
        result[path] = element.text.strip()
    for attr_name, attr_value in element.attrib.items():
        result[f'{path}@{_localname(attr_name)}'] = attr_value
    tag_counts = Counter(_localname(child.tag) for child in element)
    seen = {}
    for child in element:
        tag = _localname(child.tag)
        if tag_counts[tag] > 1:
            index = seen.get(tag, 0)
            seen[tag] = index + 1
            child_path = f'{path}/{tag}[{index}]'
        else:
            child_path = f'{path}/{tag}'
        _flatten_et(child, result, child_path)


def _parse_oir(buffer):
    """
    Parse a single OIR buffer into a structured dictionary describing its image channels, or ``None``
    if it carries no image data.
    """
    frame_xml, blocks = _extract_image_blocks(buffer)
    if blocks is None:
        return None

    frame_root = _parse_xml(frame_xml)
    width = int(_find_text(frame_root, 'imageDefinition/width'))
    height = int(_find_text(frame_root, 'imageDefinition/height'))
    depth = int(_find_text(frame_root, 'imageDefinition/depth'))

    device = None
    pixel_calibration = {'x': 1.0, 'y': 1.0, 'z': 1.0}
    channels = []
    image_props_xml = _extract_named_fragment(buffer, 'imageProperties')
    image_props_root = None
    if image_props_xml is not None:
        image_props_root = _parse_xml(image_props_xml)
        device = _find_text(image_props_root, 'imageInfo/acquireDevice')
        for channel in _find_all(image_props_root, 'imageInfo/phase/group/channel'):
            channels.append({
                'id': channel.get('id'),
                'imageType': _find_text(channel, 'imageDefinition/imageType'),
                'length': (
                    _find_text(channel, 'length/x'),
                    _find_text(channel, 'length/y'),
                    _find_text(channel, 'length/z'),
                ),
            })
        for axis in ('x', 'y', 'z'):
            value = _find_text(image_props_root, f'acquisition/microscopeConfiguration/pixelCalibration/{axis}')
            if value is not None:
                pixel_calibration[axis] = float(value)

    return {
        'width': width,
        'height': height,
        'depth': depth,
        'blocks': blocks,
        'device': device,
        'channels': channels,
        'pixel_calibration': pixel_calibration,
        'image_props_root': image_props_root,
        'buffer': buffer,
    }


def _match_channel(uuid, channels):
    """Return the channel metadata whose id is contained in the block uuid, or ``None``."""
    for channel in channels:
        if channel['id'] and channel['id'] in uuid:
            return channel
    return None


def _collect_metadata(parsed, metadata):
    """Flatten the imageProperties of one OIR into ``metadata`` and extract a timestamp if present."""
    prefix = (parsed['device'] or 'oir')
    if parsed['image_props_root'] is not None:
        flat = {}
        _flatten_et(parsed['image_props_root'], flat, _localname(parsed['image_props_root'].tag))
        for key, value in flat.items():
            metadata[f'{prefix}/{key}'] = value
    if 'timestamp' not in metadata:
        match = re.search(rb'creationDateTime>([^<]+)<', parsed['buffer'])
        if match:
            try:
                metadata['timestamp'] = date_parser.parse(match.group(1).decode('latin1').strip())
            except (ValueError, OverflowError):
                pass


def _build_surface(oir_buffers, read_image_layers):
    """Build a RawSurface from one or more OIR buffers (the height channel plus optional image layers)."""
    height_data = None
    step_x = step_y = None
    image_layers = {}
    metadata = {}

    for buffer in oir_buffers:
        parsed = _parse_oir(buffer)
        if parsed is None:
            continue
        width, height, depth = parsed['width'], parsed['height'], parsed['depth']
        dtype = DEPTH_DTYPE.get(depth)
        if dtype is None:
            continue
        channels = parsed['channels']
        calibration = parsed['pixel_calibration']
        device = parsed['device']

        typed_blocks = [(_match_channel(uuid, channels), data) for uuid, data in parsed['blocks']]

        is_camera = (device is not None and device.lower() == 'camera') or all(ch is None for ch, _ in typed_blocks)
        if is_camera:
            if read_image_layers and len(typed_blocks) == 3:
                planes = [np.frombuffer(data, dtype=np.uint8).reshape(height, width) for _, data in typed_blocks]
                image_layers.setdefault('RGB', np.stack(planes, axis=2))
        else:
            for channel, data in typed_blocks:
                image_type = channel['imageType'] if channel else None
                array = np.frombuffer(data, dtype=dtype).reshape(height, width)
                if image_type == 'HEIGHT' and height_data is None:
                    length_x, length_y, length_z = channel['length']
                    height_data = array.astype(np.float64) * float(length_z) * calibration['z']
                    step_x = float(length_x) * calibration['x']
                    step_y = float(length_y) * calibration['y']
                elif read_image_layers and image_type is not None:
                    image_layers.setdefault(image_type.capitalize(), array.copy())

        _collect_metadata(parsed, metadata)

    if height_data is None:
        raise CorruptedFileError('The file does not contain a height channel.')

    return RawSurface(height_data, step_x, step_y, metadata=metadata, image_layers=image_layers)


@FileHandler.register_reader(suffix='.oir', magic=OIR_MAGIC)
def read_oir(filehandle, read_image_layers=False, encoding='utf-8'):
    return _build_surface([filehandle.read()], read_image_layers)


@FileHandler.register_reader(suffix='.poir')
def read_poir(filehandle, read_image_layers=False, encoding='utf-8'):
    with zipfile.ZipFile(io.BytesIO(filehandle.read())) as archive:
        oir_buffers = [archive.read(name) for name in archive.namelist() if name.lower().endswith('.oir')]
    if not oir_buffers:
        raise CorruptedFileError('The packed OIR archive does not contain any .oir files.')
    return _build_surface(oir_buffers, read_image_layers)
