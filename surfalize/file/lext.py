# Information on the file format was reconstructed from the Gwyddion `lextfile.c` module
# (https://gwyddion.net). The Olympus LEXT OLS4000 (.lext) format is a multi-directory little-endian
# TIFF. The ImageDescription tag of the first directory contains an XML block (<TiffTagDescData>) that
# holds all acquisition parameters, including the lateral and vertical scaling. Each image directory
# represents a channel whose title is stored in its own ImageDescription tag (e.g. "Height", "Color",
# "Intensity", "Thumbnail" or "Invalid"). Optional per-axis calibration factors are stored in the EXIF
# DeviceSettingDescription tag as a second XML block (<ExifTagDescData>).

import xml.etree.ElementTree as ET

import numpy as np
from dateutil import parser as date_parser
from PIL import Image
from PIL.TiffImagePlugin import OPEN_INFO, II, MM

from .common import RawSurface, FileHandler
from ..exceptions import CorruptedFileError, FileFormatError

# LEXT files store the mask of invalid (non-measured) pixels as a 1-bit TIFF transparency mask
# (photometric = 4). Pillow has no entry for this combination and raises "unknown pixel mode" while
# traversing the directories. We register it as a plain 1-bit image so Pillow can decode it.
for _prefix in (II, MM):
    for _fillorder in (1, 2):
        OPEN_INFO.setdefault((_prefix, 4, (1,), _fillorder, (1,), ()), ('1', '1'))

# TIFF tag ids
TAG_IMAGE_DESCRIPTION = 270
TAG_BITS_PER_SAMPLE = 258
TAG_SAMPLES_PER_PIXEL = 277
TAG_SOFTWARE = 0x131
TAG_DATETIME = 0x132
TAG_EXIF_IFD = 0x8769
TAG_DEVICE_SETTING_DESCRIPTION = 0xA40B

# Substring that identifies the parameter XML block in the first directory's ImageDescription
MAGIC_COMMENT = '<TiffTagDescData'

# The DataPerPixel values are expressed in picometers (see Gwyddion lextfile.c). 1 pm = 1e-6 um.
PICOMETER_TO_UM = 1e-6


def _flatten_xml(xml_string):
    """
    Parse an XML block and flatten it into a dictionary that maps the element path to its text content.

    The path mirrors the keys used by Gwyddion, e.g. ``/TiffTagDescData/HeightInfo/HeightDataPerPixelX``.
    When sibling elements share a tag, the last one wins, matching the behaviour of the reference reader.

    Parameters
    ----------
    xml_string : str
        The XML block to parse.

    Returns
    -------
    dict[str, str]
    """
    # Strip a leading byte order mark or whitespace that would upset the parser
    root = ET.fromstring(xml_string.lstrip('﻿ \r\n\t'))
    result = {}

    def recurse(element, path):
        path = f'{path}/{element.tag}'
        if element.text is not None:
            text = element.text.strip()
            if text:
                result[path] = text
        for child in element:
            recurse(child, path)

    recurse(root, '')
    return result


def _read_exif_params(image):
    """
    Parse the ExifTagDescData XML block stored in the EXIF DeviceSettingDescription tag.

    This block holds the acquisition settings (objective lens, laser settings, stage position,
    Z-stack information and the per-axis maker calibration values).

    Returns
    -------
    dict[str, str]
        Flattened parameter dictionary, empty if the block is absent or cannot be parsed.
    """
    try:
        exif = image.getexif()
        exif_ifd = exif.get_ifd(TAG_EXIF_IFD)
        raw = exif_ifd.get(TAG_DEVICE_SETTING_DESCRIPTION)
        if raw is None:
            return {}
        if isinstance(raw, bytes):
            raw = raw.decode('latin1')
        start = raw.find('<ExifTagDescData')
        if start == -1:
            return {}
        return _flatten_xml(raw[start:])
    except Exception:
        # Metadata is optional; never let a parsing problem prevent the file from loading
        return {}


def _calibration_from_exif(exif_params):
    """
    Extract the optional per-axis calibration factors ``(xcal, ycal, zcal)`` from the parsed EXIF
    parameters. Each factor defaults to 1.0 if absent or unparsable.
    """
    base = '/ExifTagDescData/ImageCommonSettingsInfo/MakerCalibrationValue'
    factors = []
    for axis in ('X', 'Y', 'Z'):
        value = exif_params.get(base + axis)
        try:
            # The reference reader scales the raw maker value by 1e-6
            factors.append(1e-6 * float(value) if value is not None else 1.0)
        except (TypeError, ValueError):
            factors.append(1.0)
    return tuple(factors)


def _channel_title(image, index, guessed_first_title):
    """
    Return the title of the channel in the given directory, title-cased to match the reference reader.
    For the first directory (index 0) the title is not stored explicitly and is taken from the guess.
    """
    if index == 0:
        return guessed_first_title
    description = image.tag_v2.get(TAG_IMAGE_DESCRIPTION)
    if description is None:
        return None
    # capitalize() upper-cases the first character and lower-cases the rest, e.g. "HEIGHT" -> "Height"
    return description.strip().capitalize()


def _guess_first_title(image, n_frames, seen_titles):
    """
    Guess the title of the first directory based on its dimensions and the titles seen in the other
    directories, mirroring the heuristic of Gwyddion's ``guess_image0_title``.
    """
    image.seek(0)
    width, height = image.size
    spp = image.tag_v2.get(TAG_SAMPLES_PER_PIXEL, 1)
    bits = image.tag_v2.get(TAG_BITS_PER_SAMPLE, (8,))
    bpp0 = bits[0] if isinstance(bits, (tuple, list)) else bits

    if width == 128 and height == 128 and spp == 3 and bpp0 == 8:
        return 'Thumbnail' if 'Thumbnail' not in seen_titles else None
    if spp == 3 and bpp0 == 8:
        return 'Color' if 'Color' not in seen_titles else None
    if spp == 1 and bpp0 == 1:
        return 'Invalid' if 'Invalid' not in seen_titles else None
    if spp == 1 and bpp0 == 16:
        if 'Intensity' not in seen_titles:
            return 'Intensity'
        if 'Height' not in seen_titles:
            return 'Intensity'
        return None
    return None


def _collect_channels(image, n_frames):
    """
    Build a mapping of channel title to directory index. The first directory is title-guessed, the
    remaining directories carry their title in the ImageDescription tag.
    """
    seen_titles = set()
    for index in range(1, n_frames):
        image.seek(index)
        title = _channel_title(image, index, None)
        if title is not None:
            seen_titles.add(title)

    guessed_first_title = _guess_first_title(image, n_frames, seen_titles)

    channels = {}
    for index in range(n_frames):
        image.seek(index)
        title = _channel_title(image, index, guessed_first_title)
        if title is None or title == 'Thumbnail':
            continue
        # The first matching directory for a given title wins
        channels.setdefault(title, index)
    return channels


def _build_metadata(image, tiff_params, exif_params):
    """
    Assemble the metadata dictionary from both parameter XML blocks and the standard TIFF tags.

    The leading root path (``/TiffTagDescData/`` or ``/ExifTagDescData/``) is stripped from each key
    to keep the metadata readable, e.g. ``ImageCommonSettingsInfo/ObjectiveLenseType``. In addition,
    a parsed ``timestamp`` and the acquisition ``software`` are extracted from the standard TIFF tags.
    """
    metadata = {}
    for params in (tiff_params, exif_params):
        for key, value in params.items():
            # Drop the leading '/<RootElement>/' so only the meaningful path remains
            stripped = key.lstrip('/').split('/', 1)
            metadata[stripped[1] if len(stripped) > 1 else stripped[0]] = value

    # The DateTime and Software tags live in the first directory; getexif() reflects the directory the
    # image is currently seeked to, so we return to directory 0 before reading them.
    image.seek(0)
    exif = image.getexif()
    datetime_string = exif.get(TAG_DATETIME)
    if datetime_string:
        try:
            metadata['timestamp'] = date_parser.parse(datetime_string)
        except (ValueError, OverflowError):
            pass
    software = exif.get(TAG_SOFTWARE)
    if software:
        metadata['software'] = software.strip()
    return metadata


def _read_data_per_pixel(params, keytitle, axis):
    key = f'/TiffTagDescData/{keytitle}Info/{keytitle}DataPerPixel{axis}'
    value = params.get(key)
    if value is None:
        raise CorruptedFileError(f'Cannot find scaling parameter {key}.')
    return float(value)


@FileHandler.register_reader(suffix='.lext')
def read_lext(filehandle, read_image_layers=False, encoding='utf-8'):
    image = Image.open(filehandle)

    comment = image.tag_v2.get(TAG_IMAGE_DESCRIPTION)
    if comment is None or MAGIC_COMMENT not in comment:
        raise FileFormatError('File does not contain a LEXT TiffTagDescData comment.')

    params = _flatten_xml(comment[comment.find(MAGIC_COMMENT):])
    exif_params = _read_exif_params(image)
    xcal, ycal, zcal = _calibration_from_exif(exif_params)

    n_frames = getattr(image, 'n_frames', 1)
    channels = _collect_channels(image, n_frames)

    if 'Height' not in channels:
        raise CorruptedFileError('The file does not contain a height channel.')

    # Lateral and vertical scaling are stored for the height channel
    data_per_pixel_x = _read_data_per_pixel(params, 'Height', 'X')
    data_per_pixel_y = _read_data_per_pixel(params, 'Height', 'Y')
    data_per_pixel_z = _read_data_per_pixel(params, 'Height', 'Z')

    step_x = data_per_pixel_x * xcal * PICOMETER_TO_UM
    step_y = data_per_pixel_y * ycal * PICOMETER_TO_UM
    scale_z = data_per_pixel_z * zcal * PICOMETER_TO_UM

    image.seek(channels['Height'])
    height_data = np.array(image).astype(np.float64)
    # A height channel may be stored with more than one sample per pixel; keep the first one
    if height_data.ndim == 3:
        height_data = height_data[..., 0]
    height_data *= scale_z

    # The Invalid channel is a 1-bit mask flagging non-measured pixels. We represent those as NaN,
    # analogous to the mask handling in the other readers.
    if 'Invalid' in channels:
        image.seek(channels['Invalid'])
        invalid_mask = np.array(image).astype(bool)
        if invalid_mask.shape == height_data.shape:
            height_data[invalid_mask] = np.nan

    image_layers = {}
    if read_image_layers:
        if 'Color' in channels:
            image.seek(channels['Color'])
            image_layers['RGB'] = np.array(image.convert('RGB'))
        if 'Intensity' in channels:
            image.seek(channels['Intensity'])
            image_layers['Intensity'] = np.array(image)

    metadata = _build_metadata(image, params, exif_params)

    return RawSurface(height_data, step_x, step_y, metadata=metadata, image_layers=image_layers)
