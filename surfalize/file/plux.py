import zipfile
import dateutil
import xml.etree.ElementTree as ET
import numpy as np
from .common import RawSurface

# Names of the files in the zip archive
TOPOGRAPHY_FILE_NAME = 'LAYER_0.raw'
IMAGE_FILE_NAME = 'LAYER_0.stack.raw'
XML_METADATA_FILE_NAME = 'index.xml'

def read_plux(filepath, read_image_layers=False, encoding='utf-8'):
    with zipfile.ZipFile(filepath) as archive:
        contents = archive.namelist()
        data_raw = archive.read(TOPOGRAPHY_FILE_NAME)
        if read_image_layers and IMAGE_FILE_NAME in contents:
            img_raw = archive.read(IMAGE_FILE_NAME)
        xml_metadata = archive.read(XML_METADATA_FILE_NAME)

    xml_str = xml_metadata.decode(encoding)
    metadata = {}
    # Parse the XML string
    root = ET.fromstring(xml_str)
    metadata['timestamp'] = dateutil.parser.parse(root.find('GENERAL/DATE').text)

    # Add selected entires of XML info section to metadata
    info_section = root.find('INFO')
    tags = ['Device', 'Objective', 'Technique', 'Measurement Type', 'Algorithm', 'Comment']
    if info_section is not None:
        # Iterate over all child elements of INFO
        for item in info_section:
            if item.tag.startswith('ITEM_'):
                # Extract NAME and VALUE from each ITEM
                name = item.find('NAME').text.strip()
                if name in tags:
                    metadata[name.lower()] = item.find('VALUE').text.strip()

    shape_x = int(root.find('GENERAL/IMAGE_SIZE_X').text)
    shape_y = int(root.find('GENERAL/IMAGE_SIZE_Y').text)
    step_x = float(root.find('GENERAL/FOV_X').text)
    step_y = float(root.find('GENERAL/FOV_Y').text)
    size = shape_x * shape_y

    data = np.frombuffer(data_raw, dtype=np.float32, count=size).reshape((shape_y, shape_x))
    image_layers = {}
    if read_image_layers:
        img = np.frombuffer(img_raw, dtype=np.uint8).reshape((shape_y, shape_x, 3))
        if np.all((img[:, :, 0] == img[:, :, 1]) & (img[:, :, 0] == img[:, :, 2])):
            image_layers['Grayscale'] = img[:, :, 0]
        else:
            image_layers['RGB'] = img
    return RawSurface(data, step_x, step_y, image_layers=image_layers, metadata=metadata)