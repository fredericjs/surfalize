import zipfile
import xml.etree.ElementTree as ET
import numpy as np

def read_plux(filepath, encoding='utf-8'):
    with zipfile.ZipFile(filepath) as archive:
        data = archive.read('LAYER_0.raw')
        metadata = archive.read('index.xml')

    xml_str = metadata.decode(encoding)

    # Parse the XML string
    root = ET.fromstring(xml_str)
    shape_x = int(root.find('GENERAL/IMAGE_SIZE_X').text)
    shape_y = int(root.find('GENERAL/IMAGE_SIZE_Y').text)
    step_x = float(root.find('GENERAL/FOV_X').text)
    step_y = float(root.find('GENERAL/FOV_Y').text)
    size = shape_x * shape_y

    data = np.frombuffer(data, dtype=np.float32, count=size).reshape((shape_y, shape_x))
    return (data, step_x, step_y)