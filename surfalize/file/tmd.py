from ..exceptions import CorruptedFileError
from .common import RawSurface, FileHandler, read_array, write_array, Layout, Entry, decode

MAGIC = b'Binary TrueMap Data File v2.0'
LINE_TERMINATION = '\r\n\x00'
DTYPE = 'float32'

COMMENT_LEN = 24 - len(LINE_TERMINATION)

HEADER_LAYOUT = Layout(
    Entry('header', '32s'),
    Entry('comment', '24s'),
    Entry('width', '<I'),
    Entry('height', '<I'),
    Entry('length_x', '<f'),
    Entry('length_y', '<f'),
    Entry('offset_x', '<f'),
    Entry('offset_y', '<f'),
)

@FileHandler.register_reader(suffix='.tmd', magic=MAGIC)
def read_tmd(filehandle, read_image_layers=False, encoding='utf-8'):
    magic = filehandle.read(len(MAGIC))
    if magic != MAGIC:
        raise CorruptedFileError('Incorrect file magic detected.')
    filehandle.seek(0)
    header = HEADER_LAYOUT.read(filehandle)
    data = read_array(filehandle, dtype=DTYPE)
    if data.size != header['height'] * header['width']:
        raise CorruptedFileError('Size of data does not match expectted size.')
    data = data.reshape(header['height'], header['width'])
    step_x = header['length_x'] / header['width']
    step_y = header['length_y'] / header['height']

    return RawSurface(data, step_x, step_y, metadata=header)

@FileHandler.register_writer(suffix='.tmd')
def write_tmd(filehandle, surface, encoding='utf-8', comment='Exported by surfalize'):
    if len(comment) > COMMENT_LEN:
        raise ValueError(f'Comment has too many characters. Maximum number allows is {COMMENT_LEN}')
    header = {
        'header': decode(MAGIC, encoding) + LINE_TERMINATION,
        'comment': comment + ' ' * (COMMENT_LEN - len(comment)) + LINE_TERMINATION,
        'width': surface.size.x,
        'height': surface.size.y,
        'length_x': surface.size.x * surface.step_x,
        'length_y': surface.size.y * surface.step_y,
        'offset_x': 0,
        'offset_y': 0,
    }
    HEADER_LAYOUT.write(filehandle, header)
    write_array(surface.data.astype(DTYPE), filehandle)