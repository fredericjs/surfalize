import pytest
from pathlib import Path
import numpy as np
from surfalize import Surface
from surfalize.file import supported_formats_read
from surfalize.file.loader import dispatcher

module_path = Path(__file__).parent

supported_write_formats = [fmt for fmt, val in dispatcher.items() if 'write' in val]

def almost_equal(surface1, surface2):
    if surface1.size != surface2.size:
        return False
    if np.any(np.abs(surface1.data - surface2.data) > 1e-6):
        return False
    if abs(surface1.step_x - surface2.step_x) > 1e-6:
        return False
    if abs(surface1.step_y - surface2.step_y) > 1e-6:
        return False
    return True

@pytest.fixture
def testfile_dir():
    return module_path / 'test_files'

@pytest.mark.parametrize('fileformat', supported_formats_read)
def test_fileformat_loading(testfile_dir, fileformat):
    files = list(testfile_dir.glob(f'*{fileformat}'))
    if not files:
        pytest.skip('No testfiles found.')
    for file in files:
        Surface.load(file, read_image_layers=False)
        Surface.load(file, read_image_layers=True)

@pytest.mark.parametrize('fileformat', supported_write_formats)
def test_fileformat_writing(surface, tmpdir, fileformat):
    path = tmpdir / ('test' + fileformat)
    surface.save(path)
    assert almost_equal(Surface.load(path), surface)
