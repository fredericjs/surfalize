import pytest
from pathlib import Path
import io
import numpy as np
from surfalize import Surface
from surfalize.file import supported_formats_read, supported_formats_write

module_path = Path(__file__).parent

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
def test_fileformat_reading_from_path(testfile_dir, fileformat):
    files = list(testfile_dir.glob(f'*{fileformat}'))
    if not files:
        pytest.skip('No testfiles found.')
    for file in files:
        Surface.load(file, read_image_layers=False)
        Surface.load(file, read_image_layers=True)

@pytest.mark.parametrize('fileformat', supported_formats_read)
def test_fileformat_reading_from_buffer(testfile_dir, fileformat):
    files = list(testfile_dir.glob(f'*{fileformat}'))
    if not files:
        pytest.skip('No testfiles found.')
    for file in files:
        with open(file, 'rb') as f:
            buffer = io.BytesIO(f.read())
        Surface.load(buffer, format=file.suffix, read_image_layers=True)

@pytest.mark.parametrize('fileformat', supported_formats_write)
def test_fileformat_writing_to_path(surface, tmpdir, fileformat):
    path = tmpdir / ('test' + fileformat)
    surface.save(path)
    assert almost_equal(Surface.load(path), surface)

@pytest.mark.parametrize('fileformat', supported_formats_write)
def test_fileformat_writing_to_buffer(surface, fileformat):
    buffer = io.BytesIO()
    surface.save(buffer, format=fileformat)
    assert almost_equal(Surface.load(buffer), surface)

def test_sur_encoding(testfile_dir):
    surface = Surface.load(testfile_dir / 'test_uncompressed.sur', encoding='utf-8')
    buffer = io.BytesIO()
    surface.save(buffer, format='.sur', encoding='latin-1')
    with pytest.raises(UnicodeDecodeError):
        surface.load(buffer, format='.sur', encoding='utf-8')
    buffer.seek(0)
    surface.load(buffer, format='.sur', encoding='latin-1')
    buffer.seek(0)
    surface.load(buffer, format='.sur', encoding='auto')

def test_tmd_file_loading(testfile_dir):
    """
    Test loading of a TMD file (test_1.tmd).
    """
    tmd_file = testfile_dir / 'test_1.tmd'
    if not tmd_file.exists():
        pytest.skip("test_1.tmd file not found in test_files directory.")
    surface = Surface.load(tmd_file, read_image_layers=False)
    # Verify that the height map is non-empty.
    assert surface.data is not None
    assert surface.data.size > 0

    # Verify that expected metadata keys are present.
    metadata = surface.metadata
    for key in ['width', 'height', 'x_length', 'y_length', 'x_offset', 'y_offset', 'mmpp']:
        assert key in metadata, f"Metadata missing key: {key}"
