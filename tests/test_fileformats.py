import pytest
from pathlib import Path
import io
import numpy as np
from surfalize import Surface
from surfalize.file import supported_formats_read, supported_formats_write, reader_suffix_groups

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

# A single format may be registered under several suffixes (e.g. FITS as '.fits', '.fit', '.fts') that all map to
# the same reader. We therefore parametrize over format groups and consider a format tested if a testfile exists for
# any one of its suffixes, so that the remaining suffixes do not show up as skipped.
def _files_for_suffixes(testfile_dir, suffixes):
    files = []
    for suffix in suffixes:
        files.extend(testfile_dir.glob(f'*{suffix}'))
    return files

@pytest.mark.parametrize('suffixes', reader_suffix_groups, ids=lambda s: ','.join(s))
def test_fileformat_reading_from_path(testfile_dir, suffixes):
    files = _files_for_suffixes(testfile_dir, suffixes)
    if not files:
        pytest.skip('No testfiles found.')
    for file in files:
        Surface.load(file, read_image_layers=False)
        Surface.load(file, read_image_layers=True)

@pytest.mark.parametrize('suffixes', reader_suffix_groups, ids=lambda s: ','.join(s))
def test_fileformat_reading_from_buffer(testfile_dir, suffixes):
    files = _files_for_suffixes(testfile_dir, suffixes)
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