import pytest
from pathlib import Path
from surfalize import Surface
from surfalize.file import supported_formats

module_path = Path(__file__).parent

@pytest.fixture
def testfile_dir():
    return module_path / 'test_files'

@pytest.mark.parametrize('fileformat', supported_formats)
def test_fileformat_loading(testfile_dir, fileformat):
    files = list(testfile_dir.glob(f'*{fileformat}'))
    if not files:
        pytest.skip('No testfiles found.')
    for file in files:
        Surface.load(file, read_image_layers=False)
        Surface.load(file, read_image_layers=True)