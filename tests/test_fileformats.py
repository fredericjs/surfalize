import pytest
from pathlib import Path
from surfalize import Surface
from surfalize.file import supported_formats

@pytest.fixture
def testfile_dir():
    return Path('./tests/test_files')

@pytest.mark.parametrize('fileformat', supported_formats)
def test_fileformat_loading(testfile_dir, fileformat):
    files = list(testfile_dir.glob(f'*{fileformat}'))
    if not files:
        pytest.skip('No testfiles found.')
    for file in files:
        Surface.load(file)