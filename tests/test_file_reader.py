import pytest
from pathlib import Path

TESTFILE_DIR = Path() / 'test_files'

def get_testfiles(ext):
    return TESTFILE_DIR.glob(f'*{ext}')

def test_al3d():
    testfiles = get_testfiles('.al3d')
    for file in testfiles:
        pass

def test_nms():
    pass

def test_opd():
    pass

def test_vk4():
    pass

def test_vk6():
    pass

def test_vk7():
    pass

def test_plu():
    pass

def test_plux():
    pass

def test_zmg():
    pass