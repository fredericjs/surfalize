import pytest
from pathlib import Path
from datetime import datetime
import hashlib
import io
import zipfile
import xml.etree.ElementTree as ElementTree
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


def test_hdat_reading(testfile_dir):
    surface = Surface.load(testfile_dir / 'test_1.hdat', read_image_layers=True)
    assert surface.size == (542, 512)
    assert surface.step_x == surface.step_y == pytest.approx(0.24)
    # Heights are stored as absolute scanner positions in micrometers
    assert np.nanmean(surface.data) == pytest.approx(-20742.07, abs=0.01)
    assert set(surface.image_layers) == {'Confidence', 'pZenvelope', 'pP', 'Phase', 'Amplitude'}
    # Phase is stored as fixpoint integer with a scaling factor that maps it onto radians
    assert surface.image_layers['Phase'].data.max() == pytest.approx(2 * np.pi, abs=1e-2)
    assert surface.metadata['scanner/scanner type'] == 'heliInspectH8'
    assert surface.metadata['timestamp'].year == 2026


# X3P (ISO 25178-72) writer. X3P stores all lengths in meters, while surfalize works in micrometers, so the written
# point data is the surface data divided by 1e6. The written doubles are exactly these converted values (checked
# bit-exactly against data.bin below). Converting back to micrometers on reading can differ from the original value by
# at most one unit in the last place, since a multiplication by 1e-6 followed by 1e6 is not always exact in floating
# point arithmetic.
X3P_NAMESPACE = 'http://www.opengps.eu/2008/ISO5436_2'
# Surfaces with anisotropic steps emit a warning on construction and on loading
ignore_anisotropy_warning = pytest.mark.filterwarnings('ignore:The surface has different pixel size')

def _x3p_surface(nx=37, ny=23, step_x=0.1, step_y=0.1, nan_positions=((0, 0), (5, 7), (-1, -1))):
    rng = np.random.default_rng(42)
    data = rng.normal(size=(ny, nx)) * 3.7
    # Make the surface asymmetric so that a transposition or flip would be detected
    data += np.arange(nx)[np.newaxis, :] * 0.01 + np.arange(ny)[:, np.newaxis] * 0.1
    for y, x in nan_positions:
        data[y, x] = np.nan
    return Surface(data, step_x, step_y)

def _write_x3p(surface, **kwargs):
    buffer = io.BytesIO()
    surface.save(buffer, format='.x3p', **kwargs)
    return buffer

def _read_archive(buffer):
    with zipfile.ZipFile(buffer) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    buffer.seek(0)
    return contents

@ignore_anisotropy_warning
@pytest.mark.parametrize('nx, ny, step_x, step_y', [
    (37, 23, 0.25, 0.4),       # non-square, anisotropic
    (23, 37, 0.1, 0.1),        # non-square, transposed shape
    (16, 16, 1.234567, 0.08),  # square, anisotropic with steps not exactly representable in meters
])
def test_x3p_roundtrip(nx, ny, step_x, step_y):
    surface = _x3p_surface(nx, ny, step_x, step_y)
    loaded = Surface.load(_write_x3p(surface))
    assert loaded.size == surface.size
    # NaN marks non-measured points and must survive at exactly the same positions
    assert np.array_equal(np.isnan(loaded.data), np.isnan(surface.data))
    assert np.isnan(loaded.data).sum() == 3
    valid = ~np.isnan(surface.data)
    np.testing.assert_array_max_ulp(loaded.data[valid], surface.data[valid], maxulp=1)
    assert loaded.step_x == pytest.approx(step_x, rel=1e-15)
    assert loaded.step_y == pytest.approx(step_y, rel=1e-15)

def test_x3p_roundtrip_to_path(tmp_path):
    surface = _x3p_surface()
    path = tmp_path / 'surface.x3p'
    surface.save(path)
    loaded = Surface.load(path)
    assert np.array_equal(np.isnan(loaded.data), np.isnan(surface.data))
    assert loaded.size == surface.size

def test_x3p_double_data_is_stored_bit_exact():
    surface = _x3p_surface(nx=37, ny=23)
    data_bin = _read_archive(_write_x3p(surface))['bindata/data.bin']
    expected = (np.asarray(surface.data, dtype=np.float64) / 1e6).astype('<f8')
    assert data_bin == expected.tobytes()
    # X is the fastest varying index: the second stored value is the second point of the first line
    stored = np.frombuffer(data_bin, dtype='<f8')
    assert stored[1] == expected[0, 1]
    assert stored[surface.size.x] == expected[1, 0]

def test_x3p_reference_file_roundtrip_is_bit_exact(testfile_dir):
    # The reference file stores doubles in meters. Converting them to micrometers and back reproduces them exactly, so
    # writing the file again must yield an identical data.bin as well as identical data, steps and metadata.
    original_path = testfile_dir / 'test_1.x3p'
    surface = Surface.load(original_path)
    buffer = _write_x3p(surface)
    with zipfile.ZipFile(original_path) as original:
        assert original.read('bindata/data.bin') == _read_archive(buffer)['bindata/data.bin']
    loaded = Surface.load(buffer)
    assert np.array_equal(loaded.data, surface.data, equal_nan=True)
    assert loaded.step_x == surface.step_x
    assert loaded.step_y == surface.step_y
    assert loaded.metadata == surface.metadata

def test_x3p_float_dtype():
    surface = _x3p_surface()
    buffer = _write_x3p(surface, dtype='F')
    contents = _read_archive(buffer)
    assert len(contents['bindata/data.bin']) == surface.size.x * surface.size.y * 4
    assert ElementTree.fromstring(contents['main.xml']).find('Record1/Axes/CZ/DataType').text == 'F'
    loaded = Surface.load(buffer)
    assert np.array_equal(np.isnan(loaded.data), np.isnan(surface.data))
    valid = ~np.isnan(surface.data)
    np.testing.assert_allclose(loaded.data[valid], surface.data[valid], rtol=1e-6, atol=1e-6)

def test_x3p_invalid_dtype_raises():
    with pytest.raises(ValueError):
        _write_x3p(_x3p_surface(), dtype='I')

def test_x3p_checksums():
    contents = _read_archive(_write_x3p(_x3p_surface()))
    assert set(contents) == {'main.xml', 'bindata/data.bin', 'md5checksum.hex'}
    main_xml = contents['main.xml']
    # md5checksum.hex has the format '<hex digest> *main.xml'
    assert contents['md5checksum.hex'].decode('ascii').strip() == f'{hashlib.md5(main_xml).hexdigest()} *main.xml'
    root = ElementTree.fromstring(main_xml)
    assert root.find('Record4/ChecksumFile').text == 'md5checksum.hex'
    assert root.find('Record3/DataLink/PointDataLink').text == 'bindata/data.bin'
    stored_checksum = root.find('Record3/DataLink/MD5ChecksumPointData').text
    assert stored_checksum.lower() == hashlib.md5(contents['bindata/data.bin']).hexdigest()

def test_x3p_xml_structure():
    main_xml = _read_archive(_write_x3p(_x3p_surface(nx=37, ny=23)))['main.xml']
    assert main_xml.startswith(b'<?xml version="1.0" encoding="UTF-8"')
    # Parsing fails if the document is not well-formed
    root = ElementTree.fromstring(main_xml)
    assert root.tag == f'{{{X3P_NAMESPACE}}}ISO5436_2'
    assert [child.tag for child in root] == ['Record1', 'Record2', 'Record3', 'Record4']

    record1 = root.find('Record1')
    assert [child.tag for child in record1] == ['Revision', 'FeatureType', 'Axes']
    assert record1.find('Revision').text == 'ISO5436 - 2000'
    assert record1.find('FeatureType').text == 'SUR'
    axes = record1.find('Axes')
    assert [child.tag for child in axes] == ['CX', 'CY', 'CZ']
    for name in ('CX', 'CY', 'CZ'):
        assert [child.tag for child in axes.find(name)] == ['AxisType', 'DataType', 'Increment', 'Offset']
    for name in ('CX', 'CY'):
        assert axes.find(f'{name}/AxisType').text == 'I'
        # Lengths are stored in meters
        assert float(axes.find(f'{name}/Increment').text) == pytest.approx(0.1e-6, rel=1e-15)
    assert axes.find('CZ/AxisType').text == 'A'
    assert axes.find('CZ/DataType').text == 'D'

    record2 = root.find('Record2')
    assert [child.tag for child in record2] == ['Date', 'Instrument', 'CalibrationDate', 'ProbingSystem', 'Comment']
    assert [child.tag for child in record2.find('Instrument')] == ['Manufacturer', 'Model', 'Serial', 'Version']
    assert [child.tag for child in record2.find('ProbingSystem')] == ['Type', 'Identification']
    assert record2.find('ProbingSystem/Type').text in ('Contacting', 'NonContacting', 'Software')
    # The dates must be valid ISO 8601 timestamps
    datetime.fromisoformat(record2.find('Date').text)
    datetime.fromisoformat(record2.find('CalibrationDate').text)

    record3 = root.find('Record3')
    assert [child.tag for child in record3] == ['MatrixDimension', 'DataLink']
    assert [child.tag for child in record3.find('DataLink')] == ['PointDataLink', 'MD5ChecksumPointData']
    assert record3.find('MatrixDimension/SizeX').text == '37'
    assert record3.find('MatrixDimension/SizeY').text == '23'
    assert record3.find('MatrixDimension/SizeZ').text == '1'

def test_x3p_metadata():
    surface = _x3p_surface()
    surface.metadata = {
        'timestamp': datetime(2024, 5, 17, 9, 30, 15),
        'InstrumentManufacturer': 'ACME & Sons <Metrology>',
        'ProbingSystemType': 'NonContacting',
    }
    loaded = Surface.load(_write_x3p(surface, comment='Some comment'))
    assert loaded.metadata['Date'] == datetime(2024, 5, 17, 9, 30, 15)
    assert loaded.metadata['CalibrationDate'] == datetime(2024, 5, 17, 9, 30, 15)
    assert loaded.metadata['InstrumentManufacturer'] == 'ACME & Sons <Metrology>'
    assert loaded.metadata['InstrumentModel'] == 'not available'
    assert loaded.metadata['ProbingSystemType'] == 'NonContacting'
    assert loaded.metadata['Comment'] == 'Some comment'
