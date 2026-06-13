import io
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from surfalize import Surface
from surfalize.batch import Batch, FilenameParser, _Parameter, _Operation, _Token, FileInput
from surfalize.exceptions import BatchError

module_path = Path(__file__).parent

TOKENS = [
    _Token('power|float|P'),
    _Token('pulses|int|N'),
    _Token('fluence|float|F'),
    _Token('frequency|float|FREP|kHz')
]
SEPARATORS = ['', '_', '_', '_', '']

@pytest.fixture
def regex():
    r = 'P(?P<power>\\d+(?:(?:\\.|,)\\d+)?)_N(?P<pulses>\\d+)_F(?P<fluence>\\d+(?:(?:\\.|,)\\d+)?)_'
    r += 'FREP(?P<frequency>\\d+(?:(?:\\.|,)\\d+)?)kHz'
    return r

@pytest.fixture
def template_string():
    return '<power|float|P>_<pulses|int|N>_<fluence|float|F>_<frequency|float|FREP|kHz>'

@pytest.fixture
def dataframe():
    data = [
        'P90_N10_F1.21_FREP10kHz.vk4',
        'P80_N10_F1.21_FREP10kHz.vk4',
        'P70_N10_F1.21_FREP10kHz.vk4'
    ]
    return pd.DataFrame(data, columns=['file'])

@pytest.fixture
def expected_dataframe_output():
    data = [
        [
            'P90_N10_F1.21_FREP10kHz.vk4',
            'P80_N10_F1.21_FREP10kHz.vk4',
            'P70_N10_F1.21_FREP10kHz.vk4'
        ],
        [90., 80., 70.], [10, 10, 10], [1.21, 1.21, 1.21], [10., 10., 10.]
    ]
    columns = ['file', 'power', 'pulses', 'fluence', 'frequency']
    return pd.DataFrame(dict(zip(columns, data)))


class TestFilenameParser:

    def test_parse_template(self, template_string):
        parser = FilenameParser(template_string)
        tokens, separators = parser.parse_template()
        assert tokens == TOKENS
        assert separators == SEPARATORS

    def test_construct_regex(self, template_string, regex):
        parser = FilenameParser(template_string)
        regex_result = parser.construct_regex(TOKENS, SEPARATORS)
        assert regex_result == regex

    def test_extract_from(self, template_string, dataframe, expected_dataframe_output):
        parser = FilenameParser(template_string)
        df = parser.extract_from(dataframe, 'file')
        expected = expected_dataframe_output.drop('file', axis=1)
        assert_frame_equal(df, expected, check_dtype=False)

def test_batch_from_dir():
    batch = Batch.from_dir(module_path / 'test_files')
    assert set(batch._files) == set((module_path / 'test_files').iterdir())


# Batch execution #####################################################################################################

def _make_periodic_surface(scale=1.0, n=128, period_px=16):
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    z = scale * np.sin(2 * np.pi * x / period_px)
    return Surface(z, 0.1, 0.1)

@pytest.fixture
def batch_dir(tmp_path):
    # Three periodic surfaces of increasing amplitude, saved as .sur
    for i in range(1, 4):
        _make_periodic_surface(scale=i).save(tmp_path / f'S{i}_surface.sur')
    return tmp_path

def test_batch_execute(batch_dir):
    batch = Batch.from_dir(batch_dir, file_extensions='.sur')
    batch.level().Sa().Sq()
    df = batch.execute(multiprocessing=False).get_dataframe()
    assert len(df) == 3
    assert {'file', 'Sa', 'Sq'} <= set(df.columns)
    assert df['Sa'].notna().all()
    # Batch result matches a direct Surface computation of the same file and operations
    direct = Surface.load(batch_dir / 'S1_surface.sur').level()
    row = df[df['file'] == 'S1_surface.sur'].iloc[0]
    assert row['Sa'] == pytest.approx(direct.Sa())
    assert row['Sq'] == pytest.approx(direct.Sq())

def test_batch_execute_multiprocessing(batch_dir):
    # Exercises the ThreadPool dispatch branch (default multiprocessing=True)
    batch = Batch.from_dir(batch_dir, file_extensions='.sur')
    batch.Sa()
    df = batch.execute().get_dataframe()
    assert len(df) == 3
    assert df['Sa'].notna().all()

def test_batch_multiple_return_values(batch_dir):
    # depth returns (mean, std) -> two labelled columns
    batch = Batch.from_dir(batch_dir, file_extensions='.sur')
    batch.depth()
    df = batch.execute(multiprocessing=False).get_dataframe()
    assert {'depth_mean', 'depth_std'} <= set(df.columns)

def test_batch_custom_name(batch_dir):
    batch = Batch.from_dir(batch_dir, file_extensions='.sur')
    batch.Sa(custom_name='roughness')
    df = batch.execute(multiprocessing=False).get_dataframe()
    assert 'roughness' in df.columns
    assert 'Sa' not in df.columns

def test_batch_duplicate_parameter_raises(batch_dir):
    batch = Batch.from_dir(batch_dir, file_extensions='.sur')
    batch.Sa()
    with pytest.raises(BatchError):
        batch.Sa()

def test_batch_no_steps_raises(batch_dir):
    batch = Batch.from_dir(batch_dir, file_extensions='.sur')
    with pytest.raises(BatchError):
        batch.execute(multiprocessing=False)

def test_batch_add_files(batch_dir):
    files = list(batch_dir.glob('*.sur'))
    batch = Batch([])
    assert len(batch) == 0
    batch.add_files(files)
    assert len(batch) == len(files)

def test_batch_add_dir(batch_dir):
    batch = Batch([])
    batch.add_dir(batch_dir, file_extensions='.sur')
    assert len(batch) == 3

def test_batch_fileinput_buffer(batch_dir):
    path = batch_dir / 'S1_surface.sur'
    buffer = io.BytesIO(path.read_bytes())
    batch = Batch([FileInput(name='S1_surface.sur', data=buffer, format='.sur')])
    batch.Sa()
    df = batch.execute(multiprocessing=False).get_dataframe()
    assert len(df) == 1
    assert df.iloc[0]['file'] == 'S1_surface.sur'

def test_batch_extract_from_filename(batch_dir):
    batch = Batch.from_dir(batch_dir, file_extensions='.sur')
    batch.Sa().extract_from_filename('<sample|int|S>_surface')
    df = batch.execute(multiprocessing=False).get_dataframe()
    assert 'sample' in df.columns
    assert sorted(df['sample'].tolist()) == [1, 2, 3]

def test_batch_preserve_chaining_order_false(batch_dir):
    # Parameter registered before the operation; with preserve_chaining_order=False the operation still runs first
    batch = Batch.from_dir(batch_dir, file_extensions='.sur')
    batch.Sa(custom_name='Sa_leveled').level()
    df = batch.execute(multiprocessing=False, preserve_chaining_order=False).get_dataframe()
    assert 'Sa_leveled' in df.columns
    assert df['Sa_leveled'].notna().all()

def test_batch_on_file_complete(batch_dir):
    seen = []
    batch = Batch.from_dir(batch_dir, file_extensions='.sur')
    batch.Sa()
    batch.execute(multiprocessing=False, on_file_complete=lambda r: seen.append(r['file']))
    assert len(seen) == 3

def test_batch_custom_operation_and_parameter(batch_dir):
    batch = Batch.from_dir(batch_dir, file_extensions='.sur')
    batch.custom_operation(lambda s: s.zero(inplace=True))
    # A custom parameter function must return a dict of {column: value}
    batch.custom_parameter(lambda s: {'min_height': float(s.data.min())})
    df = batch.execute(multiprocessing=False).get_dataframe()
    assert len(df) == 3
    # After zero(), the minimum height is 0 for every file
    assert all(abs(v) < 1e-6 for v in df['min_height'])

def test_batch_saveto(batch_dir, tmp_path):
    batch = Batch.from_dir(batch_dir, file_extensions='.sur')
    batch.Sa()
    out = tmp_path / 'results.xlsx'
    batch.execute(multiprocessing=False, saveto=out)
    assert out.exists()
    saved = pd.read_excel(out)
    assert {'file', 'Sa'} <= set(saved.columns)


