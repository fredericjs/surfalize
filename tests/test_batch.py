import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from surfalize.batch import Batch, FilenameParser, _Parameter, _Operation, _Token

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