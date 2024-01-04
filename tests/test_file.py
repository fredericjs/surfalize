import pytest
import io
from surfalize.file.common import _sanitize_mu, get_unit_conversion, read_binary_layout

@pytest.mark.parametrize('test_input, expected',
                         [('µm', 'um'),
                         ('123µ', '123u'),
                         ('\u03bcm', 'um'),
                         (f'{chr(956)}m', 'um'),
                         (f'{chr(13211)}', 'um')],
                    )
def test_sanitize_mu(test_input, expected):
    assert _sanitize_mu(test_input) == expected

@pytest.mark.parametrize('par1, par2, expected', [('um', 'mm', 0.001), ('mm', 'pm', 10**9), ('nm', 'um', 0.001)])
def test_get_unit_conversion(par1, par2, expected):
    assert get_unit_conversion(par1, par2) == expected


@pytest.fixture
def binary_file_like_object():
    binary_data = (b"\x48\x65\x6c\x6c"
                   b"\xb6\xf3\x9d\x3f"
                   b"\x00\x00\x00"
                   b"\x00\x01")
    return io.BytesIO(binary_data)

@pytest.fixture
def layout():
    return [('field1', 'I', False), ('field2', 'f', False), (None, 3, None), ('field3', 'h', False)]

def test_read_binary_layout(binary_file_like_object, layout):
    result = read_binary_layout(binary_file_like_object, layout)
    assert result == {'field1': 1819043144, 'field2': pytest.approx(1.234), 'field3': 256}

