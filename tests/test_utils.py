import numpy as np
from surfalize.utils import is_list_like
def test_is_list_like():
    assert is_list_like([])
    assert is_list_like((1, 2))
    assert is_list_like(np.array([1, 2, 3]))
    assert not is_list_like(1)
    assert not is_list_like(range(4))