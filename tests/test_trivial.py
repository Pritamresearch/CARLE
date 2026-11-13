def test_environment_trivial():
    import numpy as np
    import torch
    import zstandard
    import lz4
    import constriction

    assert np.__version__ is not None
    assert torch.__version__ is not None
    assert zstandard.__version__ is not None
