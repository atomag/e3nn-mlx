import pytest
import mlx.core as mx

# Suppress linter errors
float_tolerance = None


@pytest.fixture(autouse=True)
def set_random_seed() -> None:
    """Set the random seeds to try to get some reproducibility"""
    mx.random.seed(0)
    import numpy as np
    import random
    np.random.seed(0)
    random.seed(0)


@pytest.fixture(scope="session", autouse=True, params=["float32"])
def float_tolerance(request):
    """Run all tests with MLX default dtypes.
    
    This is a session-wide, autouse fixture — you only need to request it 
    explicitly if a test needs to know the tolerance for the current dtype.

    Returns
    --------
        A precision threshold to use for closeness tests.
    """
    if request.param == "float32":
        tolerance = 1e-3
    elif request.param == "float64":
        tolerance = 1e-9
    else:
        raise ValueError(f"Unknown dtype: {request.param}")
    
    yield tolerance