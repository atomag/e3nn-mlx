import inspect


def test_exports_parity():
    import importlib

    pairs = [
        ("e3nn.o3", "e3nn_mlx.o3"),
        ("e3nn.nn", "e3nn_mlx.nn"),
        ("e3nn.io", "e3nn_mlx.io"),
        ("e3nn.math", "e3nn_mlx.math"),
        ("e3nn.util", "e3nn_mlx.util"),
    ]

    for a, b in pairs:
        A = importlib.import_module(a)
        B = importlib.import_module(b)
        a_all = set(getattr(A, "__all__", []))
        b_all = set(getattr(B, "__all__", []))
        missing = sorted(a_all - b_all)
        assert not missing, f"Missing in {b}: {missing}"


def test_rotation_requires_grad_kw_present_and_callable():
    from e3nn_mlx import o3
    import mlx.core as mx

    funcs = [
        o3.identity_angles,
        o3.rand_angles,
        o3.rand_matrix,
        o3.identity_quaternion,
        o3.rand_quaternion,
        o3.rand_axis_angle,
    ]

    for f in funcs:
        sig = inspect.signature(f)
        assert (
            "requires_grad" in sig.parameters
        ), f"requires_grad kw missing on {f.__name__}"

    # sanity calls
    a, b, c = o3.identity_angles(2, requires_grad=True)
    assert a.shape == (2,) and b.shape == (2,) and c.shape == (2,)

    a, b, c = o3.rand_angles(3, requires_grad=True)
    assert a.shape == (3,) and b.shape == (3,) and c.shape == (3,)

    R = o3.rand_matrix(4, requires_grad=True)
    assert R.shape == (4, 3, 3)

    q = o3.identity_quaternion(2, requires_grad=True)
    assert q.shape == (2, 4)

    q = o3.rand_quaternion(5, requires_grad=True)
    assert q.shape == (5, 4)

    axis, angle = o3.rand_axis_angle(6, requires_grad=True)
    assert axis.shape == (6, 3) and angle.shape == (6,)


def test_util_torch_compat_wrappers():
    from e3nn_mlx import util
    dt = util.torch_get_default_tensor_type()
    dev = util.torch_get_default_device()
    assert dt is not None and dev is not None


def test_math_normalize_exports():
    from e3nn_mlx.math import normalize2mom, Normalize2Mom

    assert callable(normalize2mom)
    f = Normalize2Mom(lambda x: x)
    assert hasattr(f, "__call__")

