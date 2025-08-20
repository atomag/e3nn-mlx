"""
O(3) rotation utilities for MLX with e3nn-compatible conventions.

Conventions follow e3nn (0.5.x):
- Euler composition: angles_to_matrix(alpha, beta, gamma) = Y(alpha) @ X(beta) @ Y(gamma)
- Quaternions stored as [w, x, y, z]
"""

import mlx.core as mx


# ---------------------
# Helper small utilities
# ---------------------

def _ensure_last_two(R: mx.array) -> mx.array:
    """Ensure input has shape (..., 3, 3)."""
    assert R.shape[-2:] == (3, 3), f"Expected (...,3,3), got {R.shape}"
    return R


# ---------------------
# Single-axis rotations
# ---------------------

def matrix_x(angle: mx.array) -> mx.array:
    c = mx.cos(angle)
    s = mx.sin(angle)
    o = mx.ones_like(angle)
    z = mx.zeros_like(angle)
    return mx.stack(
        [
            mx.stack([o, z, z], axis=-1),
            mx.stack([z, c, -s], axis=-1),
            mx.stack([z, s, c], axis=-1),
        ],
        axis=-2,
    )


def matrix_y(angle: mx.array) -> mx.array:
    c = mx.cos(angle)
    s = mx.sin(angle)
    o = mx.ones_like(angle)
    z = mx.zeros_like(angle)
    return mx.stack(
        [
            mx.stack([c, z, s], axis=-1),
            mx.stack([z, o, z], axis=-1),
            mx.stack([-s, z, c], axis=-1),
        ],
        axis=-2,
    )


def matrix_z(angle: mx.array) -> mx.array:
    c = mx.cos(angle)
    s = mx.sin(angle)
    o = mx.ones_like(angle)
    z = mx.zeros_like(angle)
    return mx.stack(
        [
            mx.stack([c, -s, z], axis=-1),
            mx.stack([s, c, z], axis=-1),
            mx.stack([z, z, o], axis=-1),
        ],
        axis=-2,
    )


# --------------
# Euler <-> Mat
# --------------

def angles_to_matrix(alpha: mx.array, beta: mx.array, gamma: mx.array) -> mx.array:
    """Convert Euler angles to rotation matrix using Y(alpha) X(beta) Y(gamma)."""
    return mx.matmul(matrix_y(alpha), mx.matmul(matrix_x(beta), matrix_y(gamma)))


def xyz_to_angles(xyz: mx.array):
    """Convert Cartesian unit vector to (alpha, beta) using e3nn convention.

    - beta from y component: y = cos(beta)
    - alpha from XZ plane: alpha = atan2(x, z)
    """
    # Normalize to avoid NaNs on zero-radius vectors
    r = mx.maximum(mx.linalg.norm(xyz, axis=-1, keepdims=True), 1e-12)
    xyz = xyz / r
    xyz = mx.clip(xyz, -1.0, 1.0)
    beta = mx.arccos(xyz[..., 1])
    alpha = mx.arctan2(xyz[..., 0], xyz[..., 2])
    return alpha, beta


def angles_to_xyz(alpha: mx.array, beta: mx.array) -> mx.array:
    """Unit vector from angles using e3nn convention.

    y = cos(beta), (z, x) = (cos(alpha), sin(alpha)) * sin(beta)
    """
    sa, ca = mx.sin(alpha), mx.cos(alpha)
    sb, cb = mx.sin(beta), mx.cos(beta)
    x = sb * sa
    y = cb
    z = sb * ca
    return mx.stack([x, y, z], axis=-1)


def matrix_to_angles(R: mx.array):
    """Recover (alpha, beta, gamma) from R using e3nn convention.

    Algorithm (matches e3nn):
    - x = R @ [0,1,0]
    - (alpha, beta) = xyz_to_angles(x)
    - R' = angles_to_matrix(alpha, beta, 0)^T @ R
    - gamma = atan2(R'[..., 0, 2], R'[..., 0, 0])
    """
    R = _ensure_last_two(R)
    # x = R @ [0,1,0]
    y_axis = mx.array([0.0, 1.0, 0.0], dtype=R.dtype)
    x = mx.matmul(R, y_axis)
    alpha, beta = xyz_to_angles(x)
    # R' = A^T @ R with A = angles_to_matrix(alpha,beta,0)
    A = angles_to_matrix(alpha, beta, mx.zeros_like(alpha))
    Rt = mx.swapaxes(A, -1, -2)
    Rp = mx.matmul(Rt, R)
    gamma = mx.arctan2(Rp[..., 0, 2], Rp[..., 0, 0])
    return alpha, beta, gamma


# -----------------
# Angles utilities
# -----------------

def identity_angles(*shape, dtype=mx.float32):
    z = mx.zeros(shape, dtype=dtype)
    return z, z, z


def rand_angles(*shape, dtype=mx.float32):
    two_pi = 2.0 * mx.array(mx.pi, dtype=dtype)
    # alpha, gamma uniform in [0, 2pi)
    u = mx.random.uniform(0.0, 1.0, (2,) + shape).astype(dtype)
    alpha = two_pi * u[0]
    gamma = two_pi * u[1]
    # beta: acos of uniform(-1,1)
    v = mx.random.uniform(-1.0, 1.0, shape).astype(dtype)
    beta = mx.arccos(mx.clip(v, -1.0, 1.0))
    return alpha, beta, gamma


def compose_angles(a1, b1, c1, a2, b2, c2):
    R1 = angles_to_matrix(a1, b1, c1)
    R2 = angles_to_matrix(a2, b2, c2)
    return matrix_to_angles(mx.matmul(R1, R2))


def inverse_angles(a, b, c):
    return -c, -b, -a


# -----------------
# Quaternions utils
# -----------------

def axis_angle_to_quaternion(axis: mx.array, angle: mx.array) -> mx.array:
    axis = mx.array(axis)
    axis = axis / mx.maximum(mx.linalg.norm(axis, axis=-1, keepdims=True), 1e-12)
    half = 0.5 * angle
    w = mx.cos(half)
    s = mx.sin(half)
    v = axis * s[..., None]
    return mx.concatenate([w[..., None], v], axis=-1)


def quaternion_to_axis_angle(q: mx.array):
    q = q / mx.maximum(mx.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w = q[..., 0]
    v = q[..., 1:4]
    vnorm = mx.linalg.norm(v, axis=-1)
    angle = 2.0 * mx.arctan2(vnorm, w)
    # Avoid division by zero; default axis = [1,0,0] when angle ~ 0
    safe = vnorm > 1e-12
    axis = mx.where(safe[..., None], v / vnorm[..., None], mx.array([1.0, 0.0, 0.0], dtype=q.dtype))
    return axis, angle


def quaternion_to_matrix(q: mx.array) -> mx.array:
    q = q / mx.maximum(mx.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    r00 = ww + xx - yy - zz
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)
    r10 = 2 * (xy + wz)
    r11 = ww - xx + yy - zz
    r12 = 2 * (yz - wx)
    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = ww - xx - yy + zz
    row0 = mx.stack([r00, r01, r02], axis=-1)
    row1 = mx.stack([r10, r11, r12], axis=-1)
    row2 = mx.stack([r20, r21, r22], axis=-1)
    return mx.stack([row0, row1, row2], axis=-2)


def matrix_to_quaternion(R: mx.array) -> mx.array:
    # Robust conversion via axis-angle
    axis, angle = matrix_to_axis_angle(R)
    return axis_angle_to_quaternion(axis, angle)


def angles_to_quaternion(alpha: mx.array, beta: mx.array, gamma: mx.array) -> mx.array:
    qa = axis_angle_to_quaternion(mx.array([0.0, 1.0, 0.0], dtype=alpha.dtype), alpha)
    qb = axis_angle_to_quaternion(mx.array([1.0, 0.0, 0.0], dtype=alpha.dtype), beta)
    qc = axis_angle_to_quaternion(mx.array([0.0, 1.0, 0.0], dtype=alpha.dtype), gamma)
    return compose_quaternion(qa, compose_quaternion(qb, qc))


def quaternion_to_angles(q: mx.array):
    return matrix_to_angles(quaternion_to_matrix(q))


def compose_quaternion(q1: mx.array, q2: mx.array) -> mx.array:
    # Hamilton product, [w,x,y,z]
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return mx.stack([w, x, y, z], axis=-1)


def inverse_quaternion(q: mx.array) -> mx.array:
    # Unit quaternion inverse
    return mx.concatenate([q[..., 0:1], -q[..., 1:4]], axis=-1)


def identity_quaternion(*shape, dtype=mx.float32) -> mx.array:
    q = mx.zeros(shape + (4,), dtype=dtype)
    # Set w=1
    if q.ndim == 1:  # shape == (4,)
        q = q.at[0].set(1.0)
    else:
        # Broadcast-safe set first component
        ones = mx.ones(shape, dtype=dtype)
        q = mx.concatenate([ones[..., None], q[..., 1:]], axis=-1)
    return q


def rand_quaternion(*shape, dtype=mx.float32) -> mx.array:
    a, b, c = rand_angles(*shape, dtype=dtype)
    return angles_to_quaternion(a, b, c)


# -----------------
# Axis-angle utils
# -----------------

def axis_angle_to_matrix(axis: mx.array, angle: mx.array) -> mx.array:
    axis = mx.array(axis)
    axis = axis / mx.maximum(mx.linalg.norm(axis, axis=-1, keepdims=True), 1e-12)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c = mx.cos(angle)
    s = mx.sin(angle)
    t = 1.0 - c
    r00 = t * x * x + c
    r01 = t * x * y - s * z
    r02 = t * x * z + s * y
    r10 = t * x * y + s * z
    r11 = t * y * y + c
    r12 = t * y * z - s * x
    r20 = t * x * z - s * y
    r21 = t * y * z + s * x
    r22 = t * z * z + c
    row0 = mx.stack([r00, r01, r02], axis=-1)
    row1 = mx.stack([r10, r11, r12], axis=-1)
    row2 = mx.stack([r20, r21, r22], axis=-1)
    return mx.stack([row0, row1, row2], axis=-2)


def matrix_to_axis_angle(R: mx.array):
    R = _ensure_last_two(R)
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    # angle in [0, pi]
    angle = mx.arccos(mx.clip(0.5 * (trace - 1.0), -1.0, 1.0))
    # For small angles, default axis
    eps = 1e-7
    sin_angle = mx.sin(angle)
    # Compute axis via anti-symmetric part
    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]
    axis_unnorm = mx.stack([rx, ry, rz], axis=-1)
    axis = axis_unnorm / mx.maximum(2.0 * sin_angle[..., None], eps)
    # Handle angle ~ 0: choose any axis (e.g., [1,0,0])
    near_zero = (mx.abs(angle) < 1e-6)
    default_axis = mx.array([1.0, 0.0, 0.0], dtype=R.dtype)
    axis = mx.where(near_zero[..., None], default_axis, axis)
    return axis, angle


def angles_to_axis_angle(alpha: mx.array, beta: mx.array, gamma: mx.array):
    q = angles_to_quaternion(alpha, beta, gamma)
    return quaternion_to_axis_angle(q)


def axis_angle_to_angles(axis: mx.array, angle: mx.array):
    return quaternion_to_angles(axis_angle_to_quaternion(axis, angle))


# --------------
# Matrix helpers
# --------------

def identity(dtype=mx.float32) -> mx.array:
    return mx.eye(3, dtype=dtype)


def rand(dtype=mx.float32) -> mx.array:
    return rand_matrix(dtype=dtype)


def rand_matrix(*shape, dtype=mx.float32) -> mx.array:
    a, b, c = rand_angles(*shape, dtype=dtype)
    return angles_to_matrix(a, b, c)


def axis_angle(axis: mx.array, angle: mx.array) -> mx.array:
    return axis_angle_to_matrix(axis, angle)
