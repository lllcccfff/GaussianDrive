from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_EPS = 1e-9


def build_rotation(r):
    q = r / torch.norm(r, dim=-1, keepdim=True)

    R = torch.zeros((*q.shape[:-1], 3, 3)).to(r.device)

    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


# Refer to Street Gaussian by Yan et al. in 2024.
def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """

    def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
        """
        Returns torch.sqrt(torch.max(0, x))
        but with a zero subgradient where x is 0.
        """
        ret = torch.zeros_like(x)
        positive_mask = x > 0
        ret[positive_mask] = torch.sqrt(x[positive_mask])
        return ret

    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))


def slerp(v1, v2, t, DOT_THR=0.99975, to_cpu=False, dim=-1):
    """SLERP for pytorch tensors interpolating `v1` to `v2` with scale of `t`.

    `DOT_THR` determines when the vectors are too close to parallel.
        If they are too close, then a regular linear interpolation is used.

    `dim` is the feature dimension over which to compute norms and find angles.
        For example: if a sequence of 5 vectors is input with shape [5, 768]
        Then `dim = 1` or `dim = -1` computes SLERP along the feature dim of 768.

    Theory Reference:
    https://splines.readthedocs.io/en/latest/rotation/slerp.html
    PyTorch reference:
    https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
    Numpy reference:
    https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c
    """

    # take the dot product between normalized vectors
    v1_norm = v1 / torch.norm(v1, dim=dim, keepdim=True)
    v2_norm = v2 / torch.norm(v2, dim=dim, keepdim=True)
    dot = (v1_norm * v2_norm).sum(dim)

    # if the vectors are too close, return a simple linear interpolation
    if (torch.abs(dot) > DOT_THR).any():
        res = (1 - t) * v1 + t * v2
    else:  # else apply SLERP
        # compute the angle terms we need
        theta = torch.acos(dot)
        theta_t = theta * t
        sin_theta = torch.sin(theta)
        sin_theta_t = torch.sin(theta_t)

        # compute the sine scaling terms for the vectors
        s1 = torch.sin(theta - theta_t) / sin_theta
        s2 = sin_theta_t / sin_theta

        # interpolate the vectors
        res = (s1.unsqueeze(dim) * v1) + (s2.unsqueeze(dim) * v2)
    return res


class Trajectory:
    """Timestamp-keyed container supporting interpolation for scalars, vectors, or tensors."""

    def __init__(self, values: Optional[Dict[int, Iterable]] = None):
        self.trans: Dict[int, torch.Tensor] = {}
        self.quats: Dict[int, torch.Tensor] = {}
        self._sorted_ts: list[int] = []
        if values:
            for ts, value in values.items():
                self._add(ts, value)

    def __len__(self) -> int:
        return len(self._sorted_ts)

    def _add(self, timestamp: int, transform: Iterable) -> None:
        ts = int(timestamp)
        if ts not in self._sorted_ts:
            transform = torch.tensor(transform).float()

            self.trans[ts] = transform[:3, 3]
            self.quats[ts] = matrix_to_quaternion(transform[:3, :3])

            self._sorted_ts.append(ts)
            self._sorted_ts.sort()

    def get_transform(self, timestamp: int, allow_extrapolation: bool = True) -> torch.Tensor:
        transform = torch.eye(4)
        transform[:3, :3] = self.get_rotation(timestamp, allow_extrapolation=allow_extrapolation)
        transform[:3, 3] = self.get_translation(timestamp, allow_extrapolation=allow_extrapolation)
        return transform

    def get_rotation(self, timestamp: int, allow_extrapolation: bool = True) -> torch.Tensor:
        return build_rotation(self.get_quaternion(timestamp, allow_extrapolation=allow_extrapolation))

    def get_quaternion(self, timestamp: int, allow_extrapolation: bool = True) -> torch.Tensor:
        ts = int(timestamp)
        t_prev, t_next, ratio = self._sample(ts, allow_extrapolation=allow_extrapolation)
        if ratio == 0:
            return self.quats[t_prev]
        elif ratio == 1:
            return self.quats[t_next]
        else:
            return slerp(self.quats[t_prev], self.quats[t_next], ratio, to_cpu=True)

    def get_translation(self, timestamp: int, allow_extrapolation: bool = True) -> torch.Tensor:
        ts = int(timestamp)
        t_prev, t_next, ratio = self._sample(ts, allow_extrapolation=allow_extrapolation)
        if ratio == 0:
            return self.trans[t_prev]
        elif ratio == 1:
            return self.trans[t_next]
        else:
            return (1 - ratio) * self.trans[t_prev] + ratio * self.trans[t_next]

    def _sample(self, ts: int, clamp: bool = False, allow_extrapolation: bool = False) -> Tuple[int, int, float]:
        if not self._sorted_ts:
            raise ValueError("Empty.")
        elif len(self._sorted_ts) == 1:
            return self._sorted_ts[0], self._sorted_ts[0], 0.0

        if ts < self._sorted_ts[0]:
            if allow_extrapolation:
                t_prev = self._sorted_ts[0]
                t_next = self._sorted_ts[1]
                ratio = (ts - t_prev) / (t_next - t_prev)
                return t_prev, t_next, ratio
            if not clamp:
                raise ValueError(f"Timestamp {ts} precedes first value.")
            ts = self._sorted_ts[0]
        if ts > self._sorted_ts[-1]:
            if allow_extrapolation:
                t_prev = self._sorted_ts[-2]
                t_next = self._sorted_ts[-1]
                ratio = (ts - t_prev) / (t_next - t_prev)
                return t_prev, t_next, ratio
            if not clamp:
                raise ValueError(f"Timestamp {ts} exceeds last value.")
            ts = self._sorted_ts[-1]

        for idx, t_prev in enumerate(self._sorted_ts[:-1]):
            t_next = self._sorted_ts[idx + 1]
            if t_prev <= ts <= t_next:
                ratio = (ts - t_prev) / (t_next - t_prev)
                return t_prev, t_next, ratio

        raise ValueError(f"Timestamp {ts} could not be indexing.")

    def sub_trajectory(self, start: int, end: int) -> "Trajectory":
        sub_values = {ts: self.get_transform(ts) for ts in self._sorted_ts if start <= ts <= end}
        if not sub_values:
            raise ValueError("No values found in the specified range.")
        return Trajectory(sub_values)

    @property
    def start(self) -> int:
        if not self._sorted_ts:
            raise ValueError("Interpolatable is empty.")
        return self._sorted_ts[0]

    @property
    def end(self) -> int:
        if not self._sorted_ts:
            raise ValueError("Interpolatable is empty.")
        return self._sorted_ts[-1]
