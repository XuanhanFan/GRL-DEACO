#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for converting Blender quaternions into routing directions."""

import math
from typing import Tuple

import numpy as np


_EPSILON = 1e-8
_DEFAULT_DIRECTION = np.array([0.0, 1.0, 0.0], dtype=float)


class BlenderQuaternionConverter:
    """Convert Blender-style quaternion rotations to pipe-port directions.

    Blender stores quaternions in ``(w, x, y, z)`` order. In this project, GLB
    port nodes are interpreted as facing along their local positive Y axis, so
    the converter extracts the rotated local Y axis and returns it as a unit
    direction vector.
    """

    def blender_quaternion_to_direction_vector(
        self,
        w: float,
        x: float,
        y: float,
        z: float,
        relative_position: np.ndarray,
    ) -> np.ndarray:
        """
        Convert a Blender quaternion into a normalized direction vector.

        If the quaternion-derived direction is degenerate, the normalized port
        position relative to the device center is used as a fallback. If that is
        also degenerate, the default local +Y direction is returned.

        Args:
            w: Real component of the Blender quaternion.
            x: X component of the Blender quaternion.
            y: Y component of the Blender quaternion.
            z: Z component of the Blender quaternion.
            relative_position: Port position relative to the device center.

        Returns:
            A unit direction vector as a NumPy array with shape ``(3,)``.
        """
        w, x, y, z = self._normalize_quaternion(w, x, y, z)

        rotation_matrix = self._quaternion_to_rotation_matrix(w, x, y, z)
        local_y_axis = rotation_matrix[:, 1]

        if np.linalg.norm(local_y_axis) >= _EPSILON:
            return self._normalize_vector(local_y_axis)

        relative_position = np.asarray(relative_position, dtype=float)
        if np.linalg.norm(relative_position) >= _EPSILON:
            return self._normalize_vector(relative_position)

        return _DEFAULT_DIRECTION.copy()

    @staticmethod
    def _normalize_quaternion(
        w: float,
        x: float,
        y: float,
        z: float,
    ) -> Tuple[float, float, float, float]:
        """Normalize a quaternion, falling back to identity for degenerate input."""
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm < _EPSILON:
            return 1.0, 0.0, 0.0, 0.0
        return w / norm, x / norm, y / norm, z / norm

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        """Return a unit vector, falling back to the default direction if needed."""
        norm = np.linalg.norm(vector)
        if norm < _EPSILON:
            return _DEFAULT_DIRECTION.copy()
        return vector / norm

    @staticmethod
    def _quaternion_to_rotation_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
        """
        Convert a unit quaternion into a 3x3 rotation matrix.

        Args:
            w: Real quaternion component.
            x: X quaternion component.
            y: Y quaternion component.
            z: Z quaternion component.

        Returns:
            A 3x3 rotation matrix.
        """
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        return np.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
            ],
            dtype=float,
        )

    def quaternion_to_euler_angles(
        self,
        w: float,
        x: float,
        y: float,
        z: float,
    ) -> Tuple[float, float, float]:
        """
        Convert a quaternion to XYZ Euler angles.

        Args:
            w: Real quaternion component.
            x: X quaternion component.
            y: Y quaternion component.
            z: Z quaternion component.

        Returns:
            ``(roll, pitch, yaw)`` in radians.
        """
        w, x, y, z = self._normalize_quaternion(w, x, y, z)

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def euler_angles_to_quaternion(
        self,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> Tuple[float, float, float, float]:
        """
        Convert XYZ Euler angles to a quaternion.

        Args:
            roll: X-axis rotation in radians.
            pitch: Y-axis rotation in radians.
            yaw: Z-axis rotation in radians.

        Returns:
            Quaternion components in ``(w, x, y, z)`` order.
        """
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return w, x, y, z
