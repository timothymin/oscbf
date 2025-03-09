"""OSCBF for velocity-controlled robots"""

import time
from typing import Optional
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array
import numpy as np
from cbfpy import CBF, CBFConfig
from oscbf.core.manipulator import Manipulator, load_panda
from oscbf.core.manipulation_env import FrankaVelocityControlEnv
from oscbf.utils.controllers import orientation_error_3D, _format_gain, _format_limit
from oscbf.core.franka_collision_model import collision_data


class OSCBFVelocityConfig(CBFConfig):
    """CBF Configuration for safe velocity-controlled manipulation

    z = [q] (length = num_joints)
    u = [joint velocities] (length = num_joints)
    """

    def __init__(self, robot: Manipulator):
        self.robot = robot
        self.num_joints = self.robot.num_joints
        self.task_dim = 6  # Pose
        self.is_redundant = self.robot.num_joints > self.task_dim

        # TODO reorganize where these go??
        self.pos_obj_weight = 1.0
        self.rot_obj_weight = 1.0
        self.joint_space_obj_weight = 1.0
        # Store the diagonal of W.T @ W for the task and joint space weighting matrices
        # This assumes that W is a diagonal matrix with only positive values
        self.W_T_W_task_diag = tuple(
            np.array([self.pos_obj_weight] * 3 + [self.rot_obj_weight] * 3) ** 2
        )
        self.W_T_W_joint_diag = tuple(
            np.array([self.joint_space_obj_weight] * self.num_joints) ** 2
        )

        super().__init__(
            n=self.num_joints,
            m=self.num_joints,
            u_min=-np.asarray(robot.joint_max_velocities),
            u_max=np.asarray(robot.joint_max_velocities),
        )

    def f(self, z, **kwargs):
        return jnp.zeros(self.n)

    def g(self, z, **kwargs):
        return jnp.eye(self.num_joints)

    def _P(self, z):
        q = z
        transforms = self.robot.joint_to_world_transforms(q)
        J = self.robot._ee_jacobian(transforms)
        M = self.robot._mass_matrix(transforms)
        M_inv = jnp.linalg.inv(M)
        task_inertia_inv = J @ M_inv @ J.T
        task_inertia = jnp.linalg.inv(task_inertia_inv)
        J_bar = M_inv @ J.T @ task_inertia
        J_hash = J_bar
        N = jnp.eye(self.num_joints) - J_hash @ J
        W_T_W_joint = jnp.diag(jnp.asarray(self.W_T_W_joint_diag))
        W_T_W_task = jnp.diag(jnp.asarray(self.W_T_W_task_diag))

        return N.T @ W_T_W_joint @ N + J.T @ W_T_W_task @ J

    def P(self, z, u_des, **kwargs):
        return self._P(z)

    def q(self, z, u_des, **kwargs):
        return -u_des.T @ self._P(z)
