"""OSCBF for torque-controlled robots"""

from functools import partial
import time
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
from cbfpy import CBFConfig, CBF

from oscbf.core.manipulation_env import FrankaTorqueControlEnv
from oscbf.core.manipulator import Manipulator, load_panda
from oscbf.utils.controllers import PoseTaskTorqueController

USE_CENTRIFUGAL_CORIOLIS_IN_CBF = True


class OSCBFTorqueConfig(CBFConfig):
    """CBF Configuration for safe torque-controlled manipulation

    z = [q, qdot] (length = 2 * num_joints)
    u = [joint torques] (length = num_joints)
    """

    def __init__(
        self,
        robot: Manipulator,
        pos_obj_weight: float = 1.0,
        rot_obj_weight: float = 1.0,
        joint_obj_weight: float = 1.0,
    ):
        assert isinstance(robot, Manipulator)
        assert isinstance(pos_obj_weight, (tuple, float)) and pos_obj_weight >= 0
        assert isinstance(rot_obj_weight, (tuple, float)) and rot_obj_weight >= 0
        assert isinstance(joint_obj_weight, (tuple, float)) and joint_obj_weight >= 0
        self.robot = robot
        self.num_joints = self.robot.num_joints
        self.task_dim = 6  # Pose
        self.is_redundant = self.robot.num_joints > self.task_dim

        self.pos_obj_weight = float(pos_obj_weight)
        self.rot_obj_weight = float(rot_obj_weight)
        self.joint_space_obj_weight = float(joint_obj_weight)
        # Store the diagonal of W.T @ W for the task and joint space weighting matrices
        # This assumes that W is a diagonal matrix with only positive values
        self.W_T_W_task_diag = tuple(
            np.array([self.pos_obj_weight] * 3 + [self.rot_obj_weight] * 3) ** 2
        )
        self.W_T_W_joint_diag = tuple(
            np.array([self.joint_space_obj_weight] * self.num_joints) ** 2
        )

        super().__init__(
            n=self.num_joints * 2,
            m=self.num_joints,
            u_min=-np.asarray(robot.joint_max_forces),
            u_max=np.asarray(robot.joint_max_forces),
        )

    def f(self, z, **kwargs):
        q = z[: self.num_joints]
        q_dot = z[self.num_joints : self.num_joints * 2]
        M = self.robot.mass_matrix(q)
        M_inv = jnp.linalg.inv(M)
        bias = self.robot.gravity_vector(q)
        if USE_CENTRIFUGAL_CORIOLIS_IN_CBF:
            bias += self.robot.centrifugal_coriolis_vector(q, q_dot)
        return jnp.concatenate(
            [
                q_dot,  # Joint velocity
                -M_inv @ bias,  # Joint acceleration
            ]
        )

    def g(self, z, **kwargs):
        q = z[: self.num_joints]
        M = self.robot.mass_matrix(q)
        M_inv = jnp.linalg.inv(M)
        return jnp.block(
            [
                [jnp.zeros((self.num_joints, self.m))],
                [M_inv],
            ]
        )

    def _P(self, z):
        """Helper function to compute the QP P matrix. This get reused in both the P and q QP functions"""
        q = z[: self.num_joints]
        transforms = self.robot.joint_to_world_transforms(q)
        M = self.robot._mass_matrix(transforms)
        M_inv = jnp.linalg.inv(M)
        J = self.robot._ee_jacobian(transforms)
        task_inertia_inv = J @ M_inv @ J.T
        task_inertia = jnp.linalg.inv(task_inertia_inv)
        J_bar = M_inv @ J.T @ task_inertia
        NT = jnp.eye(self.num_joints) - J.T @ J_bar.T

        W_T_W_joint = jnp.diag(jnp.asarray(self.W_T_W_joint_diag))
        W_T_W_task = jnp.diag(jnp.asarray(self.W_T_W_task_diag))

        return (
            NT.T @ M_inv.T @ W_T_W_joint @ M_inv @ NT
            + M_inv.T @ J.T @ W_T_W_task @ J @ M_inv
        )

    def P(self, z, u_des):
        return self._P(z)

    def q(self, z, u_des):
        return -u_des.T @ self._P(z)
