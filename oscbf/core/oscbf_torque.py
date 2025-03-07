"""OSCBF for torque-controlled robots"""

import time
from typing import Optional
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array
import numpy as np
from cbfpy import CBF, CBFConfig
from oscbf.core.manipulator import Manipulator, load_panda
from oscbf.core.manipulation_env import FrankaTorqueControlEnv
from oscbf.utils.controllers import orientation_error_3D, _format_gain, _format_limit


USE_ACCEL_BASED_OBJECTIVE = True
USE_CENTRIFUGAL_CORIOLIS_IN_CBF = False


class OperationalSpaceCBFConfig(CBFConfig):
    """CBF Configuration for safe torque-controlled manipulation

    z = [q, qdot] (length = 2 * num_joints)
    u = [joint torques] (length = num_joints)
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
            n=self.num_joints * 2,
            m=self.num_joints,
            u_min=-np.asarray(robot.joint_max_forces),
            u_max=np.asarray(robot.joint_max_forces),
        )

    def f(self, z, **kwargs):
        # Unpack state variable
        q = z[: self.num_joints]
        q_dot = z[self.num_joints : self.num_joints * 2]

        # Forward kinematics
        transforms = self.robot.joint_to_world_transforms(q)

        # Joint space dynamics
        M = self.robot._mass_matrix(transforms)
        M_inv = jnp.linalg.inv(M)
        g = self.robot._gravity_vector(transforms)
        if USE_CENTRIFUGAL_CORIOLIS_IN_CBF:
            c = self.robot.centrifugal_coriolis_vector(q, q_dot)
        else:
            c = jnp.zeros(self.num_joints)

        return jnp.concatenate(
            [
                q_dot,  # Joint velocity
                -M_inv @ (g + c),  # Joint acceleration
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
        if not USE_ACCEL_BASED_OBJECTIVE:
            return jnp.eye(self.m)

        return self._P(z)

    def q(self, z, u_des):
        if not USE_ACCEL_BASED_OBJECTIVE:
            return -u_des

        return -u_des.T @ self._P(z)


@jax.tree_util.register_static
class EndEffectorSafeSetConfig(OperationalSpaceCBFConfig):

    def __init__(self, robot: Manipulator, pos_min: ArrayLike, pos_max: ArrayLike):
        self.pos_min = np.asarray(pos_min)
        self.pos_max = np.asarray(pos_max)
        super().__init__(robot)

    def h_2(self, z, **kwargs):
        # Standard safe set containment barrier as usual
        q = z[: self.num_joints]
        ee_pos = self.robot.ee_position(q)
        return jnp.concatenate([self.pos_max - ee_pos, ee_pos - self.pos_min])

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2


@jax.tree_util.register_static
class SingularityConfig(OperationalSpaceCBFConfig):

    def __init__(self, robot: Manipulator, tol: float = 1e-2):
        self.tol = tol
        super().__init__(robot)

    def h_2(self, z, **kwargs):
        q = z[: self.num_joints]
        sigmas = jax.lax.linalg.svd(self.robot.ee_jacobian(q), compute_uv=False)
        return jnp.array([jnp.prod(sigmas) - self.tol])


@jax.tree_util.register_static
class WholeBodySafeSetConfig(OperationalSpaceCBFConfig):
    def __init__(self, robot: Manipulator, pos_min: ArrayLike, pos_max: ArrayLike):
        assert robot.has_collision_data
        self.pos_min = np.asarray(pos_min)
        self.pos_max = np.asarray(pos_max)
        super().__init__(robot)

    def h_2(self, z, **kwargs):
        q = z[: self.num_joints]
        collision_pos_rad = self.robot.link_collision_data(q)
        collision_positions = collision_pos_rad[:, :3]
        collision_radii = collision_pos_rad[:, 3, None]
        n_pts = collision_pos_rad.shape[0]
        # TODO: Decide if all of these should be incorporated into the barrier function or only the minimum
        h_upper = (
            jnp.tile(self.pos_max, (n_pts, 1)) - collision_positions - collision_radii
        ).ravel()
        h_lower = (
            collision_positions - jnp.tile(self.pos_min, (n_pts, 1)) - collision_radii
        ).ravel()
        return jnp.concatenate([h_upper, h_lower])


@jax.tree_util.register_static
class JointLimitsConfig(OperationalSpaceCBFConfig):

    def __init__(self, robot: Manipulator):
        self.q_min = robot.joint_lower_limits
        self.q_max = robot.joint_upper_limits
        super().__init__(robot)

    def h_2(self, z, **kwargs):
        q_min = jnp.asarray(self.q_min)
        q_max = jnp.asarray(self.q_max)
        q = z[: self.num_joints]
        return jnp.concatenate([q_max - q, q - q_min])


@jax.tree_util.register_static
class OperationalSpaceCBFController:
    def __init__(
        self,
        robot: Manipulator,
        config: OperationalSpaceCBFConfig,
        kp_task: ArrayLike,
        kd_task: ArrayLike,
        kp_joint: ArrayLike,
        kd_joint: ArrayLike,
        tau_min: Optional[ArrayLike],
        tau_max: Optional[ArrayLike],
    ):
        assert isinstance(robot, Manipulator)
        assert isinstance(config, OperationalSpaceCBFConfig)
        self.robot = robot
        self.config = config
        self.cbf = CBF.from_config(config)
        self.dim_space = 3  # 3D
        self.dim_task = 6  # Position and orientation in 3D
        self.n_joints = self.robot.num_joints
        self.is_redundant = self.n_joints > self.dim_task
        self.kp_task = _format_gain(kp_task, self.dim_task)
        self.kd_task = _format_gain(kd_task, self.dim_task)
        self.kp_joint = _format_gain(kp_joint, self.n_joints)
        self.kd_joint = _format_gain(kd_joint, self.n_joints)
        self.tau_min = _format_limit(tau_min, self.n_joints, "lower")
        self.tau_max = _format_limit(tau_max, self.n_joints, "upper")

    def __call__(
        self,
        joint_state: ArrayLike,
        ee_state: ArrayLike,
        joint_state_des: ArrayLike,
        ee_state_des: ArrayLike,
    ) -> Array:
        """Compute the safe control input to the robot

        Args:
            joint_state (ArrayLike): Joint state vector: [q, qdot], shape (2 * num_joints,)
            ee_state (ArrayLike): End-effector state vector: [pos, rot, vel, omega], shape (3 + 9 + 3 + 3,)
            joint_state_des (ArrayLike): Desired joint state vector: [q_des, qdot_des], shape (2 * num_joints,)
            ee_state_des (ArrayLike): Desired end-effector state vector: [pos_des, rot_des, vel_des, omega_des],
                shape (3 + 9 + 3 + 3,)

        Returns:
            Array: Safe joint torque command, shape (num_joints,)
        """
        u = self._osc(joint_state, ee_state, joint_state_des, ee_state_des)
        return self.cbf.safety_filter(joint_state, u)

    def _osc(
        self,
        joint_state: ArrayLike,
        ee_state: ArrayLike,
        joint_state_des: ArrayLike,
        ee_state_des: ArrayLike,
    ) -> Array:
        """Compute the nominal joint torques from operational space control

        Args:
            joint_state (ArrayLike): Joint state vector: [q, qdot], shape (2 * num_joints,)
            ee_state (ArrayLike): End-effector state vector: [pos, rot, vel, omega], shape (3 + 9 + 3 + 3,)
            joint_state_des (ArrayLike): Desired joint state vector: [q_des, qdot_des], shape (2 * num_joints,)
            ee_state_des (ArrayLike): Desired end-effector state vector: [pos_des, rot_des, vel_des, omega_des],
                shape (3 + 9 + 3 + 3,)

        Returns:
            Array: Joint torque command, shape (num_joints,)
        """
        num_joints = self.robot.num_joints

        q = joint_state[:num_joints]
        qdot = joint_state[num_joints : num_joints * 2]
        pos = ee_state[:3]
        rot = ee_state[3:12].reshape(3, 3)
        vel = ee_state[12:15]
        omega = ee_state[15:18]

        q_des = joint_state_des[:num_joints]
        qdot_des = joint_state_des[num_joints : num_joints * 2]
        pos_des = ee_state_des[:3]
        rot_des = ee_state_des[3:12].reshape(3, 3)
        vel_des = ee_state_des[12:15]
        omega_des = ee_state_des[15:18]

        # TEMP: Assume that the trajectory does not specify feed-forward acceleration terms
        qddot_des = jnp.zeros(self.n_joints)
        accel_des = jnp.zeros(self.dim_space)
        alpha_des = jnp.zeros(self.dim_space)

        # Compute dynamics terms
        transforms = self.robot.joint_to_world_transforms(q)
        J = self.robot._ee_jacobian(transforms)
        # Joint space dynamics
        M = self.robot._mass_matrix(transforms)
        M_inv = jnp.linalg.inv(M)
        g = self.robot._gravity_vector(transforms)
        if USE_CENTRIFUGAL_CORIOLIS_IN_CBF:
            c = self.robot.centrifugal_coriolis_vector(q, qdot)
        else:
            c = jnp.zeros(num_joints)
        # Operational space dynamics
        task_inertia_inv = J @ M_inv @ J.T
        task_inertia = jnp.linalg.inv(task_inertia_inv)  # Op-space inertia matrix
        J_bar = M_inv @ J.T @ task_inertia  # Dynamically consistent Jacobian inverse

        # Errors
        pos_error = pos - pos_des
        vel_error = vel - vel_des
        rot_error = orientation_error_3D(rot, rot_des)
        omega_error = omega - omega_des
        task_p_error = jnp.concatenate([pos_error, rot_error])
        task_d_error = jnp.concatenate([vel_error, omega_error])

        # Compute task torques
        task_accel = (
            jnp.concatenate([accel_des, alpha_des])
            - self.kp_task * task_p_error
            - self.kd_task * task_d_error
        )
        task_wrench = task_inertia @ task_accel

        # Handle secondary joint task in the nullspace
        if self.is_redundant:
            NT = jnp.eye(num_joints) - J.T @ J_bar.T  # Nullspace Projection
            q_error = q - q_des
            qdot_error = qdot - qdot_des
            joint_accel = (
                qddot_des - self.kp_joint * q_error - self.kd_joint * qdot_error
            )
            secondary_joint_torques = M @ joint_accel
            tau_null = NT @ secondary_joint_torques
        else:
            tau_null = jnp.zeros(num_joints)

        return J.T @ task_wrench + tau_null + g + c


def main():
    robot = load_panda()
    config = EndEffectorSafeSetConfig(
        robot, np.array([-0.25, -0.25, 0.0]), np.array([0.5, 0.25, 0.75])
    )
    env = FrankaTorqueControlEnv(config.pos_min, config.pos_max, real_time=True)

    kp_pos = 50.0
    kp_rot = 20.0
    kd_pos = 20.0
    kd_rot = 10.0
    kp_joint = 10.0
    kd_joint = 5.0
    controller = OperationalSpaceCBFController(
        robot,
        config,
        kp_task=np.array([kp_pos, kp_pos, kp_pos, kp_rot, kp_rot, kp_rot]),
        kd_task=np.array([kd_pos, kd_pos, kd_pos, kd_rot, kd_rot, kd_rot]),
        kp_joint=kp_joint,
        kd_joint=kd_joint,
        tau_min=-np.asarray(robot.joint_max_forces),
        tau_max=np.asarray(robot.joint_max_forces),
    )

    # TODO: Reorganize this so that we don't have to make the additional calls to robot.ee_transform and
    # robot.ee_jacobian -- move these inside the controller somehow
    @jax.jit
    def compute_control(joint_state, ee_state_des):
        # Nullspace task
        q_des = jnp.array(
            [0.0, -jnp.pi / 6, 0.0, -3 * jnp.pi / 4, 0.0, 5 * jnp.pi / 9, 0.0]
        )
        qdot_des = jnp.zeros(robot.num_joints)

        q = joint_state[: robot.num_joints]
        qdot = joint_state[robot.num_joints :]
        t_ee = robot.ee_transform(q)
        J = robot.ee_jacobian(q)
        pos_ee = t_ee[:3, 3]
        rot_ee_flat = t_ee[:3, :3].ravel()
        twist_ee = J @ qdot
        joint_state_des = jnp.concatenate([q_des, qdot_des])
        ee_state = jnp.concatenate([pos_ee, rot_ee_flat, twist_ee])
        return controller(joint_state, ee_state, joint_state_des, ee_state_des)

    times = []
    try:
        while True:
            joint_state = env.get_joint_state()
            ee_state_des = env.get_desired_ee_state()
            start_time = time.perf_counter()
            tau = compute_control(joint_state, ee_state_des)
            times.append(time.perf_counter() - start_time)
            env.apply_control(tau)
            env.step()
    finally:
        times = np.asarray(times)
        init_solve_time = times[0]
        avg_solve_time = np.mean(times[1:])
        print(f"Initial solve time: {init_solve_time} seconds")
        print(f"Average solve time: {avg_solve_time} seconds")
        print(f"Average Hz: {1 / avg_solve_time:.2f}")


if __name__ == "__main__":
    main()
