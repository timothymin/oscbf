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


# TODO
# Clean up the inputs -- it's weird to pass in the ee_twist_des separately

USE_CONSISTENT_OBJECTIVE = True


class OperationalSpaceVelocityCBFConfig(CBFConfig):
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
        if not USE_CONSISTENT_OBJECTIVE:
            return jnp.eye(self.m)

        return self._P(z)

    def q(self, z, u_des, **kwargs):
        if not USE_CONSISTENT_OBJECTIVE:
            return -u_des

        return -u_des.T @ self._P(z)


@jax.tree_util.register_static
class EndEffectorSafeSetConfig(OperationalSpaceVelocityCBFConfig):

    def __init__(self, robot: Manipulator, pos_min: ArrayLike, pos_max: ArrayLike):
        self.pos_min = np.asarray(pos_min)
        self.pos_max = np.asarray(pos_max)
        super().__init__(robot)

    def h_1(self, z, **kwargs):
        # Standard safe set containment barrier as usual
        q = z
        ee_pos = self.robot.ee_position(q)
        return jnp.concatenate([self.pos_max - ee_pos, ee_pos - self.pos_min])

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2


@jax.tree_util.register_static
class SingularityConfig(OperationalSpaceVelocityCBFConfig):

    def __init__(self, robot: Manipulator, tol: float = 1e-2):
        self.tol = tol
        super().__init__(robot)

    def h_1(self, z, **kwargs):
        q = z
        sigmas = jax.lax.linalg.svd(self.robot.ee_jacobian(q), compute_uv=False)
        return jnp.array([jnp.prod(sigmas) - self.tol])


@jax.tree_util.register_static
class WholeBodySafeSetConfig(OperationalSpaceVelocityCBFConfig):
    def __init__(self, robot: Manipulator, pos_min: ArrayLike, pos_max: ArrayLike):
        assert robot.has_collision_data
        self.pos_min = np.asarray(pos_min)
        self.pos_max = np.asarray(pos_max)
        super().__init__(robot)

    def h_1(self, z, **kwargs):
        q = z
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
class JointLimitsConfig(OperationalSpaceVelocityCBFConfig):

    def __init__(self, robot: Manipulator):
        self.q_min = robot.joint_lower_limits
        self.q_max = robot.joint_upper_limits
        super().__init__(robot)

    def h_1(self, z, **kwargs):
        q_min = jnp.asarray(self.q_min)
        q_max = jnp.asarray(self.q_max)
        q = z
        return jnp.concatenate([q_max - q, q - q_min])


@jax.tree_util.register_static
class OperationalSpaceVelocityCBFController:
    def __init__(
        self,
        robot: Manipulator,
        config: OperationalSpaceVelocityCBFConfig,
        kp_task: ArrayLike,
        kp_joint: ArrayLike,
        qdot_min: Optional[ArrayLike],
        qdot_max: Optional[ArrayLike],
    ):
        assert isinstance(robot, Manipulator)
        assert isinstance(config, OperationalSpaceVelocityCBFConfig)
        self.robot = robot
        self.config = config
        self.cbf = CBF.from_config(config)
        self.dim_space = 3  # 3D
        self.dim_task = 6  # Position and orientation in 3D
        self.n_joints = self.robot.num_joints
        self.is_redundant = self.n_joints > self.dim_task
        self.kp_task = _format_gain(kp_task, self.dim_task)
        self.kp_joint = _format_gain(kp_joint, self.n_joints)
        self.qdot_min = _format_limit(qdot_min, self.n_joints, "lower")
        self.qdot_max = _format_limit(qdot_max, self.n_joints, "upper")

    def __call__(
        self,
        joint_state: ArrayLike,
        ee_state: ArrayLike,
        joint_state_des: ArrayLike,
        ee_state_des: ArrayLike,
        ee_twist_des: ArrayLike,
    ) -> Array:
        """Compute the safe control input to the robot

        Args:
            z (ArrayLike): State vector: [q, qdot, ee_pose, ee_twist]
            z_des (ArrayLike): Desired state vector: [q_des, qdot_des, ee_pose_des, ee_twist_des]

        Returns:
            Array: Joint torque command, shape (num_joints,)
        """
        u = self._osc(
            joint_state, ee_state, joint_state_des, ee_state_des, ee_twist_des
        )
        return self.cbf.safety_filter(joint_state, u)

    def _osc(
        self,
        joint_state: ArrayLike,
        ee_state: ArrayLike,
        joint_state_des: ArrayLike,
        ee_state_des: ArrayLike,
        ee_twist_des: ArrayLike,
    ) -> Array:
        """Compute the desired forces/torques in the null space and operational space

        Args:
            z (Array): State vector: [q, ee_pose]
            z_des (Array): Desired state vector: [q_des, ee_pose_des]
            transforms (Array): Joint-to-world transformation matrices, shape (num_joints, 4, 4)
                This is passed in to avoid recomputing the forward kinematics

        Returns:
            Array: Nominal control: [tau_null, task_wrench], shape (num_joints + 6,)
        """
        num_joints = self.robot.num_joints

        q = joint_state[:num_joints]
        pos = ee_state[:3]
        rot = ee_state[3:].reshape(3, 3)

        q_des = joint_state_des
        pos_des = ee_state_des[:3]
        rot_des = ee_state_des[3:].reshape(3, 3)

        # Errors
        pos_error = pos - pos_des
        rot_error = orientation_error_3D(rot, rot_des)
        task_p_error = jnp.concatenate([pos_error, rot_error])

        transforms = self.robot.joint_to_world_transforms(q)
        J = self.robot._ee_jacobian(transforms)
        M = self.robot._mass_matrix(transforms)
        M_inv = jnp.linalg.inv(M)
        task_inertia_inv = J @ M_inv @ J.T
        task_inertia = jnp.linalg.inv(task_inertia_inv)
        J_bar = M_inv @ J.T @ task_inertia
        J_hash = J_bar

        # Compute task velocities
        task_vel = ee_twist_des - self.kp_task * task_p_error
        # Map to joint velocities
        v = J_hash @ task_vel

        # Handle secondary joint task in the nullspace
        if self.is_redundant:
            # Nullspace projection
            N = jnp.eye(self.n_joints) - J_hash @ J
            # Add nullspace joint task
            q_error = q - q_des
            secondary_joint_vel = -self.kp_joint * q_error
            v_null = N @ secondary_joint_vel
        else:
            v_null = jnp.zeros(self.n_joints)

        return v + v_null


def main():
    robot = load_panda()
    config = EndEffectorSafeSetConfig(
        robot, np.array([-0.25, -0.25, 0.0]), np.array([0.5, 0.25, 0.75])
    )
    env = FrankaVelocityControlEnv(config.pos_min, config.pos_max, real_time=True)

    kp_pos = 50.0
    kp_rot = 20.0
    kp_joint = 10.0
    controller = OperationalSpaceVelocityCBFController(
        robot,
        config,
        kp_task=np.array([kp_pos, kp_pos, kp_pos, kp_rot, kp_rot, kp_rot]),
        kp_joint=kp_joint,
        qdot_min=-np.asarray(robot.joint_max_velocities),
        qdot_max=np.asarray(robot.joint_max_velocities),
    )

    @jax.jit
    def compute_control(q_qdot, z_zdot_ee_des):
        # Nullspace task
        q_des = np.array(
            [
                0.0,
                -np.pi / 6,
                0.0,
                -3 * np.pi / 4,
                0.0,
                5 * np.pi / 9,
                0.0,
            ]
        )
        q = q_qdot[: robot.num_joints]
        t_ee = robot.ee_transform(q)
        pos_ee = t_ee[:3, 3]
        rot_ee_flat = t_ee[:3, :3].ravel()
        ee_pos_des = z_zdot_ee_des[:3]
        ee_rot_flat_des = z_zdot_ee_des[3:12]
        ee_twist_des = z_zdot_ee_des[12:]

        joint_state = q
        ee_state = jnp.concatenate([pos_ee, rot_ee_flat])
        joint_state_des = q_des
        ee_state_des = jnp.concatenate([ee_pos_des, ee_rot_flat_des])

        return controller(
            joint_state, ee_state, joint_state_des, ee_state_des, ee_twist_des
        )

    times = []
    try:
        while True:
            q_qdot = env.get_joint_state()
            z_zdot_ee_des = env.get_desired_ee_state()
            start_time = time.perf_counter()
            tau = compute_control(q_qdot, z_zdot_ee_des)
            times.append(time.perf_counter() - start_time)
            env.apply_control(tau)
            env.step()
    finally:
        times = np.asarray(times)
        init_solve_time = times[0]
        avg_solve_time = np.mean(times[1:])
        hzs = 1 / times[1:]
        mean_hz = np.mean(hzs)
        std_hz = np.std(hzs)
        print(f"Initial solve time: {init_solve_time} seconds")
        print(f"Average solve time: {avg_solve_time} seconds")
        print(f"Average Hz: {mean_hz}")
        print(f"Std Hz: {std_hz}")
        # Save the times
        np.save("artifacts/solve_times.npy", times)


if __name__ == "__main__":
    main()
