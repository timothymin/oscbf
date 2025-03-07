import os
import time
import argparse

import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array
import matplotlib.pyplot as plt

from cbfpy import CBF, CBFConfig
from oscbf.core.manipulator import Manipulator, load_panda
from oscbf.core.manipulation_env import FrankaTorqueControlEnv
from oscbf.core.oscbf_torque import (
    OperationalSpaceCBFController,
    OperationalSpaceCBFConfig,
)


DATA_DIR = "oscbf/experiments/data/"
SAVE_DATA = False


@jax.tree_util.register_static
class CombinedConfig(OperationalSpaceCBFConfig):

    def __init__(
        self,
        robot: Manipulator,
        pos_min: ArrayLike,
        pos_max: ArrayLike,
        collision_positions: ArrayLike,
        collision_radii: ArrayLike,
        whole_body_pos_min: ArrayLike,
        whole_body_pos_max: ArrayLike,
    ):
        self.pos_min = np.asarray(pos_min)
        self.pos_max = np.asarray(pos_max)
        self.q_min = robot.joint_lower_limits
        self.q_max = robot.joint_upper_limits
        self.singularity_tol = 1e-3
        self.collision_positions = np.atleast_2d(collision_positions)
        self.collision_radii = np.ravel(collision_radii)
        assert len(collision_positions) == len(collision_radii)
        self.num_collision_bodies = len(collision_positions)
        self.whole_body_pos_min = np.asarray(whole_body_pos_min)
        self.whole_body_pos_max = np.asarray(whole_body_pos_max)
        super().__init__(robot)

    def h_2(self, z, **kwargs):
        # Extract values
        q = z[: self.num_joints]
        # ee_pos = z[self.num_joints * 2 : self.num_joints * 2 + 3]
        ee_pos = self.robot.ee_position(q)
        q_min = jnp.asarray(self.q_min)
        q_max = jnp.asarray(self.q_max)

        # EE Set Containment
        h_ee_safe_set = jnp.concatenate([self.pos_max - ee_pos, ee_pos - self.pos_min])

        # Joint Limit Avoidance
        h_joint_limits = jnp.concatenate([q_max - q, q - q_min])

        # Singularity Avoidance
        sigmas = jax.lax.linalg.svd(self.robot.ee_jacobian(q), compute_uv=False)
        h_singularity = jnp.array([jnp.prod(sigmas) - self.singularity_tol])

        # Collision Avoidance
        robot_collision_pos_rad = self.robot.link_collision_data(q)
        robot_collision_positions = robot_collision_pos_rad[:, :3]
        robot_collision_radii = robot_collision_pos_rad[:, 3, None]
        robot_num_pts = robot_collision_positions.shape[0]
        center_deltas = (
            robot_collision_positions[:, None, :] - self.collision_positions[None, :, :]
        ).reshape(-1, 3)
        radii_sums = (
            robot_collision_radii[:, None] + self.collision_radii[None, :]
        ).reshape(-1)
        h_collision = jnp.linalg.norm(center_deltas, axis=1) - radii_sums

        # Whole-body Set Containment
        h_whole_body_upper = (
            jnp.tile(self.whole_body_pos_max, (robot_num_pts, 1))
            - robot_collision_positions
            - robot_collision_radii
        ).ravel()
        h_whole_body_lower = (
            robot_collision_positions
            - jnp.tile(self.whole_body_pos_min, (robot_num_pts, 1))
            - robot_collision_radii
        ).ravel()

        return jnp.concatenate(
            [
                h_ee_safe_set,
                h_joint_limits,
                h_singularity,
                h_collision,
                h_whole_body_upper,
                h_whole_body_lower,
            ]
        )

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2


def main():
    robot = load_panda()
    ee_pos_min = np.array([0.15, -0.25, 0.25])
    ee_pos_max = np.array([0.75, 0.25, 0.75])
    wb_pos_min = np.array([-0.5, -0.5, 0.0])
    wb_pos_max = np.array([0.75, 0.5, 1.0])
    collision_pos = np.array([[0.5, 0.5, 0.5]])
    collision_radii = np.array([0.3])
    collision_data = {"positions": collision_pos, "radii": collision_radii}
    config = CombinedConfig(
        robot,
        ee_pos_min,
        ee_pos_max,
        collision_pos,
        collision_radii,
        wb_pos_min,
        wb_pos_max,
    )
    env = FrankaTorqueControlEnv(
        config.pos_min,
        config.pos_max,
        collision_data=collision_data,
        wb_xyz_min=wb_pos_min,
        wb_xyz_max=wb_pos_max,
        load_floor=False,
        bg_color=(1, 1, 1),
        real_time=True,
    )

    env.client.resetDebugVisualizerCamera(
        cameraDistance=2,
        cameraPitch=-27.80,
        cameraYaw=36.80,
        cameraTargetPosition=(0.08, 0.49, -0.04),
    )

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
        # TODO: Realistic torque limits
        tau_min=-100.0 * np.ones(robot.num_joints),
        tau_max=100.0 * np.ones(robot.num_joints),
    )

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
    robot_states = []
    des_ee_states = []
    timestep = env.client.getPhysicsEngineParameters()["fixedTimeStep"]
    try:
        while True:
            joint_state = env.get_joint_state()
            ee_state_des = env.get_desired_ee_state()
            start_time = time.perf_counter()
            tau = compute_control(joint_state, ee_state_des)
            times.append(time.perf_counter() - start_time)
            env.apply_control(tau)
            env.step()
            robot_states.append(joint_state)
            des_ee_states.append(ee_state_des)
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

        if SAVE_DATA:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            folder_path = DATA_DIR + f"{timestamp}/"
            os.makedirs(folder_path, exist_ok=True)
            np.save(folder_path + "robot_states.npy", robot_states)
            np.save(folder_path + "des_ee_states.npy", des_ee_states)
            np.save(folder_path + "compute_times.npy", times)


if __name__ == "__main__":
    main()
