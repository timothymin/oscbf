"""Testing the performance of the OSCBF on end-effector safe-set containment.

Want to show:
- Standard operational space control behavior
- Standard CBF behavior
- OSCBF behavior

Steps:
- Create an end-effector trajectory to follow which is unsafe
    - We can start with the end-effector in the safe set and then reach down to an unsafe level
- Track the EE trajectory with each controller
- Plot the data:
    - Safety criteria (h)
    - Tracking error in non-safety-critical directions
"""

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
from oscbf.core.manipulation_env import FrankaTorqueControlEnv, FrankaVelocityControlEnv
from oscbf.core.oscbf_torque import (
    OperationalSpaceCBFConfig,
    OperationalSpaceCBFController,
)
from oscbf.core.oscbf_velocity import (
    OperationalSpaceVelocityCBFConfig,
    OperationalSpaceVelocityCBFController,
)

DATA_DIR = "oscbf/experiments/data/scaling_collisions/"
SAVE_DATA = False
PAUSE_FOR_PICTURES = False
RECORD_VIDEO = False
PICTURE_IDXS = [1000, 1250, 1600, 1900, 2200]

np.random.seed(0)


@jax.tree_util.register_static
class CollisionsConfig(OperationalSpaceCBFConfig):

    def __init__(
        self,
        robot: Manipulator,
        z_min: float,
        collision_positions: ArrayLike,
        collision_radii: ArrayLike,
    ):
        self.z_min = z_min
        self.collision_positions = np.atleast_2d(collision_positions)
        self.collision_radii = np.ravel(collision_radii)
        super().__init__(robot)

    def h_2(self, z, **kwargs):
        # Extract values
        q = z[: self.num_joints]

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

        # Whole body table avoidance
        h_table = (
            robot_collision_positions[:, 2] - self.z_min - robot_collision_radii.ravel()
        )

        return jnp.concatenate([h_collision, h_table])

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2


@jax.tree_util.register_static
class CollisionsVelocityConfig(OperationalSpaceVelocityCBFConfig):

    def __init__(
        self,
        robot: Manipulator,
        z_min: float,
        collision_positions: ArrayLike,
        collision_radii: ArrayLike,
    ):
        self.z_min = z_min
        self.collision_positions = np.atleast_2d(collision_positions)
        self.collision_radii = np.ravel(collision_radii)
        super().__init__(robot)

    def h_1(self, z, **kwargs):
        # Extract values
        q = z[: self.num_joints]

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

        # Whole body table avoidance
        h_table = robot_collision_positions[:, 2] - self.z_min

        return jnp.concatenate([h_collision, h_table])

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2


def main(control_method="torque", num_bodies=25):
    assert control_method in ["torque", "velocity"]

    robot = load_panda()
    z_min = 0.1

    max_num_bodies = 50

    # Sample a lot of collision point and radii
    # Positions should be between xyz = (0.2, -0.4, 0.1), (0.8, 0.4, 0.3)
    # Radii should be between 0.05 and 0.25
    all_collision_pos = np.random.uniform(
        low=[0.2, -0.4, 0.1], high=[0.8, 0.4, 0.3], size=(max_num_bodies, 3)
    )
    all_collision_radii = np.random.uniform(low=0.01, high=0.1, size=(max_num_bodies,))

    collision_pos = np.atleast_2d(all_collision_pos[:num_bodies])
    collision_radii = all_collision_radii[:num_bodies]

    # collision_pos = np.array(
    #     [
    #         [0.5, 0.5, 0.5],
    #     ]
    # )
    # collision_radii = np.array(
    #     [
    #         0.3,
    #     ]
    # )
    collision_data = {"positions": collision_pos, "radii": collision_radii}
    config = CollisionsConfig(robot, z_min, collision_pos, collision_radii)
    velocity_config = CollisionsVelocityConfig(
        robot, z_min, collision_pos, collision_radii
    )
    timestep = 1 / 240  #  1 / 1000
    bg_color = (1, 1, 1)
    if control_method == "torque":
        env = FrankaTorqueControlEnv(
            real_time=True,
            bg_color=bg_color,
            load_floor=False,
            timestep=timestep,
            collision_data=collision_data,
        )
    else:
        env = FrankaVelocityControlEnv(
            real_time=True,
            bg_color=bg_color,
            load_floor=False,
            timestep=timestep,
            collision_data=collision_data,
        )

    env.client.resetDebugVisualizerCamera(
        cameraDistance=1.40,
        cameraYaw=104.40,
        cameraPitch=-37,
        cameraTargetPosition=(0.20, 0.07, -0.09),
    )

    kp_pos = 50.0
    kp_rot = 20.0
    kd_pos = 20.0
    kd_rot = 10.0
    kp_joint = 10.0
    kd_joint = 5.0
    oscbf_controller = OperationalSpaceCBFController(
        robot,
        config,
        kp_task=np.array([kp_pos, kp_pos, kp_pos, kp_rot, kp_rot, kp_rot]),
        kd_task=np.array([kd_pos, kd_pos, kd_pos, kd_rot, kd_rot, kd_rot]),
        kp_joint=kp_joint,
        kd_joint=kd_joint,
        tau_min=-np.asarray(robot.joint_max_forces),
        tau_max=np.asarray(robot.joint_max_forces),
    )

    oscbf_velocity_controller = OperationalSpaceVelocityCBFController(
        robot,
        velocity_config,
        kp_task=np.array([kp_pos, kp_pos, kp_pos, kp_rot, kp_rot, kp_rot]),
        kp_joint=kp_joint,
        qdot_min=-np.asarray(robot.joint_max_velocities),
        qdot_max=np.asarray(robot.joint_max_velocities),
    )

    @jax.jit
    def compute_oscbf_control(joint_state, ee_state_des):
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
        return oscbf_controller(joint_state, ee_state, joint_state_des, ee_state_des)

    @jax.jit
    def compute_oscbf_velocity_control(q_qdot, z_zdot_ee_des):
        # Nullspace task
        q_des = jnp.array(
            [0.0, -jnp.pi / 6, 0.0, -3 * jnp.pi / 4, 0.0, 5 * jnp.pi / 9, 0.0]
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

        return oscbf_velocity_controller(
            joint_state, ee_state, joint_state_des, ee_state_des, ee_twist_des
        )

    if control_method == "torque":
        compute_control = compute_oscbf_control
    elif control_method == "velocity":
        compute_control = compute_oscbf_velocity_control
    else:
        raise ValueError(f"Invalid control method: {control_method}")

    times = []
    # robot_states = []
    # des_ee_states = []

    if RECORD_VIDEO:
        input("Press Enter to start recording...")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_id = env.client.startStateLogging(
            env.client.STATE_LOGGING_VIDEO_MP4,
            f"artifacts/oscbf_dynamic_{control_method}_{timestamp}.mp4",
        )
    try:
        while True:
            q_qdot = env.get_joint_state()
            z_zdot_ee_des = env.get_desired_ee_state()
            start_time = time.perf_counter()
            tau = compute_control(q_qdot, z_zdot_ee_des)
            times.append(time.perf_counter() - start_time)
            env.apply_control(tau)
            env.step()
            # robot_states.append(q_qdot)
            # des_ee_states.append(z_zdot_ee_des)
    finally:
        if RECORD_VIDEO:
            env.client.stopStateLogging(log_id)
        all_times = np.asarray(times)
        init_solve_time = all_times[0]
        # Ignore initial jit compilation time
        times = np.asarray(all_times[1:])
        avg_solve_time = np.mean(times)
        hzs = 1 / times
        mean_hz = np.mean(hzs)
        std_hz = np.std(hzs)
        print(f"Initial solve time: {init_solve_time} seconds")
        print(f"Average solve time: {avg_solve_time} seconds")
        print(f"Average Hz: {mean_hz}")
        print(f"Std Hz: {std_hz}")
        print(f"Worst case solve time: {np.max(times):.2f} seconds")
        print(f"Worst case Hz: {1 / np.max(times):.2f}")

        if SAVE_DATA:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            mean_hz_int = int(mean_hz)
            std_hz_int = int(std_hz)
            filename = (
                DATA_DIR
                + f"{control_method}_{num_bodies}_bodies_{mean_hz_int}_mean_{std_hz_int}_std.npy"
            )
            np.save(filename, all_times)


# def analyze_data(robot_states, des_ee_states, compute_times):

#     init_solve_time = compute_times[0]
#     # Ignore initial jit compilation time
#     times = np.asarray(compute_times[1:])
#     avg_solve_time = np.mean(times)
#     hzs = 1 / times
#     mean_hz = np.mean(hzs)
#     std_hz = np.std(hzs)
#     print(f"Initial solve time: {init_solve_time} seconds")
#     print(f"Average solve time: {avg_solve_time} seconds")
#     print(f"Average Hz: {mean_hz}")
#     print(f"Std Hz: {std_hz}")
#     print(f"Worst case solve time: {np.max(times):.2f} seconds")
#     print(f"Worst case Hz: {1 / np.max(times):.2f}")

# robot = load_panda()

# @jax.jit
# def h(robot_state):
#     q = robot_state[: robot.num_joints]
#     # Distance from the x limit
#     x_max = 0.65
#     return x_max - robot.ee_position(q)[0]

# @jax.jit
# def xz_position(robot_state):
#     q = robot_state[: robot.num_joints]
#     ee_pos = robot.ee_position(q)
#     return jnp.array([ee_pos[0], ee_pos[2]])

# @jax.jit
# def tracking_error(robot_state, des_ee_state):
#     q = robot_state[: robot.num_joints]
#     robot_tmat = robot.ee_transform(q)
#     robot_pos = robot_tmat[:3, 3]
#     robot_rot = robot_tmat[:3, :3]
#     des_pos = des_ee_state[:3]
#     des_rot = des_ee_state[3:12].reshape((3, 3))
#     pos_error = robot_pos - des_pos
#     rot_error = orientation_error_3D(robot_rot, des_rot)
#     pos_error_norm = jnp.linalg.norm(pos_error)
#     rot_error_norm = jnp.linalg.norm(rot_error)
#     # Hack: copied from above
#     q_des = jnp.array(
#         [
#             0.0,
#             -np.pi / 6,
#             0.0,
#             -3 * np.pi / 4,
#             0.0,
#             5 * np.pi / 9,
#             0.0,
#         ]
#     )
#     posture_error_norm = jnp.linalg.norm(q - q_des)
#     return jnp.array([pos_error_norm, rot_error_norm, posture_error_norm])

# h_values = jax.vmap(h)(robot_states)
# xz_positions = jax.vmap(xz_position)(robot_states)
# # Neglect the first few values
# xz_positions = xz_positions[1000:]
# tracking_errors = jax.vmap(tracking_error)(robot_states, des_ee_states)

# # Plot the XZ position of the end-effector
# plt.figure()
# plt.plot(xz_positions[:, 0], xz_positions[:, 1])
# plt.ylim(0.3, 0.6)
# plt.title("End-effector XZ position")
# plt.xlabel("X")
# plt.ylabel("Z")
# plt.show()

# fig, axs = plt.subplots(1, 4, figsize=(15, 5))
# axs[0].plot(h_values)
# axs[0].set_title("h")
# axs[1].plot(tracking_errors[:, 0])
# axs[1].set_title("Position tracking error")
# axs[2].plot(tracking_errors[:, 1])
# axs[2].set_title("Orientation tracking error")
# axs[3].plot(tracking_errors[:, 2])
# axs[3].set_title("Posture tracking error")
# plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run end-effector safe-set containment experiment."
    )
    parser.add_argument(
        "--control_method",
        type=str,
        choices=["torque", "velocity"],
        default="torque",
        help="Control method to use (default: torque)",
    )
    parser.add_argument(
        "--num_bodies",
        type=int,
        default=25,
        help="Number of collision bodies to simulate (default: 25)",
    )
    args = parser.parse_args()
    main(control_method=args.control_method, num_bodies=args.num_bodies)
