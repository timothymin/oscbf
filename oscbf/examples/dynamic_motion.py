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
from oscbf.utils.trajectory import SinusoidalTaskTrajectory
from oscbf.utils.controllers import PoseTaskTorqueController, orientation_error_3D

DATA_DIR = "oscbf/experiments/data/"
SAVE_DATA = False
PAUSE_FOR_PICTURES = False
RECORD_VIDEO = False
PICTURE_IDXS = [1000, 1250, 1600, 1900, 2200]


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
class EndEffectorSafeSetVelocityConfig(OperationalSpaceVelocityCBFConfig):

    def __init__(self, robot: Manipulator, pos_min: ArrayLike, pos_max: ArrayLike):
        self.pos_min = np.asarray(pos_min)
        self.pos_max = np.asarray(pos_max)
        super().__init__(robot)

    def h_1(self, z, **kwargs):
        # Standard safe set containment barrier as usual
        q = z[: self.num_joints]
        ee_pos = self.robot.ee_position(q)
        return jnp.concatenate([self.pos_max - ee_pos, ee_pos - self.pos_min])

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2


def main(control_method="torque"):
    assert control_method in ["torque", "velocity"]

    robot = load_panda()
    pos_min = (0.25, -0.25, 0.25)
    pos_max = (0.65, 0.25, 0.65)

    config = EndEffectorSafeSetConfig(robot, pos_min, pos_max)
    velocity_config = EndEffectorSafeSetVelocityConfig(robot, pos_min, pos_max)
    traj = SinusoidalTaskTrajectory(
        init_pos=(0.55, 0, 0.45),
        init_rot=np.array(
            [
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ]
        ),
        amplitude=(0.25, 0, 0),
        angular_freq=(5, 0, 0),
        phase=(0, 0, 0),
    )
    timestep = 1 / 1000
    bg_color = (1, 1, 1)
    if control_method == "torque":
        env = FrankaTorqueControlEnv(
            config.pos_min,
            config.pos_max,
            traj=traj,
            real_time=True,
            bg_color=bg_color,
            load_floor=False,
            timestep=timestep,
        )
    else:
        env = FrankaVelocityControlEnv(
            config.pos_min,
            config.pos_max,
            traj=traj,
            real_time=True,
            bg_color=bg_color,
            load_floor=False,
            timestep=timestep,
        )

    env.client.resetDebugVisualizerCamera(
        cameraDistance=1.00,
        cameraYaw=12,
        cameraPitch=-2.6,
        cameraTargetPosition=(0.44, 0.16, 0.28),
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

    @jax.jit
    def get_ee_pos(q_qdot):
        q = q_qdot[: robot.num_joints]
        return robot.ee_position(q)

    if control_method == "torque":
        compute_control = compute_oscbf_control
    elif control_method == "velocity":
        compute_control = compute_oscbf_velocity_control
    else:
        raise ValueError(f"Invalid control method: {control_method}")

    times = []
    robot_states = []
    des_ee_states = []
    duration = 3 * (traj.angular_freq[0] / (2 * np.pi)) ** (-1)
    n_timesteps = int(duration / env.dt)

    # env.client.addUserDebugPoints(
    #     [get_ee_pos(env.get_joint_state())], [[0, 0, 1]], 5, 10
    # )

    if RECORD_VIDEO:
        input("Press Enter to start recording...")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_id = env.client.startStateLogging(
            env.client.STATE_LOGGING_VIDEO_MP4,
            f"artifacts/oscbf_dynamic_{control_method}_{timestamp}.mp4",
        )
    try:
        for i in range(n_timesteps):
            if PAUSE_FOR_PICTURES and i in PICTURE_IDXS:
                input("Press Enter to continue...")
            q_qdot = env.get_joint_state()
            z_zdot_ee_des = env.get_desired_ee_state()
            start_time = time.perf_counter()
            tau = compute_control(q_qdot, z_zdot_ee_des)
            times.append(time.perf_counter() - start_time)
            env.apply_control(tau)
            env.step()
            # env.client.addUserDebugPoints([get_ee_pos(q_qdot)], [[0, 0, 1]], 5, 10)
            robot_states.append(q_qdot)
            des_ee_states.append(z_zdot_ee_des)
            # print(i)
            # time.sleep(1 / 120)
    finally:
        if RECORD_VIDEO:
            env.client.stopStateLogging(log_id)
        times = np.asarray(times)
        robot_states = np.asarray(robot_states)
        des_ee_states = np.asarray(des_ee_states)

        if SAVE_DATA:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            folder_path = DATA_DIR + f"{control_method}_{timestamp}/"
            os.makedirs(folder_path, exist_ok=True)
            np.save(folder_path + "robot_states.npy", robot_states)
            np.save(folder_path + "des_ee_states.npy", des_ee_states)
            np.save(folder_path + "compute_times.npy", times)

        analyze_data(robot_states, des_ee_states, times)


def analyze_data(robot_states, des_ee_states, compute_times):

    init_solve_time = compute_times[0]
    # Ignore initial jit compilation time
    times = compute_times[1:]
    avg_solve_time = np.mean(times)
    print(f"Initial solve time: {init_solve_time} seconds")
    print(f"Average solve time: {avg_solve_time} seconds")
    print(f"Average Hz: {1 / avg_solve_time:.2f}")
    print(f"Worst case solve time: {np.max(times):.2f} seconds")
    print(f"Worst case Hz: {1 / np.max(times):.2f}")

    robot = load_panda()

    @jax.jit
    def h(robot_state):
        q = robot_state[: robot.num_joints]
        # Distance from the x limit
        x_max = 0.65
        return x_max - robot.ee_position(q)[0]

    @jax.jit
    def xz_position(robot_state):
        q = robot_state[: robot.num_joints]
        ee_pos = robot.ee_position(q)
        return jnp.array([ee_pos[0], ee_pos[2]])

    @jax.jit
    def tracking_error(robot_state, des_ee_state):
        q = robot_state[: robot.num_joints]
        robot_tmat = robot.ee_transform(q)
        robot_pos = robot_tmat[:3, 3]
        robot_rot = robot_tmat[:3, :3]
        des_pos = des_ee_state[:3]
        des_rot = des_ee_state[3:12].reshape((3, 3))
        pos_error = robot_pos - des_pos
        rot_error = orientation_error_3D(robot_rot, des_rot)
        pos_error_norm = jnp.linalg.norm(pos_error)
        rot_error_norm = jnp.linalg.norm(rot_error)
        # Hack: copied from above
        q_des = jnp.array(
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
        posture_error_norm = jnp.linalg.norm(q - q_des)
        return jnp.array([pos_error_norm, rot_error_norm, posture_error_norm])

    h_values = jax.vmap(h)(robot_states)
    xz_positions = jax.vmap(xz_position)(robot_states)
    # Neglect the first few values
    xz_positions = xz_positions[1000:]
    tracking_errors = jax.vmap(tracking_error)(robot_states, des_ee_states)

    # Plot the XZ position of the end-effector
    plt.figure()
    plt.plot(xz_positions[:, 0], xz_positions[:, 1])
    plt.ylim(0.3, 0.6)
    plt.title("End-effector XZ position")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.show()

    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    axs[0].plot(h_values)
    axs[0].set_title("h")
    axs[1].plot(tracking_errors[:, 0])
    axs[1].set_title("Position tracking error")
    axs[2].plot(tracking_errors[:, 1])
    axs[2].set_title("Orientation tracking error")
    axs[3].plot(tracking_errors[:, 2])
    axs[3].set_title("Posture tracking error")
    plt.show()


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
    args = parser.parse_args()
    main(control_method=args.control_method)
