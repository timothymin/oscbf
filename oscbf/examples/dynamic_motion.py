"""Testing the performance of OSCBF during dynamic motions and input constraints

In general, we will command a rapid motion of the end-effector into the unsafe set,
and observe the controller's behavior under velocity control and torque control.

The reduced-order (velocity-control) OSCBF has no lower-level understanding of torque
limits, so the full-order (torque-control) OSCBF should perform better in this case.
"""

import argparse
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


from cbfpy import CBF
from oscbf.core.manipulator import Manipulator, load_panda
from oscbf.core.manipulation_env import FrankaTorqueControlEnv, FrankaVelocityControlEnv
from oscbf.core.oscbf_configs import OSCBFTorqueConfig, OSCBFVelocityConfig
from oscbf.utils.trajectory import SinusoidalTaskTrajectory
from oscbf.core.controllers import (
    PoseTaskTorqueController,
    PoseTaskVelocityController,
)

DATA_DIR = "oscbf/experiments/data/"
SAVE_DATA = False
PAUSE_FOR_PICTURES = False
RECORD_VIDEO = False
PICTURE_IDXS = [1000, 1250, 1600, 1900, 2200]


@jax.tree_util.register_pytree_node_class
class EESafeSetTorqueConfig(OSCBFTorqueConfig):

    def __init__(
        self,
        robot: Manipulator,
        pos_min: ArrayLike,
        pos_max: ArrayLike,
        compensate_centrifugal_coriolis: bool,
    ):
        self.pos_min = np.asarray(pos_min)
        self.pos_max = np.asarray(pos_max)
        super().__init__(
            robot, compensate_centrifugal_coriolis=compensate_centrifugal_coriolis
        )

    def h_2(self, z, **kwargs):
        q = z[: self.num_joints]
        ee_pos = self.robot.ee_position(q)
        return jnp.concatenate([self.pos_max - ee_pos, ee_pos - self.pos_min])

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2

    def tree_flatten(self):
        """Flatten the EESafeSetTorqueConfig object for JAX pytree registration."""
        children = ()  # No dynamic JAX arrays in this object
        aux_data = (
            self.pos_min,
            self.pos_max,
            self.robot,
            self.num_joints,
            self.n,
            self.m,
            self.num_cbf,
            self.u_min,
            self.u_max,
            self.control_constrained,
            self.relax_cbf,
            self.cbf_relaxation_penalty,
            self.solver_tol,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct the EESafeSetTorqueConfig object from flattened representation."""
        (
            pos_min,
            pos_max,
            robot,
            num_joints,
            n,
            m,
            num_cbf,
            u_min,
            u_max,
            control_constrained,
            relax_cbf,
            cbf_relaxation_penalty,
            solver_tol,
        ) = aux_data
        
        # Create a new instance without going through __init__ validation
        instance = cls.__new__(cls)
        instance.pos_min = pos_min
        instance.pos_max = pos_max
        instance.robot = robot
        instance.num_joints = num_joints
        instance.n = n
        instance.m = m
        instance.num_cbf = num_cbf
        instance.u_min = u_min
        instance.u_max = u_max
        instance.control_constrained = control_constrained
        instance.relax_cbf = relax_cbf
        instance.cbf_relaxation_penalty = cbf_relaxation_penalty
        instance.solver_tol = solver_tol
        
        return instance


@jax.tree_util.register_pytree_node_class
class EESafeSetVelocityConfig(OSCBFVelocityConfig):

    def __init__(self, robot: Manipulator, pos_min: ArrayLike, pos_max: ArrayLike):
        self.pos_min = np.asarray(pos_min)
        self.pos_max = np.asarray(pos_max)
        super().__init__(robot)

    def h_1(self, z, **kwargs):
        q = z[: self.num_joints]
        ee_pos = self.robot.ee_position(q)
        return jnp.concatenate([self.pos_max - ee_pos, ee_pos - self.pos_min])

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2

    def tree_flatten(self):
        """Flatten the EESafeSetVelocityConfig object for JAX pytree registration."""
        children = ()  # No dynamic JAX arrays in this object
        aux_data = (
            self.pos_min,
            self.pos_max,
            self.robot,
            self.num_joints,
            self.n,
            self.m,
            self.num_cbf,
            self.u_min,
            self.u_max,
            self.control_constrained,
            self.relax_cbf,
            self.cbf_relaxation_penalty,
            self.solver_tol,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct the EESafeSetVelocityConfig object from flattened representation."""
        (
            pos_min,
            pos_max,
            robot,
            num_joints,
            n,
            m,
            num_cbf,
            u_min,
            u_max,
            control_constrained,
            relax_cbf,
            cbf_relaxation_penalty,
            solver_tol,
        ) = aux_data
        
        # Create a new instance without going through __init__ validation
        instance = cls.__new__(cls)
        instance.pos_min = pos_min
        instance.pos_max = pos_max
        instance.robot = robot
        instance.num_joints = num_joints
        instance.n = n
        instance.m = m
        instance.num_cbf = num_cbf
        instance.u_min = u_min
        instance.u_max = u_max
        instance.control_constrained = control_constrained
        instance.relax_cbf = relax_cbf
        instance.cbf_relaxation_penalty = cbf_relaxation_penalty
        instance.solver_tol = solver_tol
        
        return instance


# @partial(jax.jit, static_argnums=(0, 1, 2, 3))
def compute_torque_control(
    robot: Manipulator,
    osc_controller: PoseTaskTorqueController,
    cbf: CBF,
    compensate_centrifugal_coriolis: bool,
    z: ArrayLike,
    z_ee_des: ArrayLike,
):
    q = z[: robot.num_joints]
    qdot = z[robot.num_joints :]
    M, M_inv, g, c, J, ee_tmat = robot.torque_control_matrices(q, qdot)

    if not compensate_centrifugal_coriolis:
        c = jnp.zeros(robot.num_joints)

    # Set nullspace desired joint position
    nullspace_posture_goal = jnp.array(
        [
            0.0,
            -jnp.pi / 6,
            0.0,
            -3 * jnp.pi / 4,
            0.0,
            5 * jnp.pi / 9,
            0.0,
        ]
    )

    # Compute nominal control
    u_nom = osc_controller(
        q,
        qdot,
        pos=ee_tmat[:3, 3],
        rot=ee_tmat[:3, :3],
        des_pos=z_ee_des[:3],
        des_rot=jnp.reshape(z_ee_des[3:12], (3, 3)),
        des_vel=z_ee_des[12:15],
        des_omega=z_ee_des[15:18],
        des_accel=jnp.zeros(3),
        des_alpha=jnp.zeros(3),
        des_q=nullspace_posture_goal,
        des_qdot=jnp.zeros(robot.num_joints),
        J=J,
        M=M,
        M_inv=M_inv,
        g=g,
        c=c,
    )
    # Apply the CBF safety filter
    return cbf.safety_filter(z, u_nom)


# @partial(jax.jit, static_argnums=(0, 1, 2))
def compute_velocity_control(
    robot: Manipulator,
    osc_controller: PoseTaskVelocityController,
    cbf: CBF,
    z: ArrayLike,
    z_ee_des: ArrayLike,
):
    q = z[: robot.num_joints]
    M_inv, J, ee_tmat = robot.dynamically_consistent_velocity_control_matrices(q)
    pos = ee_tmat[:3, 3]
    rot = ee_tmat[:3, :3]
    des_pos = z_ee_des[:3]
    des_rot = jnp.reshape(z_ee_des[3:12], (3, 3))
    des_vel = z_ee_des[12:15]
    des_omega = z_ee_des[15:18]
    # Set nullspace desired joint position
    des_q = jnp.array(
        [
            0.0,
            -jnp.pi / 6,
            0.0,
            -3 * jnp.pi / 4,
            0.0,
            5 * jnp.pi / 9,
            0.0,
        ]
    )
    u_nom = osc_controller(
        q, pos, rot, des_pos, des_rot, des_vel, des_omega, des_q, J, M_inv
    )
    return cbf.safety_filter(q, u_nom)


def main(control_method="torque"):
    assert control_method in ["torque", "velocity"]

    robot = load_panda()
    pos_min = (0.25, -0.25, 0.25)
    pos_max = (0.65, 0.25, 0.65)

    # NOTE: This term has a noticeable impact on the performance for this demo.
    # It's often neglected due to computational demands and model error
    compensate_centrifugal_coriolis = False

    torque_config = EESafeSetTorqueConfig(
        robot,
        pos_min,
        pos_max,
        compensate_centrifugal_coriolis=compensate_centrifugal_coriolis,
    )
    torque_cbf = CBF.from_config(torque_config)
    velocity_config = EESafeSetVelocityConfig(robot, pos_min, pos_max)
    velocity_cbf = CBF.from_config(velocity_config)
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
            torque_config.pos_min,
            torque_config.pos_max,
            traj=traj,
            real_time=True,
            bg_color=bg_color,
            load_floor=False,
            timestep=timestep,
        )
    else:
        env = FrankaVelocityControlEnv(
            velocity_config.pos_min,
            velocity_config.pos_max,
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
    osc_torque_controller = PoseTaskTorqueController(
        n_joints=robot.num_joints,
        kp_task=np.concatenate([kp_pos * np.ones(3), kp_rot * np.ones(3)]),
        kd_task=np.concatenate([kd_pos * np.ones(3), kd_rot * np.ones(3)]),
        kp_joint=kp_joint,
        kd_joint=kd_joint,
        # Note: torque limits will be enforced via the QP. We'll set them to None here
        # because we don't want to clip the values before the QP
        tau_min=None,
        tau_max=None,
    )

    osc_velocity_controller = PoseTaskVelocityController(
        n_joints=robot.num_joints,
        kp_task=np.array([kp_pos, kp_pos, kp_pos, kp_rot, kp_rot, kp_rot]),
        kp_joint=kp_joint,
        # Note: velocity limits will be enforced via the QP
        qdot_min=None,
        qdot_max=None,
    )

    @jax.jit
    def compute_torque_control_jit(z, z_ee_des):
        return compute_torque_control(
            robot,
            osc_torque_controller,
            torque_cbf,
            compensate_centrifugal_coriolis,
            z,
            z_ee_des,
        )

    @jax.jit
    def compute_velocity_control_jit(z, z_ee_des):
        return compute_velocity_control(
            robot, osc_velocity_controller, velocity_cbf, z, z_ee_des
        )

    if control_method == "torque":
        compute_control = compute_torque_control_jit
    elif control_method == "velocity":
        compute_control = compute_velocity_control_jit
    else:
        raise ValueError(f"Invalid control method: {control_method}")

    while True:
        q_qdot = env.get_joint_state()
        z_zdot_ee_des = env.get_desired_ee_state()
        tau = compute_control(q_qdot, z_zdot_ee_des)
        env.apply_control(tau)
        env.step()


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
