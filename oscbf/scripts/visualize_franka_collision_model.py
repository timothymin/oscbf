"""Script to visualize the hand-designed collision model of the Franka Panda in Pybullet"""

import pybullet
import numpy as np

import oscbf.core.franka_collision_model as colmodel
from oscbf.core.manipulator import create_transform_numpy, load_panda
from oscbf.utils.visualization import visualize_3D_sphere

np.random.seed(0)

URDF = "oscbf/assets/franka_panda/panda.urdf"
FRANKA_INIT_QPOS = np.array(
    [0.0, -np.pi / 6, 0.0, -3 * np.pi / 4, 0.0, 5 * np.pi / 9, 0.0]
)
RANDOMIZE = False


def main():
    pybullet.connect(pybullet.GUI)
    robot = pybullet.loadURDF(
        URDF,
        useFixedBase=True,
        flags=pybullet.URDF_USE_INERTIA_FROM_FILE | pybullet.URDF_MERGE_FIXED_LINKS,
    )
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

    manipulator = load_panda()
    for link_idx in range(manipulator.num_joints):
        pybullet.changeVisualShape(robot, link_idx, rgbaColor=(0, 0, 0, 0.5))

    # input("Press Enter to randomize the joints")
    if RANDOMIZE:
        q = np.random.rand(manipulator.num_joints)
    else:
        q = FRANKA_INIT_QPOS
    for i in range(manipulator.num_joints):
        pybullet.resetJointState(robot, i, q[i])
    pybullet.stepSimulation()
    joint_transforms = manipulator.joint_to_world_transforms(q)

    sphere_ids = []
    # Determine the world-frame positions of the collision geometry
    for i in range(manipulator.num_joints):
        link_name = f"link_{i+1}"
        parent_to_world_tf = joint_transforms[i]
        num_collision_spheres = len(colmodel.positions[link_name])
        for j in range(num_collision_spheres):
            collision_to_parent_tf = create_transform_numpy(
                np.eye(3), colmodel.positions[link_name][j]
            )
            collision_to_world_tf = parent_to_world_tf @ collision_to_parent_tf
            sphere_ids.append(
                visualize_3D_sphere(
                    collision_to_world_tf[:3, 3], colmodel.radii[link_name][j]
                )
            )

    input("Press Enter to exit")


if __name__ == "__main__":
    main()
