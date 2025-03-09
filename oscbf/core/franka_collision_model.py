"""Creating a link collision model for the Franka with a series of spheres of various radii"""

# All positions are in link frame, not link COM frame

link_1_pos = (
    (0, 0, -0.15),
    (0, -0.065, 0),
)
link_1_radii = (
    0.06,
    0.06,
)

link_2_pos = (
    (0, 0, 0.065),
    (0, -0.14, 0),
)
link_2_radii = (
    0.06,
    0.06,
)

link_3_pos = (
    (0, 0, -0.065),
    (0.08, 0.065, 0),
)
link_3_radii = (
    0.06,
    0.055,
)

link_4_pos = (
    (0, 0, 0.065),
    (-0.08, 0.08, 0),
)
link_4_radii = (
    0.055,
    0.055,
)

link_5_pos = (
    (0, 0, -0.23),
    (0, 0.06, -0.18),
    (0, 0.08, -0.125),
    (0, 0.09, -0.075),
    (0, 0.08, 0),
)
link_5_radii = (
    0.06,
    0.04,
    0.025,
    0.025,
    0.055,
)

link_6_pos = (
    (0, 0, 0.01),
    (0.08, 0.035, 0),
    # (0.08, -0.02, 0),
)
link_6_radii = (
    0.05,
    0.05,
    # 0.05,
)

link_7_pos = (
    (0, 0, 0.08),
    (0.04, 0.04, 0.09),
    (0.055, 0.055, 0.15),
    (-0.055, -0.055, 0.15),
    (-0.055, -0.055, 0.11),
    (0, 0, 0.20),
)
link_7_radii = (
    0.05,
    0.04,
    0.03,
    0.03,
    0.03,
    0.02,
)

# TODO: Get rid of these dictionaries and just use lists
positions = {
    "link_1": link_1_pos,
    "link_2": link_2_pos,
    "link_3": link_3_pos,
    "link_4": link_4_pos,
    "link_5": link_5_pos,
    "link_6": link_6_pos,
    "link_7": link_7_pos,
}
radii = {
    "link_1": link_1_radii,
    "link_2": link_2_radii,
    "link_3": link_3_radii,
    "link_4": link_4_radii,
    "link_5": link_5_radii,
    "link_6": link_6_radii,
    "link_7": link_7_radii,
}

positions_list = (
    link_1_pos,
    link_2_pos,
    link_3_pos,
    link_4_pos,
    link_5_pos,
    link_6_pos,
    link_7_pos,
)
radii_list = (
    link_1_radii,
    link_2_radii,
    link_3_radii,
    link_4_radii,
    link_5_radii,
    link_6_radii,
    link_7_radii,
)

franka_collision_data = {"positions": positions_list, "radii": radii_list}
