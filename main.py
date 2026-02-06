import re
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union

from swarm import AntSwarm


# ==============================================================================
# === Helper Functions for Geometry Parsing & Point Generation
# ==============================================================================

def get_contour_and_body_z(xml_path, body_name="load"):
    """
    Parses a MuJoCo XML to find a body, merge its box geoms into a 2D contour,
    and extract the body's Z-position.

    Returns:
        tuple: (corners_2d, body_z)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    target_body = root.find(f".//body[@name='{body_name}']")
    if target_body is None:
        raise ValueError(f"Body '{body_name}' not found in XML.")

    body_pos_str = target_body.get("pos", "0 0 0")
    body_pos_vals = [float(x) for x in body_pos_str.split()]
    body_z = body_pos_vals[2]

    polygons = []
    for geom in target_body.findall("geom"):
        if geom.get("type") == "box":
            size = np.array([float(x) for x in geom.get("size").split()], dtype=float)
            pos = np.array([float(x) for x in geom.get("pos", "0 0 0").split()], dtype=float)
            minx, miny = pos[0] - size[0], pos[1] - size[1]
            maxx, maxy = pos[0] + size[0], pos[1] + size[1]
            polygons.append(box(minx, miny, maxx, maxy))

    if not polygons:
        raise ValueError(f"No 'box' geoms found in body '{body_name}'")

    merged_shape = unary_union(polygons)
    if merged_shape.geom_type != "Polygon":
        # If union returns MultiPolygon, take exterior of the largest part
        merged_shape = max(list(merged_shape.geoms), key=lambda g: g.area)

    x, y = merged_shape.exterior.xy
    corners = np.column_stack((x, y))[:-1]  # remove duplicate closing point
    return corners, body_z


def get_puzzle_geometry(xml_path, load_z_height):
    """
    Parses the XML to find puzzle slit positions and chamber boundaries.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    puzzle_body = root.find(".//body[@name='puzzle_walls']")
    if puzzle_body is None:
        raise ValueError("Body 'puzzle_walls' not found in XML.")

    puzzle_pos = np.array([float(v) for v in puzzle_body.get("pos", "0 0 0").split()], dtype=float)

    slit1_wall, slit2_wall = None, None
    for geom in puzzle_body.findall("geom"):
        geom_name = geom.get("name", "")
        if "slit1" in geom_name and slit1_wall is None:
            slit1_wall = geom
        if "slit2" in geom_name and slit2_wall is None:
            slit2_wall = geom
        if slit1_wall is not None and slit2_wall is not None:
            break

    if slit1_wall is None or slit2_wall is None:
        raise ValueError("Could not find 'slit1' or 'slit2' geoms in XML.")

    slit1_pos_local = np.array([float(v) for v in slit1_wall.get("pos").split()], dtype=float)
    slit2_pos_local = np.array([float(v) for v in slit2_wall.get("pos").split()], dtype=float)

    chamber1_boundary_x = puzzle_pos[0] + slit1_pos_local[0]
    chamber2_boundary_x = puzzle_pos[0] + slit2_pos_local[0]

    slit1_midpoint = np.array([chamber1_boundary_x, 0.0, load_z_height], dtype=float)
    slit2_midpoint = np.array([chamber2_boundary_x, 0.0, load_z_height], dtype=float)

    geometry = {
        "chamber1_x_boundary": chamber1_boundary_x,
        "chamber2_x_boundary": chamber2_boundary_x,
        "slit1_midpoint": slit1_midpoint,
        "slit2_midpoint": slit2_midpoint,
    }

    print("Parsed Puzzle Geometry:")
    for key, val in geometry.items():
        print(f"  - {key}: {val}")

    return geometry


def get_wall_aabbs_2d(xml_path, body_name="puzzle_walls"):
    """
    Returns a list of 2D AABBs (minx, miny, maxx, maxy) for all box geoms in puzzle_walls.
    Used for epsilon wall-proximity detachment/attachment blocking.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    puzzle_body = root.find(f".//body[@name='{body_name}']")
    if puzzle_body is None:
        raise ValueError(f"Body '{body_name}' not found in XML.")

    body_pos = np.array([float(v) for v in puzzle_body.get("pos", "0 0 0").split()], dtype=float)

    aabbs = []
    for geom in puzzle_body.findall("geom"):
        if geom.get("type") != "box":
            continue
        size = np.array([float(x) for x in geom.get("size").split()], dtype=float)
        pos = np.array([float(x) for x in geom.get("pos", "0 0 0").split()], dtype=float)
        world_pos = body_pos + pos
        minx, miny = world_pos[0] - size[0], world_pos[1] - size[1]
        maxx, maxy = world_pos[0] + size[0], world_pos[1] + size[1]
        aabbs.append((float(minx), float(miny), float(maxx), float(maxy)))

    return aabbs


def generate_perimeter_points(corners_2d, num_ants, ant_radius, body_z):
    """
    Generates evenly spaced positions around a 2D contour and their corresponding
    local outward-facing normal vectors.

    Returns:
        ant_local_pos_vis: (N,3) local positions with z chosen for visualization (ground)
        ant_local_pos_dyn: (N,3) local positions planar z=0 for dynamics (torque/vel)
        ant_normals_local: (N,3) local outward normals (planar)
    """
    corners_2d = np.asarray(corners_2d, dtype=float)
    poly = Polygon(corners_2d)
    if not poly.is_valid:
        poly = poly.buffer(0)

    z_ground_local = -float(body_z) + float(ant_radius)

    points_vis = []
    points_dyn = []
    normals = []

    closed = np.vstack([corners_2d, corners_2d[0]])
    seg_vecs = closed[1:] - closed[:-1]
    seg_lengths = np.linalg.norm(seg_vecs, axis=1)
    total = float(np.sum(seg_lengths))
    if total < 1e-12:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))

    step = total / float(num_ants)
    eps_test = max(1e-6, ant_radius * 0.1)

    for i in range(num_ants):
        target = i * step
        acc = 0.0
        for sidx, slen in enumerate(seg_lengths):
            if acc + slen >= target:
                p1 = closed[sidx]
                v = seg_vecs[sidx]
                if slen > 1e-12:
                    t = (target - acc) / slen
                else:
                    t = 0.0
                pos_on = p1 + v * t

                # Candidate normals (left and right)
                left = np.array([-v[1], v[0]], dtype=float)
                right = -left

                def normed(n):
                    nn = float(np.linalg.norm(n))
                    return n / nn if nn > 1e-12 else np.array([1.0, 0.0], dtype=float)

                left_u = normed(left)
                right_u = normed(right)

                # Choose outward by testing a point slightly along the normal
                test_left = Point(pos_on + eps_test * left_u)
                left_is_inside = poly.contains(test_left)

                # If left points inside, outward is right; else outward is left
                n_out = right_u if left_is_inside else left_u

                # Offset ant position by ant radius along outward normal
                pos_xy = pos_on + n_out * ant_radius

                points_vis.append([pos_xy[0], pos_xy[1], z_ground_local])
                points_dyn.append([pos_xy[0], pos_xy[1], 0.0])

                normals.append([n_out[0], n_out[1], 0.0])
                break
            acc += float(slen)

    return np.asarray(points_vis, dtype=float), np.asarray(points_dyn, dtype=float), np.asarray(normals, dtype=float)


def load_model_with_ants(xml_path, ant_positions_local_vis, ant_radius, zero_joint_damping=True):
    """
    Loads a MuJoCo model from XML and injects visualization sites for ants.
    Optionally zeroes the planar joint dampings (recommended; we add damping in Python).
    """
    with open(xml_path, "r") as f:
        xml_content = f.read()

    # Optionally zero joint damping so we don't double-count damping
    if zero_joint_damping:
        # Replace damping="..." on the three known joints; keep it simple and safe.
        xml_content = re.sub(r'(<joint[^>]*name="slide_x"[^>]*damping=")[^"]*(")',
                             r"\g<1>0\2", xml_content)
        xml_content = re.sub(r'(<joint[^>]*name="slide_y"[^>]*damping=")[^"]*(")',
                             r"\g<1>0\2", xml_content)
        xml_content = re.sub(r'(<joint[^>]*name="hinge_z"[^>]*damping=")[^"]*(")',
                             r"\g<1>0\2", xml_content)

    site_str = ""
    for i, pos in enumerate(ant_positions_local_vis):
        site_str += (
            f'<site name="ant_{i}" pos="{pos[0]} {pos[1]} {pos[2]}" '
            f'size="{ant_radius}" rgba="0.5 0.5 0.5 0.5"/>\n'
        )

    pattern = r'(<body[^>]*name="load"[^>]*>)'
    match = re.search(pattern, xml_content)
    if not match:
        raise ValueError("Could not find body 'load' in XML")

    insertion_point = match.end()
    new_xml = xml_content[:insertion_point] + "\n" + site_str + xml_content[insertion_point:]

    return mujoco.MjModel.from_xml_string(new_xml)


# ==============================================================================
# === Main Simulation
# ==============================================================================

def main():
    xml_path = "env.xml"
    ANT_COUNT = 50
    ANT_RADIUS = 0.015

    # Paper epsilon is 0.5 mm. If you scaled your maze up, consider scaling this too.
    EPS_WALL = 0.0005

    print(f"Extracting geometry from {xml_path}...")
    try:
        raw_corners_2d, body_z = get_contour_and_body_z(xml_path, body_name="load")
        raw_corners_3d = np.hstack([raw_corners_2d, np.zeros((raw_corners_2d.shape[0], 1))])
        print(f"Detected body 'load' at Z-height: {body_z}")

        puzzle_geometry = get_puzzle_geometry(xml_path, load_z_height=body_z)
        wall_aabbs_2d = get_wall_aabbs_2d(xml_path, body_name="puzzle_walls")
        print(f"Parsed {len(wall_aabbs_2d)} wall AABBs for epsilon detachment.")
    except Exception as e:
        print(f"Error parsing XML geometry: {e}")
        return

    ant_local_pos_vis, ant_local_pos_dyn, ant_normals_local = generate_perimeter_points(
        raw_corners_2d, ANT_COUNT, ANT_RADIUS, body_z
    )
    print(f"Generated {len(ant_local_pos_vis)} ants around the contour.")

    model = load_model_with_ants(xml_path, ant_local_pos_vis, ANT_RADIUS, zero_joint_damping=True)
    data = mujoco.MjData(model)

    jid_x = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "slide_x")
    jid_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "slide_y")
    jid_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge_z")

    dof_x = model.jnt_dofadr[jid_x]
    dof_y = model.jnt_dofadr[jid_y]
    dof_r = model.jnt_dofadr[jid_r]

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "load")

    # COM offset in the body frame: used to convert site positions to COM-relative lever arms
    com_local = np.array(model.body_ipos[body_id], dtype=float)  # COM position expressed in body frame

    # Convert dynamic attachment points to COM-relative local vectors
    r_local_com = ant_local_pos_dyn - com_local[None, :]

    swarm = AntSwarm(
        num_sites=ANT_COUNT,
        attachment_sites_local_pos_dyn=r_local_com,     # COM-relative for correct v_loc and torques
        attachment_sites_local_normals=ant_normals_local,
        puzzle_geometry=puzzle_geometry,
        wall_aabbs_2d=wall_aabbs_2d,
        eps_wall=EPS_WALL,
    )

    color_detached = np.array([0.5, 0.5, 0.5, 0.3])
    color_informed = np.array([0.8, 0.1, 0.1, 1.0])
    color_puller = np.array([0.1, 0.1, 0.8, 1.0])
    color_lifter = np.array([0.1, 0.8, 0.1, 1.0])

    simulation_finished = False

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat = [1.5, 0.0, 0.0]
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -90.0

        print("\nViewer launched.")
        print("Pausing for 3 seconds...")
        time.sleep(3)
        print("Resuming simulation...\n")

        while viewer.is_running():
            step_start = time.time()

            # Body pose
            body_pos = data.xpos[body_id].copy()  # body frame origin in world
            rot_mat = data.xmat[body_id].reshape(3, 3).copy()
            com_world = data.xipos[body_id].copy()  # COM position in world

            # World positions of ant sites for chamber logic / wall epsilon checks / visualization
            ant_global_positions = body_pos + (rot_mat @ ant_local_pos_vis.T).T

            # World lever arms about COM for dynamics (planar, COM-relative)
            r_world = (rot_mat @ r_local_com.T).T  # (N,3)

            # --- Success check ---
            global_corners = body_pos + (rot_mat @ raw_corners_3d.T).T
            all_points_to_check = np.vstack((ant_global_positions, global_corners))
            min_x_of_assembly = float(np.min(all_points_to_check[:, 0]))
            chamber_3_boundary_x = float(puzzle_geometry["chamber2_x_boundary"])

            if (not simulation_finished) and (min_x_of_assembly > chamber_3_boundary_x + 0.3):
                simulation_finished = True
                print("\nSUCCESS: Object has cleared the final chamber.")

            # --- Swarm forces + overdamped damping (paper-consistent) ---
            # Use COM-based velocities from cvel (com-based 6D velocity).
            # If you later want, we can switch to a safer COM finite-difference velocity.
            ang_vel_world = data.cvel[body_id][0:3].copy()
            lin_vel_world = data.cvel[body_id][3:6].copy()

            # Clear any previous applied wrench
            data.xfrc_applied[body_id][:] = 0.0

            if not simulation_finished:
                forces_world = swarm.update_logic(
                    dt=model.opt.timestep,
                    load_velocity_world=lin_vel_world,
                    load_angular_vel_world=ang_vel_world,
                    load_rotation_matrix=rot_mat,
                    ant_global_positions=ant_global_positions,
                )
                
                # Add dynamic damping 
                model.dof_damping[dof_x] = swarm.load.linear_damping
                model.dof_damping[dof_y] = swarm.load.linear_damping
                model.dof_damping[dof_r] = swarm.load.angular_damping

                # Net ant force and torque about COM
                net_force = np.sum(forces_world, axis=0)
                torques = np.cross(r_world, forces_world)
                net_torque = np.sum(torques, axis=0)

                # Apply to MuJoCo (world frame wrench on body)
                data.xfrc_applied[body_id][0:3] = net_force
                data.xfrc_applied[body_id][3:6] = net_torque

            mujoco.mj_step(model, data)

            # --- Update Visuals (sites are fixed in body frame; only colors change) ---
            for i in range(ANT_COUNT):
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"ant_{i}")
                state = int(swarm.states[i])
                if state == 0:
                    model.site_rgba[site_id] = color_detached
                elif state == 1:
                    model.site_rgba[site_id] = color_informed
                elif state == 2:
                    model.site_rgba[site_id] = color_puller
                elif state == 3:
                    model.site_rgba[site_id] = color_lifter

            # --- Force Arrow Visualization ---
            viewer.user_scn.ngeom = 0

            if not simulation_finished:
                def add_arrow(pos, vector, color, scale=1.0, radius=0.02):
                    norm = float(np.linalg.norm(vector))
                    if norm < 1e-6 or viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                        return

                    def get_z_alignment_matrix(target_vec):
                        z_new = target_vec / np.linalg.norm(target_vec)
                        ref = np.array([0.0, 0.0, 1.0]) if abs(z_new[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
                        x_new = np.cross(ref, z_new)
                        x_new /= np.linalg.norm(x_new)
                        y_new = np.cross(z_new, x_new)
                        y_new /= np.linalg.norm(y_new)
                        return np.column_stack((x_new, y_new, z_new)).flatten()

                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_ARROW,
                        size=[radius, radius, norm * scale],
                        pos=pos,
                        mat=get_z_alignment_matrix(vector),
                        rgba=color,
                    )
                    viewer.user_scn.ngeom += 1

                # Draw arrows only for active pullers/informed
                # (forces_world is only defined in this branch)
                for i in range(ANT_COUNT):
                    if swarm.states[i] in (1, 2):
                        arrow_color = color_informed if swarm.states[i] == 1 else color_puller
                        add_arrow(
                            pos=ant_global_positions[i],
                            vector=forces_world[i],
                            color=arrow_color,
                            scale=0.8,
                            radius=0.008,
                        )

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()