# main_soft.py

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
# === Helper Functions for Geometry Parsing & Point Generation (unchanged)
# ==============================================================================

def get_contour_and_body_z(xml_path, body_name="load"):
    """
    Parses a MuJoCo XML to find a body, merge its box geoms (including nested bodies)
    into a 2D contour, and extract the body's Z-position.

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
    nested_geoms = [geom for nested_body in target_body.findall("body") for geom in nested_body.findall("geom")]
    all_geoms = target_body.findall("geom") + nested_geoms

    for geom in all_geoms:
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
                t = (target - acc) / slen if slen > 1e-12 else 0.0
                pos_on = p1 + v * t

                left = np.array([-v[1], v[0]], dtype=float)
                right = -left

                def normed(n):
                    nn = float(np.linalg.norm(n))
                    return n / nn if nn > 1e-12 else np.array([1.0, 0.0], dtype=float)

                left_u = normed(left)
                right_u = normed(right)

                test_left = Point(pos_on + eps_test * left_u)
                left_is_inside = poly.contains(test_left)
                n_out = right_u if left_is_inside else left_u

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

    if zero_joint_damping:
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
# === NEW: Rigid attachment helpers (fix “attachment sites keep changing”)
# ==============================================================================

def quat_to_mat33(q):
    """MuJoCo quats are [w, x, y, z]."""
    w, x, y, z = [float(v) for v in q]
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=float)


def geoms_for_body(model, body_id, geom_type=None):
    gids = []
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) != int(body_id):
            continue
        if geom_type is not None and int(model.geom_type[gid]) != int(geom_type):
            continue
        gids.append(gid)
    return np.asarray(gids, dtype=int)


def geom_pose_world(model, data, gid):
    """World pose (xg, Rg) of geom gid."""
    bid = int(model.geom_bodyid[gid])

    xb = data.xpos[bid].copy()
    Rb = data.xmat[bid].reshape(3, 3).copy()

    pg = np.array(model.geom_pos[gid], dtype=float)
    qg = np.array(model.geom_quat[gid], dtype=float)
    Rq = quat_to_mat33(qg)

    Rg = Rb @ Rq
    xg = xb + Rb @ pg
    return xg, Rg


def point_to_box_geom_distance(model, data, gid, p_world):
    """Euclidean distance from point to oriented box surface (0 if inside)."""
    xg, Rg = geom_pose_world(model, data, gid)
    p_g = Rg.T @ (p_world - xg)

    half = np.array(model.geom_size[gid], dtype=float)  # [sx, sy, sz]
    q = np.abs(p_g) - half
    outside = np.maximum(q, 0.0)
    return float(np.linalg.norm(outside))


def choose_attachment_body(model, data, p_world, load_geom_ids, link2_geom_ids):
    """Classify which link a sampled perimeter point belongs to, by nearest box geom."""
    if len(link2_geom_ids) == 0:
        return "load"
    d_load = min(point_to_box_geom_distance(model, data, gid, p_world) for gid in load_geom_ids) if len(load_geom_ids) else float("inf")
    d_l2   = min(point_to_box_geom_distance(model, data, gid, p_world) for gid in link2_geom_ids) if len(link2_geom_ids) else float("inf")
    return "link2" if d_l2 < d_load else "load"


def world_to_body_local(data, body_id, p_world):
    xb = data.xpos[body_id]
    Rb = data.xmat[body_id].reshape(3, 3)
    return Rb.T @ (p_world - xb)


def body_local_to_world(data, body_id, p_local):
    xb = data.xpos[body_id]
    Rb = data.xmat[body_id].reshape(3, 3)
    return xb + Rb @ p_local


def get_subtree_body_ids(model, root_body_id: int) -> np.ndarray:
    """Robust subtree traversal (avoids worldbody parent self-loop)."""
    nbody = model.nbody
    children = [[] for _ in range(nbody)]
    for b in range(1, nbody):  # skip world body 0
        p = int(model.body_parentid[b])
        children[p].append(b)

    out = []
    stack = [int(root_body_id)]
    while stack:
        b = stack.pop()
        out.append(b)
        stack.extend(children[b])
    return np.asarray(out, dtype=int)


def get_subtree_box_geom_ids(model, root_body_id: int) -> np.ndarray:
    subtree = set(get_subtree_body_ids(model, root_body_id).tolist())
    gids = []
    for gid in range(model.ngeom):
        if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        if int(model.geom_bodyid[gid]) in subtree:
            gids.append(gid)
    return np.asarray(gids, dtype=int)


def box_geom_world_corners(model, data, gid):
    """8 corners of a box geom in world coordinates."""
    xg, Rg = geom_pose_world(model, data, gid)
    s = np.array(model.geom_size[gid], dtype=float)  # half extents
    corners_local = np.array([
        [-s[0], -s[1], -s[2]],
        [-s[0], -s[1],  s[2]],
        [-s[0],  s[1], -s[2]],
        [-s[0],  s[1],  s[2]],
        [ s[0], -s[1], -s[2]],
        [ s[0], -s[1],  s[2]],
        [ s[0],  s[1], -s[2]],
        [ s[0],  s[1],  s[2]],
    ], dtype=float)
    return xg[None, :] + (Rg @ corners_local.T).T


# ==============================================================================
# === Main Simulation
# ==============================================================================

def main():
    xml_path = "env_soft.xml"
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

    # Perimeter points are generated ONCE (as before).
    ant_local_pos_vis, ant_local_pos_dyn, ant_normals_local = generate_perimeter_points(
        raw_corners_2d, ANT_COUNT, ANT_RADIUS, body_z
    )
    print(f"Generated {len(ant_local_pos_vis)} ants around the contour.")

    model = load_model_with_ants(xml_path, ant_local_pos_vis, ANT_RADIUS, zero_joint_damping=True)
    data = mujoco.MjData(model)

    # Make sure kinematics are computed before we read xpos/xmat/xipos.
    mujoco.mj_forward(model, data)

    jid_x = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "slide_x")
    jid_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "slide_y")
    jid_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge_z")

    dof_x = model.jnt_dofadr[jid_x]
    dof_y = model.jnt_dofadr[jid_y]
    dof_r = model.jnt_dofadr[jid_r]

    load_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "load")

    # Optional articulated child:
    try:
        link2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link2")
    except Exception:
        link2_id = -1

    # COM offset in the body frame: used by your swarm (unchanged dynamics setup)
    com_local = np.array(model.body_ipos[load_id], dtype=float)
    r_local_com = ant_local_pos_dyn - com_local[None, :]

    swarm = AntSwarm(
        num_sites=ANT_COUNT,
        attachment_sites_local_pos_dyn=r_local_com,     # unchanged (we'll revisit dynamics later)
        attachment_sites_local_normals=ant_normals_local,
        puzzle_geometry=puzzle_geometry,
        wall_aabbs_2d=wall_aabbs_2d,
        eps_wall=EPS_WALL,
    )

    # --- NEW: build fixed attachments that follow the correct link ---
    # We keep the *dynamics application* as you had it (single wrench on 'load'),
    # but we update ant world positions + site positions so they follow link2 when needed.

    load_box_geoms = geoms_for_body(model, load_id, geom_type=mujoco.mjtGeom.mjGEOM_BOX)
    link2_box_geoms = geoms_for_body(model, link2_id, geom_type=mujoco.mjtGeom.mjGEOM_BOX) if link2_id != -1 else np.zeros((0,), dtype=int)

    # For success check (optional but more correct with articulation)
    subtree_box_geoms = get_subtree_box_geom_ids(model, load_id)

    # Fixed attachment storage
    ant_bodyid = np.full((ANT_COUNT,), load_id, dtype=int)
    ant_pos_local_vis_body = np.zeros((ANT_COUNT, 3), dtype=float)
    ant_pos_local_dyn_body = np.zeros((ANT_COUNT, 3), dtype=float)
    ant_nrm_local_body = np.zeros((ANT_COUNT, 3), dtype=float)

    # Use current load pose to convert initial load-frame samples to world,
    # then classify each sample to 'load' vs 'link2', then store in that body's local frame.
    xL = data.xpos[load_id].copy()
    RL = data.xmat[load_id].reshape(3, 3).copy()

    for i in range(ANT_COUNT):
        pL_vis = ant_local_pos_vis[i]
        pL_dyn = ant_local_pos_dyn[i]
        nL = ant_normals_local[i]

        pW_dyn = xL + RL @ pL_dyn
        pW_vis = xL + RL @ pL_vis
        nW = RL @ nL

        which = choose_attachment_body(model, data, pW_dyn, load_box_geoms, link2_box_geoms)
        bid = link2_id if (which == "link2" and link2_id != -1) else load_id
        ant_bodyid[i] = bid

        ant_pos_local_dyn_body[i] = world_to_body_local(data, bid, pW_dyn)
        ant_pos_local_vis_body[i] = world_to_body_local(data, bid, pW_vis)

        # store the normal in that body's local frame (so it rotates with the link)
        Rb = data.xmat[bid].reshape(3, 3)
        ant_nrm_local_body[i] = Rb.T @ nW

    print("Attachment classification:")
    if link2_id != -1:
        n_link2 = int(np.sum(ant_bodyid == link2_id))
        print(f"  - {n_link2}/{ANT_COUNT} ants attached to link2, {ANT_COUNT - n_link2} to load")
    else:
        print("  - link2 not found; all ants attached to load")

    # Colors
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

        # Allocate arrays once (avoid per-step realloc)
        ant_global_positions = np.zeros((ANT_COUNT, 3), dtype=float)  # used by swarm + arrows
        ant_world_dyn = np.zeros((ANT_COUNT, 3), dtype=float)         # used for torque lever arms

        while viewer.is_running():
            step_start = time.time()

            # --------------------------------------
            # Body pose (load only; unchanged inputs to swarm)
            # --------------------------------------
            body_pos = data.xpos[load_id].copy()
            rot_mat = data.xmat[load_id].reshape(3, 3).copy()

            # --------------------------------------
            # NEW: Update ant positions so they follow their attached link
            # --------------------------------------
            for i in range(ANT_COUNT):
                bid = int(ant_bodyid[i])
                ant_global_positions[i] = body_local_to_world(data, bid, ant_pos_local_vis_body[i])
                ant_world_dyn[i] = body_local_to_world(data, bid, ant_pos_local_dyn_body[i])

            # Keep MuJoCo sites (which live under body="load") synced for visualization.
            # Convert world position -> load local each step.
            for i in range(ANT_COUNT):
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"ant_{i}")
                pW = ant_global_positions[i]
                pL_vis = world_to_body_local(data, load_id, pW)
                model.site_pos[site_id] = pL_vis

            # --------------------------------------
            # Success check
            # NOTE: your old raw_corners_3d transform assumes the entire assembly is rigid.
            # If link2 articulates, this is inaccurate. We switch to current geom corners.
            # --------------------------------------
            if len(subtree_box_geoms) > 0:
                min_x_of_assembly = float("inf")
                for gid in subtree_box_geoms:
                    corners = box_geom_world_corners(model, data, int(gid))
                    min_x_of_assembly = min(min_x_of_assembly, float(np.min(corners[:, 0])))
                # also include ant points (usually redundant but harmless)
                min_x_of_assembly = min(min_x_of_assembly, float(np.min(ant_global_positions[:, 0])))
            else:
                # fallback
                global_corners = body_pos + (rot_mat @ raw_corners_3d.T).T
                all_points_to_check = np.vstack((ant_global_positions, global_corners))
                min_x_of_assembly = float(np.min(all_points_to_check[:, 0]))

            chamber_3_boundary_x = float(puzzle_geometry["chamber2_x_boundary"])
            if (not simulation_finished) and (min_x_of_assembly > chamber_3_boundary_x + 0.3):
                simulation_finished = True
                print("\nSUCCESS: Object has cleared the final chamber.")

            # --------------------------------------
            # Swarm forces + damping (KEEP DYNAMICS AS YOU HAD IT)
            # --------------------------------------
            ang_vel_world = data.cvel[load_id][0:3].copy()
            lin_vel_world = data.cvel[load_id][3:6].copy()

            # Clear any previous applied wrench (load only, unchanged)
            data.xfrc_applied[load_id][:] = 0.0

            if not simulation_finished:
                forces_world = swarm.update_logic(
                    dt=model.opt.timestep,
                    load_velocity_world=lin_vel_world,
                    load_angular_vel_world=ang_vel_world,
                    load_rotation_matrix=rot_mat,
                    ant_global_positions=ant_global_positions,
                )

                # Add dynamic damping (unchanged)
                model.dof_damping[dof_x] = swarm.load.linear_damping
                model.dof_damping[dof_y] = swarm.load.linear_damping
                model.dof_damping[dof_r] = swarm.load.angular_damping

                # Net ant force and torque about LOAD COM
                net_force = np.sum(forces_world, axis=0)

                load_com_world = data.xipos[load_id].copy()
                r_world_about_load_com = ant_world_dyn - load_com_world[None, :]
                net_torque = np.sum(np.cross(r_world_about_load_com, forces_world), axis=0)

                # Apply to MuJoCo (world frame wrench on LOAD ONLY, unchanged)
                data.xfrc_applied[load_id][0:3] = net_force
                data.xfrc_applied[load_id][3:6] = net_torque

            mujoco.mj_step(model, data)

            # --------------------------------------
            # Update visuals (colors)
            # --------------------------------------
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

            # --------------------------------------
            # Force Arrow Visualization (unchanged, but now uses updated ant positions)
            # --------------------------------------
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

                for i in range(ANT_COUNT):
                    if swarm.states[i] in (1, 2):
                        arrow_color = color_informed if swarm.states[i] == 1 else color_puller
                        add_arrow(
                            pos=ant_global_positions[i],
                            vector=forces_world[i],
                            color=arrow_color,
                            scale=0.2,
                            radius=0.008,
                        )

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()