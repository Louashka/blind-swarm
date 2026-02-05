import re
import mujoco
import mujoco.viewer
import numpy as np
import time
import xml.etree.ElementTree as ET
from shapely.geometry import box
from shapely.ops import unary_union

# ==============================================================================
# === Helper Functions for Geometry Parsing & Point Generation
# ==============================================================================

def get_contour_and_body_z(xml_path, body_name="load"):
    """
    Parses a MuJoCo XML to find a body, recursively finds all its descendant
    box geoms, merges them into a 2D contour, and extracts the body's Z-position.

    This version is corrected to properly handle nested body positions by
    manually building a parent map, avoiding unsupported XPath features.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    parent_map = {c: p for p in root.iter() for c in p}

    target_body = root.find(f".//body[@name='{body_name}']")
    if target_body is None:
        raise ValueError(f"Body '{body_name}' not found in XML.")

    body_pos_str = target_body.get('pos', "0 0 0")
    body_pos_vals = [float(x) for x in body_pos_str.split()]
    body_z = body_pos_vals[2]

    polygons = []
    geom_names = []
    
    for geom in target_body.findall('.//geom'):
        # We only care about geoms that are part of the load, not the hinge visualizer
        if geom.get('type') == 'box' and 'load_geom' in geom.get('name', ''):
            geom_name = geom.get('name')
            geom_names.append(geom_name)
            size = np.array([float(x) for x in geom.get('size').split()])

            geom_pos_relative = np.array([float(x) for x in geom.get('pos', "0 0 0").split()])
            
            current_element = geom
            while current_element in parent_map:
                parent_element = parent_map[current_element]
                
                if parent_element is target_body:
                    break
                
                if parent_element.tag == 'body':
                    parent_pos = np.array([float(x) for x in parent_element.get('pos', "0 0 0").split()])
                    geom_pos_relative += parent_pos
                
                current_element = parent_element

            minx, miny = geom_pos_relative[0] - size[0], geom_pos_relative[1] - size[1]
            maxx, maxy = geom_pos_relative[0] + size[0], geom_pos_relative[1] + size[1]
            polygons.append(box(minx, miny, maxx, maxy))

    if not polygons:
        raise ValueError(f"No 'box' geoms with 'load_geom' in their name found in body '{body_name}' or its children")

    unioned_shape = unary_union(polygons)

    # ==================================================================
    # === CORRECTED: Removed .convex_hull to get the TRUE shape      ===
    # === The convex_hull was creating the "ellipse" effect you saw. ===
    # === By removing it, we now trace the actual L-shape outline.   ===
    # ==================================================================
    merged_shape = unioned_shape
    
    x, y = merged_shape.exterior.xy
    corners = np.column_stack((x, y))[:-1] # Remove duplicate closing point

    return corners, body_z, geom_names

def get_puzzle_geometry(xml_path, load_z_height):
    """
    Parses the XML to find puzzle slit positions and chamber boundaries.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    puzzle_body = root.find(f".//body[@name='puzzle_walls']")
    if puzzle_body is None:
        raise ValueError("Body 'puzzle_walls' not found in XML.")

    puzzle_pos = np.array([float(v) for v in puzzle_body.get('pos', '0 0 0').split()])

    slit1_wall, slit2_wall = None, None
    for geom in puzzle_body.findall('geom'):
        geom_name = geom.get('name', '')
        if 'slit1' in geom_name and slit1_wall is None:
            slit1_wall = geom
        if 'slit2' in geom_name and slit2_wall is None:
            slit2_wall = geom
        if slit1_wall and slit2_wall:
            break

    if slit1_wall is None or slit2_wall is None:
        raise ValueError("Could not find 'slit1' or 'slit2' geoms in XML.")

    slit1_pos_local = np.array([float(v) for v in slit1_wall.get('pos').split()])
    slit2_pos_local = np.array([float(v) for v in slit2_wall.get('pos').split()])

    chamber1_boundary_x = puzzle_pos[0] + slit1_pos_local[0]
    chamber2_boundary_x = puzzle_pos[0] + slit2_pos_local[0]

    slit1_midpoint = np.array([chamber1_boundary_x, 0, load_z_height])
    slit2_midpoint = np.array([chamber2_boundary_x, 0, load_z_height])

    geometry = {
        'chamber1_x_boundary': chamber1_boundary_x,
        'chamber2_x_boundary': chamber2_boundary_x,
        'slit1_midpoint': slit1_midpoint,
        'slit2_midpoint': slit2_midpoint,
    }
    print("Parsed Puzzle Geometry:")
    for key, val in geometry.items():
        print(f"  - {key}: {val}")

    return geometry

def generate_perimeter_points(corners, num_ants, ant_radius, body_z):
    """
    Generates evenly spaced positions around a 2D contour and their corresponding
    local outward-facing normal vectors.
    """
    points = []
    normals = []
    z_ground = -body_z + ant_radius

    closed_corners = np.vstack([corners, corners[0]])
    segment_lengths = [np.linalg.norm(closed_corners[i+1][:2] - closed_corners[i][:2]) for i in range(len(corners))]
    total_perimeter = sum(segment_lengths)

    if total_perimeter < 1e-9:
        return np.array([]), np.array([])

    step_dist = total_perimeter / num_ants
    for i in range(num_ants):
        target_dist = i * step_dist
        accumulated_len = 0
        for seg_idx, seg_len in enumerate(segment_lengths):
            if accumulated_len + seg_len >= target_dist:
                p1, p2 = closed_corners[seg_idx], closed_corners[seg_idx+1]
                edge_vec = p2 - p1

                normal_2d = np.array([-edge_vec[1], edge_vec[0]])
                norm_len = np.linalg.norm(normal_2d)
                if norm_len > 1e-9:
                    normal_2d /= norm_len
                normals.append([normal_2d[0], normal_2d[1], 0.0])

                ratio = (target_dist - accumulated_len) / seg_len if seg_len > 1e-9 else 0.0
                pos_on_line = p1 + edge_vec * ratio

                final_pos = pos_on_line[:2] + (normal_2d * ant_radius)
                points.append([final_pos[0], final_pos[1], z_ground])
                break
            else:
                accumulated_len += seg_len

    return np.array(points), np.array(normals)

def load_model_with_ants(xml_path, ant_positions, ant_radius):
    """
    Loads a MuJoCo model from XML and injects visualization sites for ants.
    """
    with open(xml_path, 'r') as f:
        xml_content = f.read()

    site_str = ""
    for i, pos in enumerate(ant_positions):
        site_str += f'<site name="ant_{i}" pos="{pos[0]} {pos[1]} {pos[2]}" size="{ant_radius}" rgba="0.5 0.5 0.5 0.5"/>\n'

    pattern = r'(<worldbody>)'
    match = re.search(pattern, xml_content)
    if not match:
        raise ValueError("Could not find <worldbody> tag in XML")

    insertion_point = match.end()
    new_xml = xml_content[:insertion_point] + "\n" + site_str + xml_content[insertion_point:]

    return mujoco.MjModel.from_xml_string(new_xml)


# ==============================================================================
# === Main Simulation
# ==============================================================================

def main():
    xml_path = "env_soft.xml"
    ANT_COUNT = 50
    ANT_RADIUS = 0.015

    print(f"Extracting geometry from {xml_path}...")
    try:
        raw_corners_2d, body_z, load_geom_names = get_contour_and_body_z(xml_path, body_name="load")
        print(f"Detected body 'load' at Z-height: {body_z}")
        print(f"Found geoms in assembly: {load_geom_names}")
        puzzle_geometry = get_puzzle_geometry(xml_path, load_z_height=body_z)
    except Exception as e:
        print(f"Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        return

    ant_local_pos, ant_normals_local = generate_perimeter_points(raw_corners_2d, ANT_COUNT, ANT_RADIUS, body_z)
    print(f"Generated {len(ant_local_pos)} ants around the contour.")

    model = load_model_with_ants(xml_path, ant_local_pos, ANT_RADIUS)
    data = mujoco.MjData(model)

    try:
        from swarm import AntSwarm
        swarm = AntSwarm(ANT_COUNT, ant_local_pos, puzzle_geometry)
    except ImportError:
        print("\nWARNING: 'swarm.py' not found. The simulation will run without agent logic.")
        print("The object will be subject to gravity and joint damping only.")
        swarm = None 

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "load")
    load_geom_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in load_geom_names]

    color_detached = np.array([0.5, 0.5, 0.5, 0.3])
    color_informed = np.array([0.8, 0.1, 0.1, 1.0])
    color_puller = np.array([0.1, 0.1, 0.8, 1.0])
    color_lifter = np.array([0.1, 0.8, 0.1, 1.0])

    simulation_finished = False
    forces_global_vis = np.zeros((ANT_COUNT, 3))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat = [1.5, 0.0, 0.0]
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -90.0

        print("\nViewer launched. Pausing for 3 seconds...")
        time.sleep(3)
        print("Resuming simulation...\n")

        while viewer.is_running():
            step_start = time.time()
            
            load_pos = data.xpos[body_id]
            rot_mat = data.xmat[body_id].reshape(3, 3)
            
            lever_arms_global = (rot_mat @ ant_local_pos.T).T
            ant_global_positions = load_pos + lever_arms_global

            if not simulation_finished and swarm is not None:
                all_geom_positions = data.geom_xpos[load_geom_ids]
                min_x_of_assembly = np.min(all_geom_positions[:, 0])
                chamber_3_boundary_x = puzzle_geometry['chamber2_x_boundary']

                if min_x_of_assembly > chamber_3_boundary_x + 0.3:
                    simulation_finished = True
                    print("\nSUCCESS: Object has cleared the final chamber.")
                    data.xfrc_applied[body_id] = 0
                else:
                    ang_vel = data.cvel[body_id][0:3]
                    lin_vel = data.cvel[body_id][3:6]

                    forces_global = swarm.update_logic(
                        dt=model.opt.timestep,
                        load_velocity_global=lin_vel,
                        load_angular_vel=ang_vel,
                        load_rotation_matrix=rot_mat,
                        ant_global_positions=ant_global_positions,
                        ant_local_normals=ant_normals_local
                    )
                    forces_global_vis = forces_global

                    net_force = np.sum(forces_global, axis=0)
                    torques = np.cross(lever_arms_global, forces_global)
                    net_torque = np.sum(torques, axis=0)

                    data.xfrc_applied[body_id][:3] = net_torque
                    data.xfrc_applied[body_id][3:] = net_force

            mujoco.mj_step(model, data)

            for i in range(ANT_COUNT):
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"ant_{i}")
                if site_id != -1:
                    data.site_xpos[site_id] = ant_global_positions[i]
                    if swarm is not None:
                        state = swarm.states[i]
                        if state == 0: model.site_rgba[site_id] = color_detached
                        elif state == 1: model.site_rgba[site_id] = color_informed
                        elif state == 2: model.site_rgba[site_id] = color_puller
                        elif state == 3: model.site_rgba[site_id] = color_lifter

            viewer.user_scn.ngeom = 0
            if not simulation_finished and swarm is not None:
                def add_arrow(pos, vector, color, scale=1.0, radius=0.02):
                    norm = np.linalg.norm(vector)
                    if norm < 1e-6 or viewer.user_scn.ngeom >= viewer.user_scn.maxgeom: return
                    def get_z_alignment_matrix(target_vec):
                        z_new = target_vec / np.linalg.norm(target_vec)
                        ref = np.array([0., 0., 1.]) if abs(z_new[2]) < 0.99 else np.array([0., 1., 0.])
                        x_new = np.cross(ref, z_new); x_new /= np.linalg.norm(x_new)
                        y_new = np.cross(z_new, x_new); y_new /= np.linalg.norm(y_new)
                        return np.column_stack((x_new, y_new, z_new)).flatten()

                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_ARROW,
                        size=[radius, radius, norm * scale],
                        pos=pos, mat=get_z_alignment_matrix(vector), rgba=color
                    )
                    viewer.user_scn.ngeom += 1

                for i in range(ANT_COUNT):
                    if swarm.states[i] != 0:
                        arrow_color = color_informed if swarm.states[i] == 1 else (color_puller if swarm.states[i] == 2 else color_lifter)
                        add_arrow(pos=ant_global_positions[i], vector=forces_global_vis[i],
                                  color=arrow_color, scale=0.8, radius=0.008)

            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()